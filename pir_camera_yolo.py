#!/usr/bin/env python3
from gpiozero import MotionSensor, LED
from signal import pause
from time import sleep, monotonic
from datetime import datetime
from pathlib import Path
import threading, queue, subprocess, shutil

# ==== ПАРАМЕТРЫ ====
PIR_GPIO = 4
LED_PINS = [17, 27, 22]      # [red, blue, green]
BASE_DIR = Path("/home/pi/camera")
TMP_DIR = BASE_DIR / "tmp"
CATS_DIR = BASE_DIR / "cats"
NOT_CAT_DIR = BASE_DIR / "not_cat"

COOLDOWN_SEC = 5
WARMUP_PIR_SEC = 30
PHOTO_RES = "1536x864"       # ширина x высота
CAPTURE_RETRIES = 2          # сколько раз повторять, если libcamera вернул ошибку
RETRY_DELAY = 0.4            # пауза между повторами

# ==== ПАПКИ ====
for d in [BASE_DIR, TMP_DIR, CATS_DIR, NOT_CAT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==== PIR + LEDs ====
pir = MotionSensor(PIR_GPIO)
leds = [LED(p) for p in LED_PINS]
red_led, blue_led, green_led = leds

def all_leds_on():
    for l in leds: l.on()

def all_leds_off():
    for l in leds: l.off()

def blink(led, times=2, ms=120):
    for _ in range(times):
        led.on(); sleep(ms/1000); led.off(); sleep(ms/1000)

# ==== YOLOv4-tiny (OpenCV DNN) ====
import cv2, numpy as np
YOLO_CFG     = "/home/pi/models/yolo-tiny/yolov4-tiny.cfg"
YOLO_WEIGHTS = "/home/pi/models/yolo-tiny/yolov4-tiny.weights"
CONF_THRES = 0.25            # можно 0.20–0.35
NMS_THRES  = 0.45
INPUT_SIZE = 416             # 320 быстрее, 608 точнее
CAT_CLASS_ID = 15            # COCO: 15 == "cat" (нумерация с 0)

# мини-словарь для читаемых логов (без coco.names)
ID2NAME = {0: "person", 15: "cat"}

net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# имена слоёв → выходные слои (оставляем подчёркивания)
_layer_names = net.getLayerNames()
_out_layers = [_layer_names[i - 1] for i in np.atleast_1d(net.getUnconnectedOutLayers()).flatten()]

# разогрев сети (убирает «первый долгий/пустой кадр»)
_warm = cv2.dnn.blobFromImage(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8),
                              1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
net.setInput(_warm)
_ = net.forward(_out_layers)

def _nms_boxes(boxes, confs, conf_thres, nms_thres):
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_thres, nms_thres)
    if idxs is None or len(idxs) == 0:
        return []
    return np.array(idxs).flatten().tolist()

def detect_is_cat_cv(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        print("[yolo] cv2.imread вернул None")
        return (False, 0.0, [])

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(_out_layers)

    boxes, confs, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf > CONF_THRES:
                cx, cy, bw, bh = det[:4]
                bw_px = int(bw * w)
                bh_px = int(bh * h)
                x = int(cx * w - bw_px / 2)
                y = int(cy * h - bh_px / 2)
                boxes.append([x, y, bw_px, bh_px])
                confs.append(conf)
                class_ids.append(cid)

    # быстрый лог
    dets = [(confs[i], class_ids[i], boxes[i]) for i in range(len(boxes))]
    dets.sort(reverse=True, key=lambda x: x[0])
    for c, cls, box in dets[:3]:
        print(f"[yolo] class_id={cls:<2} conf={c:.2f} box={box}")

    # собираем котов
    best_cat = 0.0
    yolo_lines = []
    for i in range(len(boxes)):
        if class_ids[i] == CAT_CLASS_ID:
            best_cat = max(best_cat, confs[i])
            x, y, ww, hh = boxes[i]
            # нормализуем в 0..1 под YOLO
            x_center = (x + ww / 2) / w
            y_center = (y + hh / 2) / h
            norm_w   = ww / w
            norm_h   = hh / h
            yolo_lines.append(f"{CAT_CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    print(f"[yolo] cat_conf={best_cat:.2f}, kept={len(boxes)}, raw={len(boxes)}")
    return (best_cat > 0.0, best_cat, yolo_lines)



# ==== сериализация событий ====
work_q = queue.Queue(maxsize=1)
_last_shot = 0.0
ts_lock = threading.Lock()

def enqueue_motion():
    global _last_shot
    now = monotonic()
    with ts_lock:
        if now - _last_shot < COOLDOWN_SEC:
            return
        _last_shot = now
    try:
        work_q.put_nowait(True)
    except queue.Full:
        pass

def capture_with_libcamera(dst_path: Path):
    w, h = PHOTO_RES.split("x")
    cmd = ["libcamera-jpeg", "-n", "--immediate", "--width", w, "--height", h, "-o", str(dst_path)]
    for attempt in range(1, CAPTURE_RETRIES + 2):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            return
        except subprocess.CalledProcessError:
            if attempt <= CAPTURE_RETRIES:
                sleep(RETRY_DELAY)
            else:
                raise

def worker():
    while True:
        work_q.get()
        try:
            blue_led.on()
            now = datetime.now()
            ts = now.strftime("%d.%m.%Y_%H-%M-%S.") + f"{int(now.microsecond / 1000):03d}"
            tmp_path = TMP_DIR / f"motion_{ts}.jpg"

            capture_with_libcamera(tmp_path)
            print(f"[shot] saved: {tmp_path}")

            is_cat, best, yolo_lines = detect_is_cat_cv(str(tmp_path))

            blue_led.off()

            if is_cat:
                dst = CATS_DIR / f"cat_{ts}_{best:.2f}.jpg"
                shutil.move(str(tmp_path), str(dst))
                if yolo_lines:
                    with open(dst.with_suffix(".txt"), "w", encoding="utf-8") as f:
                        f.write("\n".join(yolo_lines))
                    print(f"[yolo] labels saved: {dst.with_suffix('.txt')}")
                blink(green_led, times=3)
                print(f"[save] -> {dst}")
            else:
                dst = NOT_CAT_DIR / f"not_cat_{ts}.jpg"
                shutil.move(str(tmp_path), str(dst))
                blink(red_led, times=3)
                print(f"[save] -> {dst} (not a cat)")

        except Exception as e:
            print(f"[err] {e}")
        finally:
            blue_led.off()
            work_q.task_done()


# ==== callbacks PIR ====
def on_motion():
    print("[motion] движение")
    enqueue_motion()

def on_no_motion():
    print("[idle] нет движения")

print(f"[init] прогрев PIR ~{WARMUP_PIR_SEC}s…")
sleep(WARMUP_PIR_SEC)

threading.Thread(target=worker, daemon=True).start()
pir.when_motion = on_motion
pir.when_no_motion = on_no_motion

print(f"[run] жду движение… (cooldown={COOLDOWN_SEC}s, save dir={BASE_DIR})")
pause()
