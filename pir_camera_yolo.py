#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gpiozero import MotionSensor, LED
from signal import pause
from time import sleep, monotonic
from datetime import datetime, timezone
from pathlib import Path
import threading, queue, subprocess, shutil, os, urllib.parse

# ===========================
# ПАРАМЕТРЫ СЪЁМКИ И ПАПКИ
# ===========================
PIR_GPIO = 4
LED_PINS = [17, 27, 22]      # [red, blue, green]
BASE_DIR = Path.home() / "camera"
TMP_DIR = BASE_DIR / "tmp"
CATS_DIR = BASE_DIR / "cats"
NOT_CAT_DIR = BASE_DIR / "not_cat"

COOLDOWN_SEC = 1
WARMUP_PIR_SEC = 10
PHOTO_RES = "4608x2592"       # ширина x высота
CAPTURE_RETRIES = 2          # сколько раз повторять, если libcamera вернул ошибку
RETRY_DELAY = 0.4            # пауза между повторами

for d in [BASE_DIR, TMP_DIR, CATS_DIR, NOT_CAT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===========================
# PIR + LEDs
# ===========================
pir = MotionSensor(PIR_GPIO)
leds = [LED(p) for p in LED_PINS]
red_led, blue_led, green_led = leds

def blink(led, times=2, ms=120):
    for _ in range(times):
        led.on(); sleep(ms/1000); led.off(); sleep(ms/1000)

# ===========================
# MINIO (дефолты как в Java)
# Можно переопределять через ENV
# ===========================
def _parse_wh(s: str, default=(384,384)):
    try:
        w,h = s.lower().split("x")
        return (int(w), int(h))
    except Exception:
        return default

# из Java:
# minio.storage.endpoint: http://localhost:9000
# minio.storage.login:    minioadmin
# minio.storage.password: minioadmin
# minio.photosBucket:     photo
# minio.thumbsBucket:     thumbs
# coordinationBucket:     coordination
MINIO_ENDPOINT_RAW = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY   = os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_LOGIN", "minioadmin"))
MINIO_SECRET_KEY   = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_PASSWORD", "minioadmin"))

def _normalize_endpoint(ep_raw: str):
    try:
        if ep_raw.startswith("http://") or ep_raw.startswith("https://"):
            u = urllib.parse.urlparse(ep_raw)
            hostport = u.netloc or u.path
            secure = (u.scheme == "https")
            return hostport, secure
    except Exception:
        pass
    return ep_raw, os.getenv("MINIO_SECURE", "false").lower() == "true"

MINIO_ENDPOINT, MINIO_SECURE = _normalize_endpoint(MINIO_ENDPOINT_RAW)

# бакеты по умолчанию как в Java
MINIO_PHOTOS_BUCKET       = os.getenv("MINIO_PHOTOS_BUCKET", "photo")
MINIO_THUMBS_BUCKET       = os.getenv("MINIO_THUMBS_BUCKET", "thumbs")
MINIO_COORDINATION_BUCKET = os.getenv("MINIO_COORDINATION_BUCKET", "coordination")

# общий префикс ключей (обычно пусто при раздельных бакетах)
MINIO_PREFIX = os.getenv("MINIO_PREFIX", "").strip().strip("/")

# размер миниатюры (максимум по большей стороне)
THUMB_SIZE = _parse_wh(os.getenv("THUMB_SIZE", "384x384"), (384,384))

_minio_client = None
_minio_ok = False

def _init_minio():
    global _minio_client, _minio_ok
    try:
        from minio import Minio
        _minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        # проверить/создать бакеты
        for bucket in {MINIO_PHOTOS_BUCKET, MINIO_THUMBS_BUCKET, MINIO_COORDINATION_BUCKET}:
            try:
                if not _minio_client.bucket_exists(bucket):
                    _minio_client.make_bucket(bucket)
                    print(f"[minio] created bucket '{bucket}'")
            except Exception as e:
                print(f"[minio] bucket '{bucket}' check/create: {e}")
        _minio_ok = True
        print(f"[minio] endpoint={MINIO_ENDPOINT} secure={MINIO_SECURE} | "
              f"photosBucket={MINIO_PHOTOS_BUCKET} thumbsBucket={MINIO_THUMBS_BUCKET} coordinationBucket={MINIO_COORDINATION_BUCKET}")
    except Exception as e:
        _minio_ok = False
        print(f"[minio] init error: {e}")

def _minio_key(*parts):
    parts = [p.strip("/").replace("\\","/") for p in parts if p]
    if MINIO_PREFIX:
        parts = [MINIO_PREFIX] + parts
    return "/".join(parts)

def _upload_file(local_path: Path, bucket: str, key: str,
                 content_type: str = "application/octet-stream",
                 metadata: dict | None = None):
    if not _minio_ok:
        return False
    try:
        extra = {f"x-amz-meta-{k}": str(v) for k, v in (metadata or {}).items()}
        _minio_client.fput_object(
            bucket, key, str(local_path),
            content_type=content_type,
            metadata=(extra or None)
        )
        print(f"[minio] uploaded s3://{bucket}/{key}")
        return True
    except Exception as e:
        print(f"[minio] upload error: {local_path} -> {bucket}/{key}: {e}")
        return False

def _make_thumbnail(src: Path, dst: Path, max_wh=(384,384)):
    try:
        from PIL import Image
    except Exception as e:
        print(f"[thumb] Pillow не установлен, пропуск миниатюры: {e}")
        return False
    try:
        with Image.open(src) as im:
            im.thumbnail(max_wh)
            im.save(dst, format="JPEG", quality=85, optimize=True)
        print(f"[thumb] saved: {dst}")
        return True
    except Exception as e:
        print(f"[thumb] error: {e}")
        return False

# ===========================
# YOLOv4-tiny (OpenCV DNN)
# ===========================
import cv2, numpy as np
YOLO_CFG     = str(Path.home() / "models/yolo-tiny/yolov4-tiny.cfg")
YOLO_WEIGHTS = str(Path.home() / "models/yolo-tiny/yolov4-tiny.weights")
CONF_THRES = 0.25
NMS_THRES  = 0.45
INPUT_SIZE = 416
CAT_CLASS_ID = 15            # COCO: 15 == "cat"

_layer_names = None
_out_layers = None
net = None

def _yolo_init():
    global _layer_names, _out_layers, net
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    _layer_names = net.getLayerNames()
    _out_layers = [_layer_names[i - 1] for i in np.atleast_1d(net.getUnconnectedOutLayers()).flatten()]
    # разогрев
    _warm = cv2.dnn.blobFromImage(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8),
                                  1/255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(_warm)
    _ = net.forward(_out_layers)

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

    best_cat = 0.0
    yolo_lines = []
    for i in range(len(boxes)):
        if class_ids[i] == CAT_CLASS_ID:
            best_cat = max(best_cat, confs[i])
            x, y, ww, hh = boxes[i]
            x_center = (x + ww / 2) / w
            y_center = (y + hh / 2) / h
            norm_w   = ww / w
            norm_h   = hh / h
            yolo_lines.append(f"{CAT_CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    print(f"[yolo] cat_conf={best_cat:.2f}, boxes={len(boxes)}")
    return (best_cat > 0.0, best_cat, yolo_lines)

# ===========================
# ОЧЕРЕДЬ РАБОТЫ
# ===========================
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

def capture_with_camera(dst_path: Path, rotation=0):
    """
    rpicam-still -> libcamera-still -> libcamera-jpeg
    """
    cmd = None
    for c in ("rpicam-still", "libcamera-still", "libcamera-jpeg"):
        if shutil.which(c):
            cmd = c
            break
    if not cmd:
        raise FileNotFoundError("Нет rpicam-still/libcamera-still/libcamera-jpeg. "
                                "Установи: sudo apt install -y rpicam-apps || libcamera-apps")

    w, h = PHOTO_RES.split("x")
    args = [cmd, "-n", "--immediate", "--width", w, "--height", h, "-o", str(dst_path)]
    if rotation:
        args[1:1] = ["--rotation", str(rotation)]

    for attempt in range(1, CAPTURE_RETRIES + 2):
        try:
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            if attempt <= CAPTURE_RETRIES:
                sleep(RETRY_DELAY)
            else:
                raise

def _save_and_upload(final_path: Path, is_cat: bool, best_conf: float, yolo_lines, ts_iso: str):
    """
    Делает миниатюру, сохраняет .txt метки (если есть) и заливает всё в MinIO в бакеты:
    - photo: оригиналы
    - thumbs: миниатюры
    - coordination: yolo .txt (basename)
    """
    # 1) миниатюра
    thumb_path = final_path.with_name(final_path.stem + "_thumb.jpg")
    _make_thumbnail(final_path, thumb_path, THUMB_SIZE)

    # 2) YOLO labels .txt (если есть)
    labels_path = None
    if yolo_lines:
        labels_path = final_path.with_suffix(".txt")
        try:
            with open(labels_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
            print(f"[yolo] labels saved: {labels_path}")
        except Exception as e:
            print(f"[yolo] не удалось сохранить labels: {e}")
            labels_path = None

    if not _minio_ok:
        return

    split = "cats" if is_cat else "not_cat"
    # раздельные бакеты → без верхних 'photos/' и 'thumbs/' в ключах
    date = datetime.fromisoformat(ts_iso.replace("Z","")).date()
    y = f"{date.year:04d}"; m = f"{date.month:02d}"; d = f"{date.day:02d}"

    photo_key = _minio_key(split, y, m, d, final_path.name)
    thumb_key = _minio_key(split, y, m, d, thumb_path.name)
    # coordination.txt кладём в отдельный бакет с базовым именем
    coord_key = (final_path.stem + ".txt") if labels_path else None

    meta = {
        "is_cat": str(is_cat).lower(),
        "confidence": f"{best_conf:.2f}",
        "ts": ts_iso,
        "source": "raspberrypi"
    }

    _upload_file(final_path, MINIO_PHOTOS_BUCKET, photo_key, content_type="image/jpeg", metadata=meta)
    if thumb_path.exists():
        _upload_file(thumb_path, MINIO_THUMBS_BUCKET, thumb_key, content_type="image/jpeg", metadata=meta)
    if coord_key and labels_path and labels_path.exists():
        _upload_file(labels_path, MINIO_COORDINATION_BUCKET, coord_key, content_type="text/plain", metadata=meta)

def worker():
    while True:
        work_q.get()
        try:
            blue_led.on()
            now = datetime.now()
            ts = now.strftime("%d.%m.%Y_%H-%M-%S.") + f"{int(now.microsecond / 1000):03d}"
            ts_iso = now.isoformat(timespec="milliseconds") + "Z"
            tmp_path = TMP_DIR / f"motion_{ts}.jpg"

            capture_with_camera(tmp_path)
            print(f"[shot] saved: {tmp_path}")

            is_cat, best, yolo_lines = detect_is_cat_cv(str(tmp_path))
            blue_led.off()

            if is_cat:
                dst = CATS_DIR / f"cat_{ts}_{best:.2f}.jpg"
                shutil.move(str(tmp_path), str(dst))
                print(f"[save] -> {dst}")
                blink(green_led, times=3)
                _save_and_upload(dst, True, best, yolo_lines, ts_iso)
            else:
                dst = NOT_CAT_DIR / f"not_cat_{ts}.jpg"
                shutil.move(str(tmp_path), str(dst))
                print(f"[save] -> {dst} (not a cat)")
                blink(red_led, times=3)
                _save_and_upload(dst, False, 0.0, [], ts_iso)
        except Exception as e:
            print(f"[err] {e}")
        finally:
            blue_led.off()
            work_q.task_done()
# ===========================
# STARTUP
# ===========================
print(f"[init] прогрев PIR ~{WARMUP_PIR_SEC}s…")
sleep(WARMUP_PIR_SEC)

# Инициализация YOLO и MinIO
_yolo_init()
_init_minio()

threading.Thread(target=worker, daemon=True).start()
pir.when_motion = lambda: (print("[motion] движение"), enqueue_motion())
pir.when_no_motion = lambda: print("[idle] нет движения")

print(f"[run] жду движение… (cooldown={COOLDOWN_SEC}s, save dir={BASE_DIR}) | "
      f"MinIO: {MINIO_ENDPOINT_RAW} -> {MINIO_ENDPOINT}, secure={MINIO_SECURE}")
pause()
