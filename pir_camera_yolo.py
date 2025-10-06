#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gpiozero import MotionSensor, LED
from signal import pause
from time import sleep, monotonic
from datetime import datetime, timezone
from pathlib import Path
import threading, queue, subprocess, shutil, os, urllib.parse
import re

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
PHOTO_RES = "4608x2592"   # ширина x высота (поддерживает '×')
CAPTURE_RETRIES = 2
RETRY_DELAY = 0.4

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
# ===========================
def _parse_wh(s: str, default=(384,384)):
    try:
        s = s.strip().lower().replace("×", "x")
        m = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", s)
        if not m: return default
        return (int(m.group(1)), int(m.group(2)))
    except Exception:
        return default

MINIO_ENDPOINT_RAW = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY   = os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_LOGIN", "minioadmin"))
MINIO_SECRET_KEY   = os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_PASSWORD", "minioadmin"))

def _normalize_endpoint(ep_raw: str):
    try:
        if ep_raw.startswith(("http://","https://")):
            u = urllib.parse.urlparse(ep_raw)
            hostport = u.netloc or u.path
            secure = (u.scheme == "https")
            return hostport, secure
    except Exception:
        pass
    return ep_raw, os.getenv("MINIO_SECURE", "false").lower() == "true"

MINIO_ENDPOINT, MINIO_SECURE = _normalize_endpoint(MINIO_ENDPOINT_RAW)

MINIO_PHOTOS_BUCKET       = os.getenv("MINIO_PHOTOS_BUCKET", "photo")
MINIO_THUMBS_BUCKET       = os.getenv("MINIO_THUMBS_BUCKET", "thumbs")
MINIO_COORDINATION_BUCKET = os.getenv("MINIO_COORDINATION_BUCKET", "coordination")
MINIO_PREFIX = os.getenv("MINIO_PREFIX", "").strip().strip("/")
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
        print(f"[minio] put -> bucket={bucket}, key={key}, file={local_path}")
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
# ДЕТЕКТОРЫ
# ===========================
import cv2, numpy as np

# ---- YOLOv8n ONNX (через OpenCV DNN) ----
DETECTOR = os.getenv("DETECTOR", "v8-onnx").strip().lower()  # v8-onnx | v4-tiny
V8_ONNX_PATH = os.getenv("V8_ONNX", str(Path.home() / "models/yolov8n.onnx"))
V8_INPUT = int(os.getenv("V8_INPUT", "640"))
V8_CONF_THRES = float(os.getenv("V8_CONF_THRES", "0.25"))
V8_IOU_THRES  = float(os.getenv("V8_IOU_THRES", "0.45"))
CAT_CLASS_ID = 15  # COCO 'cat'

_net_v8 = None
def _v8_onnx_init():
    global _net_v8
    _net_v8 = cv2.dnn.readNetFromONNX(V8_ONNX_PATH)
    _net_v8.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    _net_v8.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # прогрев
    dummy = np.zeros((V8_INPUT, V8_INPUT, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(dummy, 1/255.0, (V8_INPUT, V8_INPUT), swapRB=True, crop=False)
    _net_v8.setInput(blob)
    _ = _net_v8.forward()

def _v8_parse_output(out, img_w, img_h):
    """
    Универсальный парсер под два популярных формата экспорта Ultralytics:
    1) (1, 84, N)  -> транспонируем до (N,84)
    2) (1, N, 84)  -> вытаскиваем (N,84)
    Каждая строка: [x, y, w, h, obj, cls0..cls79] в относительных координатах.
    Возвращает списки boxes(x,y,w,h в пикселях), scores(=obj*cls), class_ids.
    """
    arr = out
    if isinstance(arr, (list, tuple)):
        arr = arr[0]
    arr = np.array(arr)
    # squeeze до 3D
    arr = np.squeeze(arr)
    # приведение к (N, 84)
    if arr.ndim == 2:
        if arr.shape[0] == 84:          # (84, N) -> (N,84)
            arr = arr.transpose(1, 0)
        elif arr.shape[1] == 84:        # (N,84) ок
            pass
        else:
            raise RuntimeError(f"Unexpected YOLOv8 ONNX output shape {arr.shape}")
    else:
        raise RuntimeError(f"Unexpected YOLOv8 ONNX output ndim={arr.ndim}, shape={arr.shape}")

    xywh = arr[:, :4]
    obj  = arr[:, 4:5]
    cls  = arr[:, 5:]                   # (N, 80)

    # лучший класс и его вероятность
    cls_ids = np.argmax(cls, axis=1)
    cls_scores = cls[np.arange(cls.shape[0]), cls_ids]
    scores = (cls_scores * obj[:, 0])   # принятой формулой: conf = obj * cls_conf

    # фильтр уверенности
    keep = scores > V8_CONF_THRES
    if not np.any(keep):
        return [], [], []

    xywh = xywh[keep]; scores = scores[keep]; cls_ids = cls_ids[keep]

    # перевод в xyxy (в пикселях)
    x = xywh[:, 0] * img_w
    y = xywh[:, 1] * img_h
    w = xywh[:, 2] * img_w
    h = xywh[:, 3] * img_h
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).astype(np.float32)  # x,y,w,h
    return boxes.tolist(), scores.tolist(), cls_ids.tolist()

def detect_is_cat_v8_onnx(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        print("[v8-onnx] cv2.imread None")
        return (False, 0.0, [])
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (V8_INPUT, V8_INPUT), swapRB=True, crop=False)
    _net_v8.setInput(blob)
    out = _net_v8.forward()
    try:
        boxes, scores, class_ids = _v8_parse_output(out, w, h)
    except Exception as e:
        print(f"[v8-onnx] parse error: {e}")
        return (False, 0.0, [])

    # NMS по всем классам
    if boxes:
        idxs = cv2.dnn.NMSBoxes(boxes, scores, V8_CONF_THRES, V8_IOU_THRES)
        keep = set(int(i) for i in np.atleast_1d(idxs).flatten()) if len(idxs) else set()
    else:
        keep = set()

    best_cat = 0.0
    yolo_lines = []
    for i in range(len(boxes)):
        if i not in keep:
            continue
        if class_ids[i] == CAT_CLASS_ID:
            conf = float(scores[i])
            best_cat = max(best_cat, conf)
            x, y, ww, hh = boxes[i]
            # в YOLO txt нужно нормализовать (cx, cy, w, h)
            x_center = (x + ww / 2) / w
            y_center = (y + hh / 2) / h
            norm_w   = ww / w
            norm_h   = hh / h
            yolo_lines.append(f"{CAT_CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    print(f"[v8-onnx] cat_conf={best_cat:.2f}, boxes={len(keep)}")
    return (best_cat > 0.0, best_cat, yolo_lines)

# ---- YOLOv4-tiny (резервный) ----
YOLO_CFG     = str(Path.home() / "models/yolo-tiny/yolov4-tiny.cfg")
YOLO_WEIGHTS = str(Path.home() / "models/yolo-tiny/yolov4-tiny.weights")
CONF_THRES_TINY = 0.25
NMS_THRES_TINY  = 0.45
INPUT_SIZE_TINY = 416
_layer_names = None
_out_layers = None
_net_tiny = None

def _yolo_tiny_init():
    global _layer_names, _out_layers, _net_tiny
    _net_tiny = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    _net_tiny.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    _net_tiny.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    _layer_names = _net_tiny.getLayerNames()
    _out_layers = [_layer_names[i - 1] for i in np.atleast_1d(_net_tiny.getUnconnectedOutLayers()).flatten()]
    _warm = cv2.dnn.blobFromImage(np.zeros((INPUT_SIZE_TINY, INPUT_SIZE_TINY, 3), dtype=np.uint8),
                                  1/255.0, (INPUT_SIZE_TINY, INPUT_SIZE_TINY), swapRB=True, crop=False)
    _net_tiny.setInput(_warm); _ = _net_tiny.forward(_out_layers)

def detect_is_cat_tiny(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        print("[tiny] cv2.imread вернул None")
        return (False, 0.0, [])

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (INPUT_SIZE_TINY, INPUT_SIZE_TINY), swapRB=True, crop=False)
    _net_tiny.setInput(blob)
    outs = _net_tiny.forward(_out_layers)

    boxes, confs, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf > CONF_THRES_TINY:
                cx, cy, bw, bh = det[:4]
                bw_px = int(bw * w)
                bh_px = int(bh * h)
                x = int(cx * w - bw_px / 2)
                y = int(cy * h - bh_px / 2)
                boxes.append([x, y, bw_px, bh_px])
                confs.append(conf)
                class_ids.append(cid)

    idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRES_TINY, NMS_THRES_TINY)
    keep = set(int(i) for i in np.atleast_1d(idxs).flatten()) if len(idxs) else set()

    best_cat = 0.0
    yolo_lines = []
    for i in range(len(boxes)):
        if i not in keep:
            continue
        if class_ids[i] == CAT_CLASS_ID:
            best_cat = max(best_cat, confs[i])
            x, y, ww, hh = boxes[i]
            x_center = (x + ww / 2) / w
            y_center = (y + hh / 2) / h
            norm_w   = ww / w
            norm_h   = hh / h
            yolo_lines.append(f"{CAT_CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    print(f"[tiny] cat_conf={best_cat:.2f}, boxes={len(keep)}")
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

def _parse_res(s: str):
    if not isinstance(s, str):
        raise ValueError(f"PHOTO_RES должен быть строкой, а не {type(s)}")
    s = s.strip().lower().replace("×", "x")
    m = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", s)
    if not m:
        raise ValueError(f"Неверный PHOTO_RES='{s}', ожидаю 'WIDTHxHEIGHT', напр. '4608x2592'")
    return int(m.group(1)), int(m.group(2))

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

    w, h = _parse_res(PHOTO_RES)
    args = [cmd, "-n", "--immediate", "--width", str(w), "--height", str(h), "-o", str(dst_path)]
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
    Делает миниатюру, сохраняет .txt метки (если есть) и заливает в MinIO:
    - photo: оригиналы
    - thumbs: миниатюры
    - coordination: yolo .txt (basename)
    """
    thumb_path = final_path.with_name(final_path.stem + "_thumb.jpg")
    _make_thumbnail(final_path, thumb_path, THUMB_SIZE)

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
        print("[minio] OFF: загрузка только на локальный диск")
        return

    split = "cats" if is_cat else "not_cat"
    date = datetime.fromisoformat(ts_iso.replace("Z","")).date()
    y = f"{date.year:04d}"; m = f"{date.month:02d}"; d = f"{date.day:02d}"

    photo_key = _minio_key(split, y, m, d, final_path.name)
    thumb_key = _minio_key(split, y, m, d, thumb_path.name)
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

            if DETECTOR == "v8-onnx":
                is_cat, best, yolo_lines = detect_is_cat_v8_onnx(str(tmp_path))
            else:
                is_cat, best, yolo_lines = detect_is_cat_tiny(str(tmp_path))
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

# Инициализация детектора и MinIO
try:
    if DETECTOR == "v8-onnx":
        _v8_onnx_init()
    elif DETECTOR == "v4-tiny":
        _yolo_tiny_init()
    else:
        raise ValueError(f"Unknown DETECTOR={DETECTOR}")
except Exception as e:
    print(f"[detector] init error: {e} — fallback to YOLOv4-tiny")
    DETECTOR = "v4-tiny"
    _yolo_tiny_init()

_init_minio()

threading.Thread(target=worker, daemon=True).start()
pir.when_motion = lambda: (print("[motion] движение"), enqueue_motion())
pir.when_no_motion = lambda: print("[idle] нет движения")

print(f"[run] жду движение… (cooldown={COOLDOWN_SEC}s, save dir={BASE_DIR}) | "
      f"PHOTO_RES={PHOTO_RES} | MinIO: {MINIO_ENDPOINT_RAW} -> {MINIO_ENDPOINT}, secure={MINIO_SECURE} | "
      f"detector={'YOLOv8n-ONNX' if DETECTOR=='v8-onnx' else 'YOLOv4-tiny'}")
pause()
