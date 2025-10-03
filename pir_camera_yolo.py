#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi: детектор движения + снимок + детекция кота (YOLOv4-tiny через OpenCV DNN).
Сохраняет файлы локально И грузит их в MinIO. Именование фото/меток совместимо с проектом gallery:
один и тот же basename для .jpg и .txt без префиксов, например:
  cats/2025-10-03_22-41-12.357.jpg
  cats/2025-10-03_22-41-12.357.txt

MinIO-ключи (object names):
  photos/cats/<basename>.jpg
  labels/cats/<basename>.txt
  photos/not_cat/<basename>.jpg

Если MinIO не сконфигурирован — локальная запись остаётся как раньше.
Зависимости: opencv-python, numpy, minio, gpiozero
"""

# ===== Импорты =====
from __future__ import annotations
from gpiozero import MotionSensor, LED
from signal import pause
from time import sleep, monotonic
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
import threading, queue, subprocess, shutil
import os

# Опционально: OpenCV и numpy
import cv2, numpy as np

# ===== Параметры (при необходимости меняйте под себя/ENV) =====
PIR_GPIO = int(os.getenv("PIR_GPIO", "4"))
LED_PINS = [int(x) for x in os.getenv("LED_PINS", "17,27,22").split(",")]  # [red, blue, green]

BASE_DIR = Path(os.getenv("BASE_DIR", "/home/vgzakirov/camera"))
TMP_DIR = BASE_DIR / "tmp"
CATS_DIR = BASE_DIR / "cats"
NOT_CAT_DIR = BASE_DIR / "not_cat"

COOLDOWN_SEC = float(os.getenv("COOLDOWN_SEC", "5"))
WARMUP_PIR_SEC = float(os.getenv("WARMUP_PIR_SEC", "30"))
PHOTO_RES = os.getenv("PHOTO_RES", "1536x864")  # "widthxheight"
CAPTURE_RETRIES = int(os.getenv("CAPTURE_RETRIES", "2"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "0.4"))

# Именование файлов, совместимое с gallery
# Базовое имя БЕЗ расширения. По умолчанию только таймстамп.
# Доступные плейсхолдеры: {ts} — YYYY-MM-DD_HH-mm-ss.SSS, {kind} — cats|not_cat, {conf} — 0.00
NAME_TEMPLATE = os.getenv("NAME_TEMPLATE", "{ts}")

# ==== Миниатюры (thumbnails) ====
THUMBS_DIR = BASE_DIR / "thumbs"
THUMBS_CATS_DIR = THUMBS_DIR / "cats"
THUMBS_NOTCAT_DIR = THUMBS_DIR / "not_cat"
THUMB_MAX_EDGE = int(os.getenv("THUMB_MAX_EDGE", "512"))  # максимальная сторона в пикселях
THUMB_QUALITY = int(os.getenv("THUMB_QUALITY", "85"))     # jpeg качество 1..100
MINIO_BUCKET_THUMBS = os.getenv("MINIO_BUCKET_THUMBS", "thumbs").strip()  # если пусто, используем MINIO_BUCKET_PHOTOS

# ===== Папки (создадим, если нет) =====
for d in [BASE_DIR, TMP_DIR, CATS_DIR, NOT_CAT_DIR, THUMBS_DIR, THUMBS_CATS_DIR, THUMBS_NOTCAT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===== PIR + LEDs =====
pir = MotionSensor(PIR_GPIO)
leds = [LED(p) for p in LED_PINS]
red_led, blue_led, green_led = leds


def all_leds_on():
    for l in leds:
        l.on()


def all_leds_off():
    for l in leds:
        l.off()


def blink(led: LED, times: int = 2, ms: int = 120):
    for _ in range(times):
        led.on(); sleep(ms / 1000); led.off(); sleep(ms / 1000)


# ===== YOLOv4-tiny (OpenCV DNN) =====
YOLO_CFG = os.getenv("YOLO_CFG", "/home/vgzakirov/models/yolo-tiny/yolov4-tiny.cfg")
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "/home/vgzakirov/models/yolo-tiny/yolov4-tiny.weights")
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
NMS_THRES = float(os.getenv("NMS_THRES", "0.45"))
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "416"))
CAT_CLASS_ID = int(os.getenv("CAT_CLASS_ID", "15"))  # COCO: 15 — cat

# Инициализация YOLO сети
net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
_layer_names = net.getLayerNames()
_out_layers = [_layer_names[i - 1] for i in np.atleast_1d(net.getUnconnectedOutLayers()).flatten()]

# тёплый прогон
_warm = cv2.dnn.blobFromImage(
    np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8), 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
)
net.setInput(_warm)
_ = net.forward(_out_layers)


def _nms_boxes(boxes: List[List[int]], confs: List[float], conf_thres: float, nms_thres: float) -> List[int]:
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_thres, nms_thres)
    if idxs is None or len(idxs) == 0:
        return []
    return np.array(idxs).flatten().tolist()


def detect_is_cat_cv(img_path: str) -> Tuple[bool, float, List[str]]:
    img = cv2.imread(img_path)
    if img is None:
        print("[yolo] cv2.imread вернул None")
        return False, 0.0, []

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
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

    # NMS (опционально)
    keep = _nms_boxes(boxes, confs, CONF_THRES, NMS_THRES)
    if keep:
        boxes = [boxes[i] for i in keep]
        confs = [confs[i] for i in keep]
        class_ids = [class_ids[i] for i in keep]

    best_cat = 0.0
    yolo_lines: List[str] = []
    for i in range(len(boxes)):
        if class_ids[i] == CAT_CLASS_ID:
            best_cat = max(best_cat, confs[i])
            x, y, ww, hh = boxes[i]
            x_center = (x + ww / 2) / w
            y_center = (y + hh / 2) / h
            norm_w = ww / w
            norm_h = hh / h
            yolo_lines.append(f"{CAT_CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return best_cat > 0.0, best_cat, yolo_lines


# ===== MinIO интеграция =====
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000").strip()        # например: 127.0.0.1:9000
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin").strip()
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin").strip()
MINIO_SECURE = bool(int(os.getenv("MINIO_SECURE", "0")))        # 0 или 1
MINIO_BUCKET_PHOTOS = os.getenv("MINIO_BUCKET_PHOTOS", "photo")
MINIO_BUCKET_LABELS = os.getenv("MINIO_BUCKET_LABELS", "coordination")
MINIO_PREFIX = os.getenv("MINIO_PREFIX", "").strip().strip("/")  # напр. "pi1"

MINIO_ENABLED = all([MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY])
_minio_client = None


def _init_minio():
    global _minio_client
    if not MINIO_ENABLED:
        return None
    try:
        from minio import Minio
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        # создать бакеты при необходимости
        for bucket in [MINIO_BUCKET_PHOTOS, MINIO_BUCKET_LABELS]:
            try:
                if not client.bucket_exists(bucket):
                    client.make_bucket(bucket)
            except Exception as e:
                print(f"[minio] make_bucket({bucket}) -> {e}")
        _minio_client = client
        print(f"[minio] ready: endpoint={MINIO_ENDPOINT}, secure={MINIO_SECURE}, prefix='{MINIO_PREFIX}'")
        return client
    except Exception as e:
        print(f"[minio] disabled due to error: {e}")
        return None


def _minio_object_name(kind: str, filename: str) -> str:
    # kind может быть: "photos/cats", "labels/cats", "photos/not_cat"
    parts = [p for p in [MINIO_PREFIX, kind, filename] if p]
    return "/".join(parts)


def upload_minio(local_path: Path, bucket: str, object_name: str, content_type: Optional[str]):
    if _minio_client is None:
        return
    try:
        _minio_client.fput_object(bucket, object_name, str(local_path), content_type=content_type)
        print(f"[minio] uploaded: s3://{bucket}/{object_name}")
    except Exception as e:
        print(f"[minio] upload failed for {local_path} -> {bucket}/{object_name}: {e}")


# ===== Именование под gallery =====

def make_timestamp(now: datetime) -> str:
    # YYYY-MM-DD_HH-mm-ss.SSS
    return now.strftime("%Y-%m-%d_%H-%M-%S.") + f"{int(now.microsecond/1000):03d}"


def make_basename(ts: str, kind: str, conf: Optional[float]) -> str:
    safe_conf = f"{conf:.2f}" if conf is not None else ""
    return NAME_TEMPLATE.format(ts=ts, kind=kind, conf=safe_conf)


def minio_object_paths(kind: str, basename: str) -> tuple[str, str]:
    photo_key = _minio_object_name(f"photos/{kind}", f"{basename}.jpg")
    label_key = _minio_object_name(f"labels/{kind}", f"{basename}.txt")
    return photo_key, label_key

def minio_thumb_path(kind: str, basename: str) -> str:
    return _minio_object_name(f"thumbs/{kind}", f"{basename}.jpg")


# ==== Thumbnails utils ====
def create_thumbnail(src: Path, dst: Path, max_edge: int = THUMB_MAX_EDGE, quality: int = THUMB_QUALITY):
    try:
        img = cv2.imread(str(src))
        if img is None:
            print(f"[thumb] cannot read {src}")
            return
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= max_edge:
            resized = img
        else:
            scale = max_edge / float(m)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        print(f"[thumb] saved: {dst}")
    except Exception as e:
        print(f"[thumb] error: {e}")

# ===== Очередь событий/воркер =====
work_q: "queue.Queue[bool]" = queue.Queue(maxsize=1)
_last_shot = 0.0
_ts_lock = threading.Lock()


def enqueue_motion():
    global _last_shot
    now = monotonic()
    with _ts_lock:
        if now - _last_shot < COOLDOWN_SEC:
            return
        _last_shot = now
    try:
        work_q.put_nowait(True)
    except queue.Full:
        pass


def capture_with_libcamera(dst_path: Path):
    w, h = PHOTO_RES.split("x")
    cmd = [
        "libcamera-jpeg",
        "-n",
        "--immediate",
        "--width",
        w,
        "--height",
        h,
        "-o",
        str(dst_path),
    ]
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
            ts_str = make_timestamp(now)

            # снимаем во временный файл
            tmp_path = TMP_DIR / f"motion_{ts_str}.jpg"
            capture_with_libcamera(tmp_path)
            print(f"[shot] saved: {tmp_path}")

            is_cat, best, yolo_lines = detect_is_cat_cv(str(tmp_path))
            blue_led.off()

            # ленивый init MinIO
            if _minio_client is None and MINIO_ENABLED:
                _init_minio()

            if is_cat:
                kind = "cats"
                basename = make_basename(ts_str, kind, best)
                dst = CATS_DIR / f"{basename}.jpg"
                shutil.move(str(tmp_path), str(dst))

                # локально координаты
                if yolo_lines:
                    label_path = dst.with_suffix(".txt")
                    with open(label_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(yolo_lines))

                # MinIO
                photo_key, label_key = minio_object_paths(kind, basename)
                upload_minio(dst, MINIO_BUCKET_PHOTOS, photo_key, content_type="image/jpeg")
                if yolo_lines:
                    upload_minio(label_path, MINIO_BUCKET_LABELS, label_key, content_type="text/plain")
                # Thumbnail (локально + MinIO)
                thumb_dst = THUMBS_CATS_DIR / f"{basename}.jpg"
                create_thumbnail(dst, thumb_dst)
                thumb_key = minio_thumb_path(kind, basename)
                upload_minio(thumb_dst, (MINIO_BUCKET_THUMBS or MINIO_BUCKET_PHOTOS), thumb_key, content_type="image/jpeg")

                blink(green_led, times=3)
                print(f"[save] cat -> {dst.name} (minio: {photo_key})")
            else:
                kind = "not_cat"
                basename = make_basename(ts_str, kind, None)
                dst = NOT_CAT_DIR / f"{basename}.jpg"
                shutil.move(str(tmp_path), str(dst))

                # MinIO (у not_cat меток нет)
                photo_key, _ = minio_object_paths(kind, basename)
                upload_minio(dst, MINIO_BUCKET_PHOTOS, photo_key, content_type="image/jpeg")
                # Thumbnail (локально + MinIO)
                thumb_dst = THUMBS_NOTCAT_DIR / f"{basename}.jpg"
                create_thumbnail(dst, thumb_dst)
                thumb_key = minio_thumb_path(kind, basename)
                upload_minio(thumb_dst, (MINIO_BUCKET_THUMBS or MINIO_BUCKET_PHOTOS), thumb_key, content_type="image/jpeg")

                blink(red_led, times=3)
                print(f"[save] not_cat -> {dst.name} (minio: {photo_key})")

        except Exception as e:
            print(f"[err] {e}")
        finally:
            blue_led.off()
            work_q.task_done()


# ===== Коллбэки PIR =====

def on_motion():
    print("[motion] движение")
    enqueue_motion()


def on_no_motion():
    print("[idle] нет движения")


# ===== Запуск =====
print(f"[init] прогрев PIR ~{int(WARMUP_PIR_SEC)}s…")
sleep(WARMUP_PIR_SEC)

threading.Thread(target=worker, daemon=True).start()
pir.when_motion = on_motion
pir.when_no_motion = on_no_motion

print(f"[run] жду движение… (cooldown={COOLDOWN_SEC}s, save dir={BASE_DIR})")

# Можно инициализировать MinIO сразу (необязательно)
if MINIO_ENABLED:
    _init_minio()
else:
    print("[minio] not configured; set MINIO_* env vars to enable uploads")

pause()
