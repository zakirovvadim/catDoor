#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path
from signal import pause
from time import monotonic, sleep
import os
import queue
import shutil
import subprocess
import threading

import cv2
import numpy as np
from gpiozero import LED, MotionSensor

# ==== CONFIG ====
PIR_GPIO = 4
LED_PINS = [17, 27, 22]  # [red, blue, green]
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "darknet").strip().lower()

BASE_DIR = Path.home() / "camera"
TMP_DIR = BASE_DIR / "tmp"
CATS_DIR = BASE_DIR / "cats"
NOT_CAT_DIR = BASE_DIR / "not_cat"

COOLDOWN_SEC = 5
WARMUP_PIR_SEC = 30
PHOTO_RES = os.getenv("PHOTO_RES", "1536x864")
CAPTURE_RETRIES = 2
RETRY_DELAY = 0.4

YOLO_CFG = os.getenv("YOLO_CFG", str(Path.home() / "models/yolo-tiny/yolov4-tiny.cfg"))
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", str(Path.home() / "models/yolo-tiny/yolov4-tiny.weights"))
CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
NMS_THRES = float(os.getenv("NMS_THRES", "0.45"))
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "416"))
CAT_CLASS_ID = int(os.getenv("CAT_CLASS_ID", "15"))  # COCO class id for "cat"

TFLITE_MODEL = os.getenv("TFLITE_MODEL", str(Path.home() / "models/tflite/model.tflite"))
TFLITE_SCORE_THRES = float(os.getenv("TFLITE_SCORE_THRES", "0.35"))
TFLITE_CLASS_OFFSET = int(os.getenv("TFLITE_CLASS_OFFSET", "0"))
TFLITE_THREADS = int(os.getenv("TFLITE_THREADS", "1"))
TFLITE_FLOAT_INPUT_NORM = os.getenv("TFLITE_FLOAT_INPUT_NORM", "zero_one").strip().lower()


for directory in [BASE_DIR, TMP_DIR, CATS_DIR, NOT_CAT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

pir = MotionSensor(PIR_GPIO)
leds = [LED(pin) for pin in LED_PINS]
red_led, blue_led, green_led = leds

net = None
out_layers = []
tflite_interpreter = None
tflite_input_detail = None
tflite_output_details = []

work_q = queue.Queue(maxsize=1)
_last_shot = 0.0
ts_lock = threading.Lock()


def blink(led: LED, times: int = 2, ms: int = 120) -> None:
    for _ in range(times):
        led.on()
        sleep(ms / 1000)
        led.off()
        sleep(ms / 1000)


def _parse_res(value: str) -> tuple[int, int]:
    normalized = value.strip().lower()
    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError(f"PHOTO_RES must be WIDTHxHEIGHT, got: '{value}'")

    w_txt, h_txt = (part.strip() for part in parts)
    if not (w_txt.isdigit() and h_txt.isdigit()):
        raise ValueError(f"PHOTO_RES must contain numbers, got: '{value}'")

    width = int(w_txt)
    height = int(h_txt)
    if width <= 0 or height <= 0:
        raise ValueError(f"PHOTO_RES must be positive, got: '{value}'")

    return width, height


def _init_yolo() -> None:
    global net, out_layers

    cfg_path = Path(YOLO_CFG)
    weights_path = Path(YOLO_WEIGHTS)
    missing = []
    if not cfg_path.exists():
        missing.append(str(cfg_path))
    if not weights_path.exists():
        missing.append(str(weights_path))
    if missing:
        raise FileNotFoundError(f"Missing YOLO files: {', '.join(missing)}")

    net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in np.atleast_1d(net.getUnconnectedOutLayers()).flatten()]

    # Warm up once so the first real inference is not extra slow.
    warm = cv2.dnn.blobFromImage(
        np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8),
        1 / 255.0,
        (INPUT_SIZE, INPUT_SIZE),
        swapRB=True,
        crop=False,
    )
    net.setInput(warm)
    _ = net.forward(out_layers)


def _load_tflite_interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except Exception:
        pass

    try:
        from tensorflow.lite import Interpreter

        return Interpreter
    except Exception as e:
        raise RuntimeError(
            "TFLite runtime is not installed. Install 'tflite-runtime' or 'tensorflow'."
        ) from e


def _init_tflite() -> None:
    global tflite_interpreter, tflite_input_detail, tflite_output_details

    model_path = Path(TFLITE_MODEL)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing TFLite model: {model_path}")

    Interpreter = _load_tflite_interpreter_class()
    try:
        tflite_interpreter = Interpreter(model_path=str(model_path), num_threads=max(1, TFLITE_THREADS))
    except TypeError:
        tflite_interpreter = Interpreter(model_path=str(model_path))

    tflite_interpreter.allocate_tensors()
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    if len(input_details) != 1:
        raise RuntimeError(f"Expected exactly one input tensor, got: {len(input_details)}")

    tflite_input_detail = input_details[0]
    tflite_output_details = output_details


def _prepare_tflite_input(img_bgr: np.ndarray) -> np.ndarray:
    if tflite_input_detail is None:
        raise RuntimeError("TFLite input tensor details are not initialized")

    input_shape = tflite_input_detail["shape"]
    if len(input_shape) != 4:
        raise RuntimeError(f"Unsupported TFLite input shape: {input_shape}")

    _, in_h, in_w, in_c = input_shape
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (int(in_w), int(in_h)), interpolation=cv2.INTER_LINEAR)
    if int(in_c) == 1:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)[..., np.newaxis]

    tensor = np.expand_dims(resized, axis=0)
    input_dtype = tflite_input_detail["dtype"]

    if input_dtype == np.float32:
        tensor = tensor.astype(np.float32)
        if TFLITE_FLOAT_INPUT_NORM == "minus1_1":
            tensor = (tensor / 127.5) - 1.0
        else:
            tensor = tensor / 255.0
        return tensor

    if input_dtype == np.uint8:
        return tensor.astype(np.uint8)

    if input_dtype == np.int8:
        tensor = tensor.astype(np.float32)
        if TFLITE_FLOAT_INPUT_NORM == "minus1_1":
            tensor = (tensor / 127.5) - 1.0
        else:
            tensor = tensor / 255.0

        scale, zero_point = tflite_input_detail.get("quantization", (0.0, 0))
        if scale and scale > 0:
            tensor = np.round(tensor / scale + zero_point).clip(-128, 127)
        return tensor.astype(np.int8)

    raise RuntimeError(f"Unsupported TFLite input dtype: {input_dtype}")


def _is_integer_like(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    diff = np.abs(values - np.round(values))
    return float(np.mean(diff < 1e-3)) > 0.8


def _parse_tflite_detection_outputs(raw_outputs: list[np.ndarray]):
    boxes = None
    one_d = []

    for tensor in raw_outputs:
        arr = np.squeeze(tensor)
        if arr.ndim == 2 and arr.shape[-1] == 4 and boxes is None:
            boxes = arr.astype(np.float32)
        elif arr.ndim == 1:
            one_d.append(arr.astype(np.float32))

    if boxes is None:
        return None, None, None

    n = boxes.shape[0]
    if n == 0:
        return boxes, np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    candidates = [arr for arr in one_d if arr.shape[0] == n]
    classes = None
    scores = None

    for arr in candidates:
        if arr.size == 0:
            continue
        if classes is None and _is_integer_like(arr):
            classes = arr
        elif scores is None and np.max(arr) <= 1.5 and np.min(arr) >= -0.1:
            scores = arr

    if classes is None and candidates:
        classes = candidates[0]
    if scores is None and candidates:
        scores = candidates[-1]
    if classes is None:
        classes = np.full((n,), -1, dtype=np.float32)
    if scores is None:
        scores = np.ones((n,), dtype=np.float32)

    return boxes, classes, scores


def _detect_is_cat_tflite_classification(raw_outputs: list[np.ndarray]):
    prob_vectors = []
    for tensor in raw_outputs:
        arr = np.squeeze(tensor)
        if arr.ndim == 1 and arr.size > 1:
            prob_vectors.append(arr.astype(np.float32))

    if not prob_vectors:
        return None

    probs = max(prob_vectors, key=lambda arr: arr.size)
    if np.max(probs) > 1.5 or np.min(probs) < -0.1:
        exps = np.exp(probs - np.max(probs))
        probs = exps / np.sum(exps)

    model_cat_class = CAT_CLASS_ID - TFLITE_CLASS_OFFSET
    if model_cat_class < 0 or model_cat_class >= probs.size:
        return False, 0.0, []

    score = float(probs[model_cat_class])
    return score >= TFLITE_SCORE_THRES, score, []


def detect_is_cat_tflite(img_path: str) -> tuple[bool, float, list[str]]:
    if tflite_interpreter is None or tflite_input_detail is None or not tflite_output_details:
        print("[tflite] model is not initialized")
        return False, 0.0, []

    img = cv2.imread(img_path)
    if img is None:
        print("[tflite] cv2.imread returned None")
        return False, 0.0, []

    img_h, img_w = img.shape[:2]

    try:
        input_tensor = _prepare_tflite_input(img)
    except Exception as e:
        print(f"[tflite] input preprocessing error: {e}")
        return False, 0.0, []

    tflite_interpreter.set_tensor(tflite_input_detail["index"], input_tensor)
    tflite_interpreter.invoke()
    raw_outputs = [tflite_interpreter.get_tensor(detail["index"]) for detail in tflite_output_details]

    boxes, classes, scores = _parse_tflite_detection_outputs(raw_outputs)
    if boxes is None:
        classification_result = _detect_is_cat_tflite_classification(raw_outputs)
        if classification_result is None:
            print("[tflite] unsupported output tensors")
            return False, 0.0, []
        is_cat, best_conf, yolo_lines = classification_result
        print(f"[tflite] classification cat_conf={best_conf:.2f}")
        return is_cat, best_conf, yolo_lines

    total = min(len(boxes), len(classes), len(scores))
    best_cat = 0.0
    yolo_lines = []

    for i in range(total):
        score = float(scores[i])
        if score < TFLITE_SCORE_THRES:
            continue

        cls_id_model = int(round(float(classes[i])))
        cls_id = cls_id_model + TFLITE_CLASS_OFFSET
        if cls_id != CAT_CLASS_ID:
            continue

        y1, x1, y2, x2 = [float(v) for v in boxes[i]]
        if max(abs(y1), abs(x1), abs(y2), abs(x2)) <= 1.5:
            x1 *= img_w
            x2 *= img_w
            y1 *= img_h
            y2 *= img_h

        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        if box_w <= 0 or box_h <= 0:
            continue

        x_center = (x1 + box_w / 2) / img_w
        y_center = (y1 + box_h / 2) / img_h
        norm_w = box_w / img_w
        norm_h = box_h / img_h
        yolo_lines.append(f"{CAT_CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
        best_cat = max(best_cat, score)

    print(f"[tflite] cat_conf={best_cat:.2f}, kept={len(yolo_lines)}, raw={total}")
    return best_cat > 0.0, best_cat, yolo_lines


def _nms_boxes(boxes: list[list[int]], confs: list[float], conf_thres: float, nms_thres: float) -> list[int]:
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_thres, nms_thres)
    if idxs is None or len(idxs) == 0:
        return []
    return np.array(idxs).flatten().tolist()


def detect_is_cat_cv(img_path: str) -> tuple[bool, float, list[str]]:
    if net is None or not out_layers:
        print("[yolo] model is not initialized")
        return False, 0.0, []

    img = cv2.imread(img_path)
    if img is None:
        print("[yolo] cv2.imread returned None")
        return False, 0.0, []

    img_h, img_w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    boxes: list[list[int]] = []
    confs: list[float] = []
    class_ids: list[int] = []

    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf <= CONF_THRES:
                continue

            cx, cy, bw, bh = det[:4]
            bw_px = int(bw * img_w)
            bh_px = int(bh * img_h)
            x = int(cx * img_w - bw_px / 2)
            y = int(cy * img_h - bh_px / 2)
            boxes.append([x, y, bw_px, bh_px])
            confs.append(conf)
            class_ids.append(class_id)

    raw_count = len(boxes)
    keep_idxs = _nms_boxes(boxes, confs, CONF_THRES, NMS_THRES)

    dets = [(confs[i], class_ids[i], boxes[i]) for i in keep_idxs]
    dets.sort(reverse=True, key=lambda x: x[0])
    for conf, cls_id, box in dets[:3]:
        print(f"[yolo] class_id={cls_id:<2} conf={conf:.2f} box={box}")

    best_cat = 0.0
    yolo_lines: list[str] = []
    for i in keep_idxs:
        if class_ids[i] != CAT_CLASS_ID:
            continue
        best_cat = max(best_cat, confs[i])
        x, y, width, height = boxes[i]
        x_center = (x + width / 2) / img_w
        y_center = (y + height / 2) / img_h
        norm_w = width / img_w
        norm_h = height / img_h
        yolo_lines.append(f"{CAT_CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    print(f"[yolo] cat_conf={best_cat:.2f}, kept={len(keep_idxs)}, raw={raw_count}")
    return best_cat > 0.0, best_cat, yolo_lines


def detect_is_cat(img_path: str) -> tuple[bool, float, list[str]]:
    if DETECTOR_BACKEND == "tflite":
        return detect_is_cat_tflite(img_path)
    return detect_is_cat_cv(img_path)


def enqueue_motion() -> None:
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


def capture_with_camera(dst_path: Path, rotation: int = 0) -> None:
    cmd = None
    for candidate in ("rpicam-still", "libcamera-still", "libcamera-jpeg"):
        if shutil.which(candidate):
            cmd = candidate
            break
    if not cmd:
        raise FileNotFoundError(
            "No rpicam-still/libcamera-still/libcamera-jpeg found. Install rpicam-apps or libcamera-apps."
        )

    width, height = _parse_res(PHOTO_RES)
    args = [cmd, "-n", "--immediate", "--width", str(width), "--height", str(height), "-o", str(dst_path)]
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


def worker() -> None:
    while True:
        work_q.get()
        try:
            blue_led.on()
            now = datetime.now()
            ts = now.strftime("%d.%m.%Y_%H-%M-%S.") + f"{int(now.microsecond / 1000):03d}"
            tmp_path = TMP_DIR / f"motion_{ts}.jpg"

            capture_with_camera(tmp_path)
            print(f"[shot] saved: {tmp_path}")

            is_cat, best, yolo_lines = detect_is_cat(str(tmp_path))

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


def on_motion() -> None:
    print("[motion] motion detected")
    enqueue_motion()


def on_no_motion() -> None:
    print("[idle] no motion")


def main() -> None:
    try:
        if DETECTOR_BACKEND == "darknet":
            _init_yolo()
        elif DETECTOR_BACKEND == "tflite":
            _init_tflite()
        else:
            raise ValueError(f"Unknown DETECTOR_BACKEND: {DETECTOR_BACKEND}")
    except Exception as e:
        print(f"[init] detector init error: {e}")
        raise SystemExit(1)

    if DETECTOR_BACKEND == "tflite":
        print(
            f"[init] detector=tflite model={TFLITE_MODEL} "
            f"score_thres={TFLITE_SCORE_THRES} class_offset={TFLITE_CLASS_OFFSET}"
        )
    else:
        print(
            f"[init] detector=darknet cfg={YOLO_CFG} input_size={INPUT_SIZE} "
            f"conf={CONF_THRES} nms={NMS_THRES}"
        )

    print(f"[init] warming up PIR for ~{WARMUP_PIR_SEC}s")
    sleep(WARMUP_PIR_SEC)

    threading.Thread(target=worker, daemon=True).start()
    pir.when_motion = on_motion
    pir.when_no_motion = on_no_motion

    print(f"[run] waiting for motion... (cooldown={COOLDOWN_SEC}s, save dir={BASE_DIR})")
    pause()


if __name__ == "__main__":
    main()
