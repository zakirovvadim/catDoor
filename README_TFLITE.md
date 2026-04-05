# catDoor TFLite Notes

`pir_camera_yolo.py` now supports two detector backends selected via `DETECTOR_BACKEND`.

- `darknet` (default): OpenCV DNN + YOLOv4-tiny (`.cfg` + `.weights`)
- `tflite`: TensorFlow Lite (`.tflite`)

## TFLite run example

```bash
export DETECTOR_BACKEND=tflite
export TFLITE_MODEL=/home/pi/models/tflite/model.tflite
export TFLITE_SCORE_THRES=0.35
export CAT_CLASS_ID=15
python3 pir_camera_yolo.py
```

Optional variables:

- `TFLITE_CLASS_OFFSET` (for models with shifted class ids, for example `-1`)
- `TFLITE_THREADS` (number of TFLite inference threads)
- `TFLITE_FLOAT_INPUT_NORM` (`zero_one` or `minus1_1`)

## Darknet run example

```bash
export DETECTOR_BACKEND=darknet
export YOLO_CFG=/home/pi/models/yolo-tiny/yolov4-tiny.cfg
export YOLO_WEIGHTS=/home/pi/models/yolo-tiny/yolov4-tiny.weights
export INPUT_SIZE=416
python3 pir_camera_yolo.py
```
