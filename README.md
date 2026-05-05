<div align="center">

<h1>YOLOv11s Custom Object Detection</h1>

<p><strong>End-to-end object detection pipeline — self-annotated dataset · Kaggle GPU training · real-time webcam inference</strong></p>

<p>
  <img src="https://img.shields.io/badge/Model-YOLOv11s-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/mAP@0.5-97.7%25-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Epochs-100-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Classes-3-lightgrey?style=flat-square"/>
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20T4%20(Kaggle)-yellow?style=flat-square"/>
  <img src="https://img.shields.io/badge/Framework-Ultralytics-red?style=flat-square"/>
</p>

</div>

---

## Overview

A complete custom object detection project using **YOLOv11s** (Ultralytics), trained on a self-built and manually annotated dataset of **5,000+ images** across three classes: **smartphone**, **ball**, and **watch**.

The project covers the full pipeline from scratch — data collection, bounding-box annotation with LabelImg, cloud GPU training on Kaggle's free NVIDIA T4, and real-time webcam inference with OpenCV.

Two training runs were conducted: a baseline 50-epoch run with no augmentation, and an improved 100-epoch run with a full augmentation strategy (mosaic, mixup, cutmix, HSV, flips). The second run achieved stable real-time detection — the primary goal.

---

## Results

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **97.7%** |
| **mAP@0.5:0.95** | 82.2% |
| **Precision** | 94.6% |
| **Recall** | 95.5% |
| **Epochs** | 100 |
| **Training Platform** | Kaggle NVIDIA T4 GPU |

### Per-Class Breakdown

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------|-----------|-----------|--------|----------|--------------|
| 📱 Smartphone | 61 | 64 | 94.8% | 93.8% | 98.1% | 85.2% |
| ⚽ Ball | 49 | 107 | 99.4% | 95.3% | 97.6% | 83.2% |
| ⌚ Watch | 79 | 81 | 99.1% | 98.8% | 99.5% | 83.0% |

---

## Pipeline

```
Data Collection → Annotation (LabelImg) → Kaggle Upload → YOLO Training → Inference
    ~5,000 imgs      YOLO .txt labels       data.yaml       T4 GPU           webcam / image
```

**Step 1 — Data Collection:** Images downloaded from the web for each of the three classes (~5,000 total).

**Step 2 — Annotation:** All images manually annotated using LabelImg in YOLO format (`class_id cx cy w h` normalised).

**Step 3 — Kaggle Upload:** Dataset uploaded as a Kaggle dataset input. `data.yaml` created programmatically inside the notebook since Kaggle inputs are read-only.

**Step 4 — Training:** Two runs on free NVIDIA T4 GPU. Run 1: 50 epochs, no augmentation. Run 2: 100 epochs, full augmentation suite.

**Step 5 — Inference:** `best.pt` downloaded and used for real-time webcam detection (OpenCV) and static image inference.

---

## Dataset

The full annotated dataset (~5,000 images with YOLO-format labels) is split into train / val / test.

```
dataset/
├── train/
│   ├── images/        # ~4,500 images
│   └── labels/        # matching .txt annotation files
├── valid/
│   ├── images/        # ~300 images
│   └── labels/
└── test/
    ├── images/        # ~200 images
    └── labels/
```

**→ [Download the Dataset here]([https://aazanalikhan.vercel.app/projects/project_4/project4.html])**

**Classes:**
- `0` — Smartphone (various models, orientations, cases)
- `1` — Ball (football, basketball, tennis ball, etc.)
- `2` — Watch (wristwatches, smartwatches, various angles)

**Label format (YOLO):**
```
class_id  center_x  center_y  width  height
    0       0.512     0.487    0.231  0.334
```
All values are normalised to [0, 1] relative to image dimensions.

---

## Training

### data.yaml

```yaml
train: /kaggle/input/your-dataset-path/dataset/train/images
val:   /kaggle/input/your-dataset-path/dataset/valid/images
test:  /kaggle/input/your-dataset-path/dataset/test/images

nc: 3
names:
  - smartphone
  - ball
  - watch
```

> On Kaggle, write this file to `/kaggle/working/data.yaml` since the input directory is read-only.

### Run 1 — Baseline (50 epochs, no augmentation)

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="/kaggle/working/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)
```

### Run 2 — Augmented (100 epochs, full augmentation)

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="/kaggle/working/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    name="train11s_aug",

    # Augmentation
    augment=True,
    flipud=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    mosaic=1.0,
    mixup=0.5,
    cutmix=0.5,
    perspective=0.0
)
```

### Export results from Kaggle

```python
import shutil

shutil.make_archive(
    '/kaggle/working/model_result',
    'zip',
    '/kaggle/working/runs'
)
# Download model_result.zip from the Kaggle Output tab
# best.pt is at: runs/detect/train11s_aug/weights/best.pt
```

---

## Inference

### Real-Time Webcam Detection

```python
from ultralytics import YOLO
import cv2
import time

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, imgsz=420)
    annotated_frame = results[0].plot()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(annotated_frame, f"FPS: {int(fps)}",
        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Static Image Detection

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
img = cv2.imread("1.jpg")

results = model(img, conf=0.5, imgsz=420)
annotated_img = results[0].plot()

cv2.imshow("YOLO Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("result.jpg", annotated_img)
print("Result saved as result.jpg")
```

---

## Run 1 vs Run 2 — Comparison

| | Run 1 | Run 2 |
|---|---|---|
| Epochs | 50 | 100 |
| Augmentation | None | Mosaic, MixUp, CutMix, HSV, Flips |
| Val mAP@0.5 | Adequate | **97.7%** |
| Real-time performance | Inconsistent | **Stable** |
| F1 peak | Lower | Higher |
| Convergence | Early plateau | Smooth to ~epoch 80 |

**Key finding:** The baseline run produced acceptable validation metrics but failed under real-world webcam conditions. The augmented run resolved this — stable, confident detections across varying backgrounds and lighting.

---

## Training Observations

- **No significant overfitting:** mAP did not drop after epoch 70–80 and validation loss continued to decrease — the augmentation strategy prevented the model from memorising training data.
- **Diminishing returns after ~epoch 80:** mAP improvement became nearly flat past epoch 80, indicating the model converged. Training beyond 100 epochs would yield minimal further gain for this dataset size.

---

## Requirements

```
ultralytics
opencv-python
```

Install:
```bash
pip install ultralytics opencv-python
```

Place `best.pt` in the same directory as your inference script and run directly.

---

---

## Challenges & Solutions

**Kaggle YAML issue** — Kaggle dataset inputs are read-only, so `data.yaml` must be written programmatically to `/kaggle/working/` at the start of each notebook session. Fixed with a one-cell setup script.

**Poor real-time generalisation (Run 1)** — The baseline model produced good val mAP but failed in webcam conditions. Solved by introducing full augmentation (mosaic, mixup, cutmix, HSV, flips) in Run 2 and doubling the epoch count.

**Class imbalance** — The ball class had significantly more instances per image (107 in 49 images) vs smartphone and watch. Per-class metrics were monitored individually each epoch to verify no class was dominating mAP unfairly.

**Session timeout risk** — Kaggle T4 sessions have a 12-hour runtime limit. Training output was zipped and downloaded immediately after completion to prevent loss on session expiry.

---

## Future Improvements

- [ ] Add more classes (laptop, headphones, bottle) for a general household-object detector
- [ ] Export to ONNX / TensorRT for edge deployment (Raspberry Pi, Jetson Nano)
- [ ] Build a Gradio or Streamlit demo for browser-based inference
- [ ] Serve detections via a FastAPI endpoint
- [ ] Explore YOLOv11n (nano) for compute-constrained deployment
- [ ] Expand dataset with harder conditions: occlusion, low light, low resolution

---

## License

Designed by **[Aazan Ali Khan](https://aazanalikhan.vercel.app)** — feel free to use, modify, and build upon it.

📄 [View Full Documentation](https://aazanalikhan.vercel.app/projects/project_3/project3.html)
