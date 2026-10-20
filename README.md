# ARGUS / Pixel Pursuit

**ARGUS** is an interactive desktop computer vision workshop application designed for **Pixel Pursuit**, organised by **AI CoLegion at VESIT**.

The workshop teaches practical computer vision concepts through an investigative murder mystery: Rohan has been found murdered in the VESIT Robotics Laboratory. Participants examine corrupted CCTV footage, a missing phone, a cricket bat, a crimson-red backpack, corridor footage, and an ARGUS system purge recorded under `kabir.lab` at 23:44.

All four investigation levels remain unlocked and accessible. The application presents evidence, provides real-time processing and playback controls, and logs observations into a persistent Evidence Notebook and Final Report.

The core algorithms reside in one authoritative folder: [`level_codes/`](level_codes/). Each teaching script runs standalone and serves as the exact implementation used by the desktop application.

---

## Workshop Structure

### Day 1: Image Enhancement & Object Detection
- **Level 1: Eyes Online** (`level_codes/level1.py`)
  - Image loading and validation with OpenCV.
  - Channel inspection (BGR format) and grayscale conversion (`cv2.cvtColor`).
  - Local contrast enhancement with CLAHE (`cv2.createCLAHE`).
  - Binary thresholding (`cv2.threshold`).
  - Clock Region of Interest (ROI) extraction and cubic enlargement (`cv2.resize`).
- **Level 2: The Missing Phone** (`level_codes/level2.py`)
  - Pretrained Ultralytics YOLO object detection (`yolo11n.pt` on CPU).
  - Extracting class names, confidence scores, and bounding boxes in source coordinates.
  - Drawing non-overlapping labels and synchronized object list.
  - Real-time inference during video playback.

### Day 2: Colour Segmentation & Motion Tracking
- **Level 3: Identify the Red Bag** (`level_codes/level3.py`)
  - BGR to HSV colour space conversion (`cv2.cvtColor`).
  - Dual red-hue range masking for OpenCV's 0–179 hue wraparound (`cv2.inRange`, `cv2.bitwise_or`).
  - Morphological noise removal via opening and closing (`cv2.morphologyEx`).
  - External contour extraction and area filtering (`cv2.findContours`, `cv2.contourArea`).
  - Live segmentation during video playback with configurable HSV and morphology sliders.
- **Level 4: Track the Red Bag** (`level_codes/level4.py`)
  - Continuous video frame processing and HSV red candidate extraction.
  - Centroid calculation using spatial moments (`cv2.moments`).
  - Nearest-neighbour centroid association and trajectory drawing.
  - Handling occlusions and tracking loss without jumping to unrelated distractors.
  - Configurable Lab 4B entrance line and directional line-crossing detection.

---

## Requirements & Setup

### Environment
- **Operating System:** Windows 10 / 11, macOS, or Linux.
- **Python Version:** Python 3.10 – 3.12.

### Installation
From the project root directory, install the required dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Launching the Application
Launch ARGUS with either:

```powershell
python run.py
```
*(or `python main.py`)*

---

## Evidence Assets & Model Delivery

### Evidence Files
Users can upload any suitable image or video directly into the application's UI screens (drag-and-drop or file selector) or pass the file path via CLI arguments.

### YOLO Model Weights
Level 2 uses a compact, CPU-compatible model:
- Place `yolo11n.pt` at `assets/models/yolo11n.pt` (or specify via `ARGUS_YOLO_WEIGHTS` environment variable).
- To download the official checkpoint once:
  ```powershell
  python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
  Move-Item yolo11n.pt assets/models/yolo11n.pt
  ```

---

## Running the Authoritative Level Scripts Directly

Each level script can be inspected, explained live, and executed standalone by participants:

### Level 1: Eyes Online (Image Enhancement)
```powershell
python level_codes/level1.py --image <path_to_image> --effect CLAHE --clip-limit 2.0
```
*Outputs are saved to `outputs/level1/`.*

### Level 2: The Missing Phone (Object Detection)
```powershell
python level_codes/level2.py --source <path_to_image_or_video> --confidence 0.25
```
*Outputs are saved to `outputs/level2/` (`_detections.png`/`_detections.mp4` and `_detections.json`).*

### Level 3: Identify the Red Bag (Color Segmentation)
```powershell
python level_codes/level3.py --source <path_to_image_or_video> --color Red
```
*Outputs are saved to `outputs/level3/` (`red_mask.png` and `red_candidates.png`).*

### Level 4: Track the Red Bag (Motion Tracking & Tripwire)
```powershell
python level_codes/level4.py --video <path_to_video> --line-x-fraction 0.60 --crossing-direction left_to_right
```
*Outputs are saved to `outputs/level4/` (`tracked_corridor.mp4` and `tracking_summary.json`).*

---

## Technical & Investigative Limitations

- **CLAHE & Image Enhancement:** Contrast adjustment highlights local intensity differences but cannot recreate visual detail absent from the sensor data. Timestamps in CCTV footage represent camera clock observations, not independently verified incident times.
- **YOLO Object Detection:** Standard COCO pretrained weights recognize 80 common categories (e.g., people, backpacks, cell phones) but have no class for cricket bats and cannot identify ownership or intent. A missed detection does not prove absence.
- **Colour Segmentation:** HSV thresholding isolates pixels matching a colour profile. It cannot distinguish between a red backpack, a red jacket, or a red fire extinguisher without human corroboration and additional documentary evidence.
- **Centroid Tracking & Line Crossings:** Nearest-neighbour tracking relies on frame-to-frame spatial continuity. It detects movement relative to the video frame; an exit line crossing indicates motion past a boundary, not the verified identity of the person carrying the item.

