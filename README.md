# ARGUS / Pixel Pursuit

ARGUS is a native PySide6 workshop application for examining supplied image and video evidence. All four levels are unlocked. The UI presents evidence, actual OpenCV/YOLO results, interactive controls, a persistent evidence notebook, and a report. The four workshop algorithms live in one folder: [`level_codes/`](level_codes/). Each script runs directly and is also the algorithm source called by the app; participants do not edit the GUI.

## Clone, install, launch

Use Windows 10/11 with Python 3.10–3.12.

```powershell
git clone https://github.com/Puneet-26/ARGUS-OpenCV-Workshop.git
cd ARGUS-OpenCV-Workshop
python -m pip install -r requirements.txt
python run.py
```

## Evidence

Evidence is supplied separately and is not bundled in this checkout. See [`assets/evidence/manifest.json`](assets/evidence/manifest.json) for expected files. Place the Level 1 image in `assets/evidence/` named `EVIDENCE-01` with its original extension (for example, `EVIDENCE-01.png`). Keep other supplied evidence in the level folders identified in the manifest. The app also allows selecting image/video evidence directly.

## Teaching scripts

Run these from the repository root. Use `--help` to see optional inputs and parameters.

```powershell
python level_codes/level1.py
python level_codes/level2.py --source assets/evidence/your_image.jpg
python level_codes/level3.py
python level_codes/level4.py
```

Level 1 displays full-frame enhancement and, when the named evidence has a validated clock region, an enlarged enhanced detail. Level 2 accepts an image or video and displays actual local-model predictions. Level 3 accepts an image and creates red mask candidates. Level 4 processes video frames and can evaluate a doorway crossing only when given a line position based on the real footage. Colour candidates are not semantic identifications.

## Level 2 model

Place a compatible Ultralytics YOLO `.pt` checkpoint at `assets/models/yolo11n.pt` (or `assets/models/custom/best.pt`). Weights are excluded from Git. For the official YOLO11 nano checkpoint, install requirements and download it once with:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
Move-Item yolo11n.pt assets/models/yolo11n.pt
```

Or set `ARGUS_YOLO_WEIGHTS` to a local checkpoint. Model predictions cover supported classes only; missing detections do not establish absence. Review Ultralytics licensing before redistribution.

## Interactive controls and interpretation

The app exposes Level 1 CLAHE and threshold controls, Level 2 confidence, Level 3 HSV/morphology/area, and Level 4 tracking/doorway controls. Outputs are saved under `outputs/level1` through `outputs/level4`; the notebook is stored in the user's application-data directory. CLAHE cannot restore detail absent from the source, YOLO boxes do not establish identity, and red segmentation/tracking can select distractors such as a fire extinguisher. Inspect the actual Level 4 footage before enabling its entrance line; without a configured line, crossings are not evaluated.
