# ARGUS

## Setup

Use Python 3.10–3.12. From the project folder, install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Level 2 model

The project includes the pretrained `yolo11n.pt` COCO model at
`assets/models/yolo11n.pt`. The desktop application and Level 2 script resolve
this path relative to the project files, so launching from another working
directory does not affect model discovery. GitHub distributes this 5.6 MB
file with the repository using regular Git; no Git LFS setup or separate
download is needed after cloning.

To use a different local checkpoint, set `ARGUS_YOLO_WEIGHTS` to its path.
Level 1, Level 3, and Level 4 do not load this model.

If the bundled file is removed, the official Ultralytics weights can be
downloaded again and placed in the expected location:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
New-Item -ItemType Directory -Force assets/models | Out-Null
Move-Item yolo11n.pt assets/models/yolo11n.pt
```

## Launch

From the project folder, run:

```powershell
python main.py
```
