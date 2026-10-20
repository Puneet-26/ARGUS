# ARGUS

## Setup

Use Python 3.10–3.12. From the project folder, install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Level 2 uses pretrained `yolo11n.pt` weights. Set `ARGUS_YOLO_WEIGHTS` to a
local checkpoint path, or place the weights at `assets/models/yolo11n.pt`.
To download the official weights once and put them in the default location:

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
