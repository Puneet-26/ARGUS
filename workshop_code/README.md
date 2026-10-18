# ARGUS student coding files

Open the corresponding `starter.py` in VS Code, finish its TODOs, and run it from the project root in PowerShell. Defaults use the manifest-listed evidence; Level 2 also accepts an actual image or video path through `--source PATH`. No example evidence is bundled. Scripts report a clear missing-file error until the source is supplied.

```powershell
python workshop_code/level1/starter.py
```

Outputs are written under `outputs/levelN/` only after processing the actual input. The reference files in each level folder are instructor materials; they are not required to run the app. Install the main project dependencies using `setup.ps1` or `python -m pip install -r requirements.txt`.

Level 2 uses `ultralytics` and the repository's `yolo11n.pt` when available. Set `ARGUS_YOLO_WEIGHTS` to override it with a compatible local `.pt` checkpoint. The checkpoint is COCO-trained; it may miss objects or classes. It has no scene-differencing or object-identity claim. All predictions come from actual inference.

Level 4 evaluates red-region candidates frame by frame, follows the nearest plausible centroid, breaks the trajectory on missed detections or large jumps, and reports a crossing only across continuous observations. Times are clip-relative unless evidence metadata establishes a wall-clock offset. A crossing does not identify the bag or carrier.
