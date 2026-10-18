# ARGUS / Pixel Pursuit: Rohan Mehta Case

ARGUS is a native Windows desktop investigation workshop for first-year engineering students at VESIT. Students inspect the case in the PySide6 application and write/run Python scripts separately in VS Code. There is no embedded editor or arbitrary-code runner.

## Project status

ARGUS is a native PySide6 desktop investigation interface. Image and video processing run locally with OpenCV, and Level 2 inference runs in a worker thread. Students edit and run separate Python starter files in VS Code.

No evidence images, videos, or case records are present in this checkout. Add only the separately supplied assets listed in `assets/evidence/manifest.json`; the app and scripts report missing files instead of substituting examples. The manifest has no Level 2 video filename, so use the Level 2 picker or `--source PATH` when that clip is supplied. The local YOLO checkpoint is present. Evidence-pipeline validation remains blocked until source assets arrive.

## Prerequisites and launch

- Windows 10/11 and Python 3.10–3.12 on `PATH`.
- Network access for initial dependency installation only.
- For Level 2 inference, the Ultralytics package and the included local `yolo11n.pt` checkpoint. It is COCO pretrained and may not recognize a cricket bat.

In PowerShell from the project root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
.\run.ps1
```

After setup, `python main.py` launches the native desktop app. The student scripts are separate:

```powershell
python workshop_code/level1/starter.py
python workshop_code/level2/starter.py
python workshop_code/level3/starter.py
python workshop_code/level4/starter.py
```

Open a level in ARGUS, read its task and concept, open that folder's `starter.py`, complete its TODOs, run the command from the project root, inspect the stated output, then record the observation in the notebook. Select **Use checkpoint and continue** if stuck; checkpoint use is recorded as assisted progress.

## Evidence placement

| ID | Required file | Level / use |
|---|---|---|
| E-01 | `assets/evidence/level1_cctv_dark.png` | Level 1 timestamp enhancement |
| E-02 | `assets/evidence/level2_lab_before.png` | Optional manifest Level 2 image (the app also accepts a user-selected image or video) |
| — | `assets/evidence/level2_lab_after.png` | Optional independent YOLO practice image |
| E-03 | `assets/evidence/level3_corridor_entry.png` | Level 3 red bag segmentation |
| E-04 | `assets/evidence/level4_corridor_exit.mp4` | Level 4 red-region exit crossing |
| — | `assets/records/bag_inventory.txt` | Distinctive repair patch and recorded bag association |
| E-05 | `assets/records/purge_job_log.txt` | Purge-job record and account details, if supplied |
| — | `assets/records/session_audit.txt` | Authentication/session context |

Check the manifest for status and intended use. Do not rename, alter, crop, or replace the supplied evidence. Once files arrive, use each level's evidence picker and run its script; no evidence pipeline can be called fully validated before then.

## Levels and methods

1. **Sight Repair:** Grayscale, CLAHE, and threshold views use actual OpenCV operations. A timestamp is recorded only by an investigator when it is legible; enhancement does not infer digits or incident time.
2. **Lab scene reconstruction:** actual pretrained YOLO predictions are available for a selected image or video. The app browses a bounded set of actual inference frames and summarizes class occurrence across every decoded frame. It is a baseline scene-analysis exercise, not evidence of a theft event. Boxes do not establish identity or absence.
3. **Red-region candidates:** BGR-to-HSV, hue wrap-around, morphological cleanup, area-filtered contours, mask and candidate overlay. The algorithm does not recognize the bag, its pale patch, or its owner.
4. **Route reconstruction:** cleaned red masks, area-filtered candidates, nearest-centroid continuity, trajectory segments, and clip-relative exit-line check. Gaps break the track; a crossing is not identity evidence.

The investigation combines recorded CV observations, source time/frame references, investigator notes, and any supplied documentary records. It does not name a suspect automatically or turn a model output into a conclusion. Missing records are listed as missing.

## Notebook, deduction, and progress

Notebook records are stored as JSON under `%LOCALAPPDATA%\\ARGUS\\evidence_notebook.json` (or the platform data directory). Entries preserve source path, actual output path, method, observation, interpretation, uncertainty, evidence ID, and whether completion used a checkpoint. Saved OpenCV result images go under `outputs/levelN`; students' starter files are never overwritten. The notebook can display manifest-listed text records when supplied. The final deduction asks for a working hypothesis, three sourced clues, an assessment of records (including an explicit missing-record note), and unresolved uncertainty.

## Outputs and reference solutions

Student output files go under `outputs/level1` through `outputs/level4`. Reference implementations are in `workshop_code/levelN/solution.py`; the corresponding README explains the concept, instructions, outputs, and limits. The Level 2 script attempts model inference only when `ARGUS_YOLO_WEIGHTS` points to a local compatible `.pt` file and Ultralytics is installed. No live model training is needed or included in the four-hour core workshop.

## Verification and current limitations

Verification: `python -m compileall -q main.py argus_app workshop_code tests` passed; `python -m unittest discover -s tests -v` passed (18 tests); and an offscreen PySide6 smoke check opened all eight pages and validated the Level 1 choices and Level 2 frame controls. The local YOLO checkpoint and Ultralytics dependencies are installed; actual model inference passed only on a test-only in-memory array, not case evidence. All 8 manifest-listed assets are absent, so real-video YOLO class findings, Level 1 source legibility, Level 3 source candidates, Level 4 tracking/crossing, and documentary interpretation remain unverified. The Level 2/4 reference entry points were run and reported their specific missing input files without substituting footage.

If YOLO is unavailable, check `ultralytics`, `yolo11n.pt`, and `ARGUS_YOLO_WEIGHTS`. If an image/video does not decode, check the supplied file and codec; the app reports a decode failure instead of substituting evidence. Evidence notebook save errors are surfaced in the notebook status.
