# ARGUS · The Lab 4B Incident

ARGUS is a native Windows desktop workshop for investigating image and video evidence with OpenCV and pretrained Ultralytics YOLO. It is built with Python and PySide6. The application does not require a browser, website, localhost server, or internet connection for ordinary OpenCV work.

Each investigation level has its own evidence picker. Levels 1–3 accept one supported image at a time. Level 4 accepts video only. Nothing is automatically loaded when ARGUS starts, and no detection or tracking output is fabricated.

## Investigation levels

1. **Eyes Online — image enhancement:** inspect the selected original image, convert it to grayscale, apply CLAHE, reduce noise, and create a threshold view. Strength and threshold sliders change the OpenCV operations. Stage previews are available as tabs. The source image remains unchanged when settings are adjusted.
2. **Object Detection — YOLO:** select one image and ARGUS runs pretrained YOLO on the CPU. The result shows model-drawn boxes, class names, confidence scores, and a detection table. Confidence and class filters trigger a new inference. Record a result explicitly in the notebook.
3. **Colour Identification — HSV segmentation:** select a colour preset or adjust hue, saturation, and value ranges. The application displays the binary mask, isolated pixels, region overlay, pixel coverage, and contour details. Red handles both ends of OpenCV's hue range.
4. **Timeline and Tracking — video:** select a video to inspect its first frame, play or pause, seek, and inspect frame time. OpenCV background subtraction, morphology, centroids, and a nearest-centroid tracker draw trails and check crossings against the adjustable vertical exit line. These are motion regions, not semantic person detections; a still image cannot establish movement.

The Evidence Notebook stores actual user-recorded observations and analysis results in the current case data. The report separates observations, computer-vision results, interpretations, and unresolved questions. Investigator notes stay in the report while the application remains open and are included in Markdown export. ARGUS does not automatically identify a person or declare guilt.

## Requirements

- Windows 10 or 11.
- Python 3.10, 3.11, or 3.12 available as `python` in PowerShell.
- Internet access for first-time dependency installation. First YOLO inference may also download the `yolo11n.pt` weights; inference can run offline after the dependencies and weights are available locally.
- A video codec supported by the OpenCV build for Level 4.

No weights are committed to this repository. Level 1, Level 3, and Level 4 remain available without Ultralytics weights; Level 2 gives a clear error if the model package or weights are unavailable.

## Windows setup and launch

From PowerShell, change to the project folder and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
.\run.ps1
```

`setup.ps1` checks the Python version, creates the project-local `.venv`, and installs `requirements.txt`. It does not download evidence or train a model. `run.ps1` launches `main.py` with that environment's Python executable. After setup, the app can also be launched with:

```powershell
.\.venv\Scripts\python.exe .\main.py
```

## Upload evidence

- In Levels 1–3, click **Open Image Evidence…** (Level 2 has **Open Image…**) or click the empty original-image viewer. Choose a supported image such as PNG, JPEG, BMP, WebP, or TIFF. The selected image is decoded and displayed before analysis; the visible filename and dimensions identify the source. Use the level's clear action before selecting a replacement if needed.
- In Level 4, click **Open Video Evidence…** or click the empty video viewer. Choose an MP4, AVI, MOV, MKV, WebM, or another listed video format. ARGUS opens the video using OpenCV, displays decoded frames, and shows a useful error when the file or codec cannot be read.
- Drag and drop a local file onto the original viewer. A file of the wrong type is rejected for that level.

No image or video is bundled. Add supplied evidence when available and open it manually in the appropriate level. ARGUS does not generate or substitute evidence.

## How the analyses work

### Level 1: image enhancement

The grayscale option converts BGR pixels to grayscale. CLAHE (Contrast Limited Adaptive Histogram Equalization) improves local contrast using an 8×8 tile grid and the selected clip limit. Noise reduction applies a light Gaussian blur. Thresholding compares grayscale pixel intensity to the chosen level and creates a binary image. The stage tabs show the computed outputs. Enhancement makes existing pixel patterns easier to inspect; it does not recover information absent from the source.

### Level 2: pretrained object detection

ARGUS uses the lightweight `yolo11n.pt` detection checkpoint through the Ultralytics Python API, with `imgsz=640` and CPU inference. The default weights were pretrained on COCO and support 80 everyday classes, including:

- People and transport: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat.
- Street and animals: traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe.
- Personal and sports items: backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket.
- Kitchen items and food: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake.
- Furniture, electronics, and other items: chair, couch, potted plant, bed, dining table, toilet, TV, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

The model can detect those classes without workshop training because its pretrained weights already contain examples of them. A bounding box marks a predicted pixel region; its confidence is a model score, not a guarantee. Small, occluded, dark, or compressed objects may be missed. A custom class (such as a distinctive local object) needs verified, manually annotated bounding boxes and fine-tuning. A class label cannot identify a real-world person.

The class filter can show all model classes, people, or supported common objects. Confidence changes the minimum score sent to inference. When present, `models/custom/best.pt` is loaded instead of the default model; `ARGUS_YOLO_WEIGHTS` can point to another checkpoint.

### Level 3: HSV colour segmentation

ARGUS converts the selected BGR image to HSV and uses `cv2.inRange()` to make a binary mask. It optionally cleans the mask with morphological operations, isolates matching pixels, draws the largest meaningful contour, and overlays the region on the original. Coverage is the number of matching mask pixels divided by all image pixels. HSV segmentation groups pixels by colour; it does not understand object meaning, identify a person, determine ownership, or establish guilt. Lighting, white balance, and compression affect results.

### Level 4: motion tracking

OpenCV MOG2 background subtraction highlights changing regions. Morphological filtering removes some noise, contours produce centroids, and a nearest-centroid tracker associates nearby regions across frames and maintains track IDs and short trails. Consecutive positions are tested against the configured line; a track can produce at most one crossing event during the current tracking run. Camera movement, shadows, occlusion, and region merges can affect tracking. The implemented Level 4 method is motion-region tracking; it does not classify a track as a person.

## Notebook, report, and case data

Notebook entries are automatically stored as JSON in:

```text
%LOCALAPPDATA%\ARGUS\evidence_notebook.json
```

On non-Windows systems the fallback is `$XDG_DATA_HOME/ARGUS/evidence_notebook.json`, or `~/.local/share/ARGUS/evidence_notebook.json`. Use **Notebook → Export Notebook (JSON)…** to make a portable copy. Use **Settings → Reset case data** to clear the saved notebook and dashboard progress after confirmation. The report refresh reads the current notebook; report notes are preserved during refresh and included in Markdown export.

## Optional custom YOLO fine-tuning

Training is optional and is never run during installation or application startup. First extract a modest number of frames from supplied video for annotation (one frame every two seconds by default):

```powershell
.\.venv\Scripts\python.exe .\tools\sample_video_frames.py .\path\to\supplied_video.mp4 --interval 2 --output .\datasets\frames
```

Use a bounding-box annotation tool such as CVAT or Label Studio to label the frames. Review the annotations; ARGUS does not invent or verify labels. Prepare the standard Ultralytics dataset directories and a dataset YAML, then fine-tune from pretrained weights:

```powershell
.\.venv\Scripts\python.exe .\tools\train_yolo.py --data .\path\to\dataset.yaml --epochs 30 --imgsz 640
```

The script accepts `--names "custom_object,another_object"`, `--weights yolo11n.pt`, `--device cpu`, and `--output .\models\custom`. It reports Ultralytics training progress and errors in the console and copies the best checkpoint to `models/custom/best.pt`. ARGUS will use that file automatically. Do not fine-tune on unreviewed annotations.

## Project structure

```text
main.py                         Native desktop entry point
run.py                          Compatibility entry point
setup.ps1 / run.ps1             Windows setup and launch
argus_app/core/                 Image/video input and tracking primitives
argus_app/labs/solutions/        OpenCV enhancement, colour, and motion operations
argus_app/evidence/             Notebook records and persistence
argus_app/ui/                   PySide6 window, levels, notebook, report, and settings
assets/evidence/                Supplied evidence kept separate from application code
tools/                          Optional frame sampling and YOLO fine-tuning utilities
requirements.txt                Direct Python dependencies
```

## Troubleshooting and limitations

- **Python version rejected:** install Python 3.10–3.12 and ensure its `python` command is available in a new PowerShell session.
- **Dependency installation fails:** check pip's error output and network access, then rerun `setup.ps1`. The app does not silently switch to global packages.
- **YOLO import/model error:** check that setup completed. The initial model download needs internet access; afterward, confirm that `yolo11n.pt`, `models/custom/best.pt`, or the path in `ARGUS_YOLO_WEIGHTS` exists and is a compatible Ultralytics checkpoint.
- **Image rejected:** confirm that the extension is supported and the image file is readable. The viewer reports a decode error instead of substituting another image.
- **Video cannot open or seek:** verify the file and codec with a player or transcode to a commonly supported MP4/H.264 file. OpenCV codec support depends on the installed build.
- **Unexpected detection or no detection:** inspect the source, confidence threshold, class filter, and model limitations. The model is not guaranteed to detect every instance of its supported classes.
- **Notebook save warning:** check that the current Windows user can write to `%LOCALAPPDATA%\ARGUS`. Export entries manually from the Notebook page if needed.

Level 4 reports motion regions rather than semantic people. Video time is relative to the selected clip; ARGUS does not invent a CCTV wall-clock time. No custom model, custom labels, or supplied video is assumed to exist.
