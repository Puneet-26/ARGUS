# Optional YOLO fine-tuning

ARGUS uses pretrained `yolo11n.pt` for people and supported common objects. First inference downloads the small pretrained weights when needed. No dataset or custom model is required.

To prepare supplied footage for annotation, sample one frame every two seconds by default:

```powershell
python tools/sample_video_frames.py path/to/supplied_clip.mp4 --output datasets/frames --interval 2
```

Annotate the sampled frames with a bounding-box tool such as CVAT or Label Studio, then arrange the labeled dataset in standard Ultralytics YOLO format and create its dataset YAML (`train`, `val`, `names`). Do not treat unreviewed annotations as ground truth.

Fine-tune from pretrained weights only when a needed class is not reliably supported:

```powershell
python tools/train_yolo.py --data path/to/dataset.yaml --epochs 30 --imgsz 640
```

Optional flags include `--names "orange_cup,lab_badge"`, `--weights yolo11n.pt`, `--device cpu`, and `--output models/custom`. The script copies the best checkpoint to `models/custom/best.pt`; ARGUS prefers that file automatically. Set `ARGUS_YOLO_WEIGHTS` to use another checkpoint.
