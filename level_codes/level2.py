"""Level 2: run actual pretrained YOLO object detection on evidence."""

from pathlib import Path
import argparse
import json
import os
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "assets" / "models" / "yolo11n.pt"
OUTPUT_DIR = ROOT / "outputs" / "level2"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv", ".3gp", ".flv", ".ts", ".mts", ".m2ts", ".ogv"}


def class_name(names, class_id):
    return str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])


def infer_frame(model, frame, confidence=0.25, image_size=640):
    """Run actual YOLO inference and annotate its boxes in source-frame pixels."""
    prediction = model.predict(frame, device="cpu", conf=confidence, imgsz=image_size, verbose=False)[0]
    records = []
    height, width = frame.shape[:2]
    average_dimension = (height + width) / 2
    line_width = max(2, round(min(height, width) / 360))
    font_scale = max(0.7, min(2.0, average_dimension / 1700))
    text_thickness = max(2, round(line_width * 0.7))
    padding = max(5, line_width * 2)
    # Bright, distinct BGR colors are indexed by the model's real class ID.
    colors = [
        (0, 220, 255), (255, 115, 35), (80, 230, 90), (235, 70, 210),
        (40, 170, 255), (225, 225, 55), (180, 100, 255), (90, 220, 220),
    ]
    annotated = frame.copy()
    label_regions = []
    box_drawings = []
    if prediction.boxes is not None:
        names = prediction.names
        for box in prediction.boxes:
            class_id = int(box.cls[0].item())
            label = class_name(names, class_id)
            confidence_score = float(box.conf[0].item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            # YOLO coordinates refer to the original input frame. Clamp only to
            # its borders; display scaling happens later in the Qt image widget.
            x1 = min(max(x1, 0.0), float(width - 1))
            y1 = min(max(y1, 0.0), float(height - 1))
            x2 = min(max(x2, 0.0), float(width - 1))
            y2 = min(max(y2, 0.0), float(height - 1))
            coordinates = [round(x1), round(y1), round(x2), round(y2)]
            records.append({
                "class_id": class_id,
                "label": label,
                "confidence": confidence_score,
                "box": coordinates,
            })
            color = colors[class_id % len(colors)]
            text = f"{label} {confidence_score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )
            label_width = min(width, text_width + padding * 2)
            label_height = min(height, text_height + baseline + padding * 2)
            left = min(max(0, coordinates[0]), max(0, width - label_width))
            top_candidates = (
                coordinates[1] - label_height,
                coordinates[3] + 1,
                coordinates[1],
                coordinates[3] - label_height,
            )
            top = next((
                candidate_top for candidate_top in top_candidates
                if 0 <= candidate_top <= height - label_height
                and all(
                    left + label_width <= old_left
                    or old_left + old_width <= left
                    or candidate_top + label_height <= old_top
                    or old_top + old_height <= candidate_top
                    for old_left, old_top, old_width, old_height in label_regions
                )
            ), min(max(0, coordinates[1]), max(0, height - label_height)))
            label_regions.append((left, top, label_width, label_height))
            box_drawings.append((coordinates, color, (left, top, label_width, label_height), text))

    # Draw boxes first and opaque labels second so labels stay readable.
    for coordinates, color, _label_region, _text in box_drawings:
        x1, y1, x2, y2 = coordinates
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)
    for _coordinates, color, (left, top, label_width, label_height), text in box_drawings:
        brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        text_color = (20, 20, 20) if brightness > 165 else (255, 255, 255)
        cv2.rectangle(
            annotated, (left, top), (left + label_width, top + label_height), color, -1
        )
        cv2.putText(
            annotated, text, (left + padding, top + label_height - padding - 1),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness, cv2.LINE_AA,
        )
    return annotated, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Path to a real evidence image or video")
    parser.add_argument("--weights", type=Path, default=None, help="Local YOLO .pt file; defaults to ARGUS_YOLO_WEIGHTS or yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum prediction confidence, from 0 to 1")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size; larger can help small objects but runs slower")
    args = parser.parse_args()

    source = args.source.expanduser()
    if not source.is_absolute() and not source.is_file():
        source = ROOT / source
    if not source.is_file():
        raise SystemExit(f"Evidence file not found: {source}")
    configured_weights = args.weights or Path(os.environ.get("ARGUS_YOLO_WEIGHTS", DEFAULT_WEIGHTS))
    configured_weights = configured_weights.expanduser()
    if not configured_weights.is_absolute() and not configured_weights.is_file():
        configured_weights = ROOT / configured_weights
    if not configured_weights.is_file():
        raise SystemExit(
            f"YOLO weights not found: {configured_weights}\n"
            "Install ultralytics and run `python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\"` "
            "once to download the official checkpoint, or set ARGUS_YOLO_WEIGHTS."
        )
    if not 0 <= args.confidence <= 1:
        raise SystemExit("--confidence must be between 0 and 1")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is missing. Install project dependencies with `python -m pip install -r requirements.txt`.") from exc

    # Load the compact COCO-pretrained detector; all results below come from it.
    model = YOLO(str(configured_weights))
    print(f"Loaded {configured_weights}; model class map: {model.names}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []


    if source.suffix.lower() in VIDEO_SUFFIXES:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise SystemExit(f"OpenCV could not open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = OUTPUT_DIR / f"{source.stem}_detections.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise SystemExit(f"Could not create annotated video: {output_path}")
        frame_number = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                annotated, detections = infer_frame(model, frame, args.confidence, args.imgsz)
                records.append({"frame_number": frame_number, "detections": detections})
                writer.write(annotated)
                cv2.imshow("Original evidence", frame)
                cv2.imshow("Actual YOLO detections", annotated)
                frame_number += 1
                # A short wait keeps the window responsive; q stops inference.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            writer.release()
            cv2.destroyAllWindows()
        if frame_number == 0:
            raise SystemExit(f"Video contained no decodable frames: {source}")
    else:
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"OpenCV could not decode image: {source}")
        annotated, detections = infer_frame(model, frame, args.confidence, args.imgsz)
        records.append({"frame_number": None, "detections": detections})
        output_path = OUTPUT_DIR / f"{source.stem}_detections.png"
        cv2.imwrite(str(output_path), annotated)
        cv2.imshow("Original evidence", frame)
        cv2.imshow("Actual YOLO detections", annotated)
        print("Press any key to close the image windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    json_path = OUTPUT_DIR / f"{source.stem}_detections.json"
    json_path.write_text(json.dumps({"source": str(source.resolve()), "weights": str(configured_weights.resolve()), "frames": records}, indent=2), encoding="utf-8")
    detected_labels = sorted({d["label"] for frame_result in records for d in frame_result["detections"]})
    print(f"Processed {len(records)} actual frame(s). Classes detected: {detected_labels or 'none'}")
    print(f"Saved annotated output: {output_path}\nSaved prediction records: {json_path}")
    print("COCO predictions do not establish object identity or prove a theft; small or obscured objects may be missed.")


if __name__ == "__main__":
    main()
