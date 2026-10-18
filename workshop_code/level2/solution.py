"""Instructor reference: actual pretrained YOLO inference on one image."""

import os
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workshop_code.vision import ASSETS, OUTPUTS, ROOT, read_image, write_image, save_json


def run_detector(image, image_path):
    configured = os.environ.get("ARGUS_YOLO_WEIGHTS", "").strip()
    weights = Path(configured).expanduser() if configured else ROOT / "yolo11n.pt"
    if not weights.is_file():
        return None, f"Detector unavailable: model file not found at {weights}. Set ARGUS_YOLO_WEIGHTS to a compatible local .pt checkpoint."
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        return None, f"Detector unavailable: install the project requirements ({exc})."
    try:
        model = YOLO(str(weights))
        result = model.predict(image, device="cpu", conf=0.25, verbose=False)[0]
    except Exception as exc:
        return None, f"Detector inference failed for {image_path}: {exc}"
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id] if not isinstance(model.names, dict) else model.names.get(class_id, str(class_id))
            detections.append({
                "label": str(label),
                "confidence": float(box.conf[0]),
                "xyxy": [round(float(value), 1) for value in box.xyxy[0].tolist()],
            })
    return (result.plot(), detections), "Actual local-model predictions; supported classes depend on the checkpoint."


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run actual pretrained YOLO on Level 2 evidence.")
    parser.add_argument("--source", type=Path, default=ASSETS / "level2_lab_before.png", help="Image or video evidence path")
    args = parser.parse_args()
    source = args.source.expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Evidence is missing or unreadable: {source}. Supply a real source; no sample is generated.")
    output = OUTPUTS / "level2"
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv"}
    if source.suffix.lower() in video_suffixes:
        frames, class_frame_hits, evaluated = run_video(source)
        if not frames:
            raise RuntimeError(f"OpenCV could not decode any frames from {source}.")
        for item in frames:
            write_image(output / f"{source.stem}_frame_{item['frame_number']:06d}_detections.png", item["annotated"])
        save_json(output / f"{source.stem}_detections.json", {
            "source": str(source.resolve()), "evaluated_frames": evaluated,
            "class_frame_hits": class_frame_hits,
            "frames": [{"frame_number": item["frame_number"], "detections": item["detections"]} for item in frames],
            "note": "Only actual model outputs are saved; a missed detection does not prove absence or identity.",
        })
        print(f"YOLO evaluated {evaluated} actual frames and saved {len(frames)} review frames.")
    else:
        image = read_image(source)
        prediction, note = run_detector(image, source)
        if prediction is None:
            raise RuntimeError(note)
        annotated, detections = prediction
        write_image(output / f"{source.stem}_detections.png", annotated)
        save_json(output / f"{source.stem}_detections.json", {"source": str(source.resolve()), "detections": detections, "note": note})
        print(f"{len(detections)} actual model detection(s) from {source.name}.")
    print("Review supported-class detections only. This baseline clip does not establish a theft event or object identity.")


def run_video(path, max_review_frames=36):
    """Infer on every real frame and retain a bounded set of actual review frames."""
    import math
    configured = os.environ.get("ARGUS_YOLO_WEIGHTS", "").strip()
    weights = Path(configured).expanduser() if configured else ROOT / "yolo11n.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Detector weights not found at {weights}.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for Level 2.") from exc
    model = YOLO(str(weights))
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Video is missing or unreadable: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(fps * 0.5)) if fps else 15)
    if declared:
        stride = max(stride, math.ceil(declared / max(1, max_review_frames - 1)))
    frames, class_frame_hits = [], {}
    evaluated = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number = evaluated
            result = model.predict(frame, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    label = model.names.get(class_id, str(class_id)) if isinstance(model.names, dict) else model.names[class_id]
                    detections.append({"label": str(label), "confidence": float(box.conf[0]), "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()]})
            evaluated += 1
            for label in {detection["label"] for detection in detections}:
                class_frame_hits[label] = class_frame_hits.get(label, 0) + 1
            if frame_number % stride == 0 and len(frames) < max_review_frames - 1:
                frames.append({"frame_number": frame_number, "detections": detections, "annotated": result.plot()})
    finally:
        capture.release()
    if evaluated and (evaluated - 1) % stride != 0 and len(frames) < max_review_frames:
        frames.append({"frame_number": evaluated - 1, "detections": detections, "annotated": result.plot()})
    return frames, class_frame_hits, evaluated


if __name__ == "__main__":
    main()
