"""Level 2 starter: inspect real YOLO predictions from an image or lab clip."""

import argparse
import os
import sys
from pathlib import Path

import cv2
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workshop_code.vision import ASSETS, OUTPUTS, read_image, write_image, save_json


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv"}


def main():
    parser = argparse.ArgumentParser(description="Run actual pretrained YOLO on laboratory evidence.")
    parser.add_argument("--source", type=Path, help="Evidence image or video. Defaults to the manifest's Level 2 image.")
    args = parser.parse_args()
    default_image = ASSETS / "level2_lab_before.png"
    source = args.source.expanduser() if args.source else default_image
    if not source.is_file():
        raise FileNotFoundError(
            f"Level 2 evidence is missing: {source}. Supply the lab video or image under assets/evidence, "
            "or pass --source PATH. No replacement footage is generated."
        )

    weights_setting = os.environ.get("ARGUS_YOLO_WEIGHTS", "").strip()
    weights = Path(weights_setting).expanduser() if weights_setting else ROOT / "yolo11n.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights}. Add a local checkpoint or set ARGUS_YOLO_WEIGHTS.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for Level 2. Install the project requirements.") from exc

    model = YOLO(str(weights))
    is_video = source.suffix.lower() in VIDEO_SUFFIXES
    # TODO 1: run model.predict on this actual source; stream video frames.
    predictions = None
    # Hint: predictions = model.predict(source=str(source), stream=is_video,
    #        imgsz=640, conf=0.25, device="cpu", verbose=False)
    if predictions is None:
        raise NotImplementedError("Complete TODO 1 to run YOLO inference on the selected evidence.")

    frame_records = None
    class_frame_hits = {}
    evaluated_frames = 0
    # TODO 2: for each actual result, collect its frame number, class labels,
    # confidence, and xyxy box. Increment evaluated_frames for every result,
    # even when you save only selected plotted frames. Count each class at most
    # once per frame in class_frame_hits. Never create a missed detection.
    if frame_records is None:
        raise NotImplementedError("Complete TODO 2 to collect actual model predictions.")

    # TODO 3: save the actual plotted outputs and JSON summary under outputs/.
    output = OUTPUTS / "level2"
    output.mkdir(parents=True, exist_ok=True)
    if not frame_records:
        raise RuntimeError("YOLO/OpenCV returned no decodable frames.")
    for item in frame_records:
        suffix = f"frame_{item['frame_number']:06d}" if item["frame_number"] is not None else "image"
        write_image(output / f"{source.stem}_{suffix}_detections.png", item["annotated"])
    save_json(output / f"{source.stem}_detections.json", {
        "source": str(source.resolve()),
        "evaluated_frames": evaluated_frames,
        "class_frame_hits": class_frame_hits,
        "frames": [{"frame_number": item["frame_number"], "detections": item["detections"]} for item in frame_records],
        "note": "Predictions cover supported classes only. Inconsistent or missing detections do not establish object absence or identity.",
    })
    print(f"Saved actual YOLO output for {len(frame_records)} inspected frame(s) from {source.name}.")
    print("Review which supported classes appear consistently; the clip is baseline scene analysis, not proof of a theft event.")


if __name__ == "__main__":
    main()
