"""Level 4: segment red regions and follow candidates through video frames.

Candidate continuity is geometric and can switch between red objects. It does
not identify a backpack or a person.
"""
from pathlib import Path
import argparse
import json
import math
import cv2
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from level_codes.level3 import COLOR_PRESETS
except ImportError:
    from level3 import COLOR_PRESETS

OUTPUT_DIR = ROOT / "outputs" / "level4"
TRACKING_MIN_AREA = 500
TRACKING_MAX_STEP_FRACTION = 0.18


def process_tracking_frame(frame, previous_centroid=None, trail=None, segments=None,
                           line_x=None, min_area=TRACKING_MIN_AREA,
                           max_step_fraction=TRACKING_MAX_STEP_FRACTION,
                           track_lost=False, crossing_direction="left_to_right",
                           h_min=170, h_max=10, s_min=70, s_max=255, v_min=40, v_max=255,
                           target_name="Red"):
    """Segment, select, and annotate colored tracking candidates in one video frame."""
    trail = list(trail or [])
    segments = [list(segment) for segment in (segments or [])]
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if h_min <= h_max:
        target_mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min], np.uint8), np.array([h_max, s_max, v_max], np.uint8))
    else:
        low = cv2.inRange(hsv, np.array([0, s_min, v_min], np.uint8), np.array([h_max, s_max, v_max], np.uint8))
        high = cv2.inRange(hsv, np.array([h_min, s_min, v_min], np.uint8), np.array([179, s_max, v_max], np.uint8))
        target_mask = cv2.bitwise_or(low, high)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel)
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > width * height * 0.75:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        candidates.append((contour, center, area))

    annotated = frame.copy()
    if line_x is not None:
        cv2.line(annotated, (int(line_x), 0), (int(line_x), height - 1), (255, 180, 0), 2)
    for contour, _, _ in candidates:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), (130, 130, 130), 1)

    selected = None
    if track_lost:
        note = "Track lost after a gap; reset or seek to begin a new segment. No replacement candidate selected."
    elif candidates and previous_centroid is None:
        selected = max(candidates, key=lambda item: item[2])
        note = f"Largest {target_name.lower()} candidate seeded; physical identity is unverified."
    elif candidates:
        max_step = max(40.0, math.hypot(width, height) * max_step_fraction)
        nearby = [item for item in candidates if math.dist(previous_centroid, item[1]) <= max_step]
        if nearby:
            selected = min(nearby, key=lambda item: math.dist(previous_centroid, item[1]))
            note = "Nearest candidate within the continuity limit."
        else:
            note = "No candidate within the continuity limit; track gap."
    else:
        note = f"No {target_name.lower()} candidate; track gap."

    crossed = False
    centroid = None
    if selected is None:
        if previous_centroid is not None:
            track_lost = True
        previous_centroid = None
        if trail:
            segments.append(trail)
            trail = []
    else:
        contour, centroid, _ = selected
        if previous_centroid is not None and line_x is not None:
            if crossing_direction == "right_to_left":
                crossed = previous_centroid[0] > line_x >= centroid[0]
            elif crossing_direction == "either":
                crossed = (
                    previous_centroid[0] < line_x <= centroid[0]
                    or previous_centroid[0] > line_x >= centroid[0]
                )
            else:
                crossed = previous_centroid[0] < line_x <= centroid[0]
        previous_centroid = centroid
        trail.append(centroid)
        x, y, box_width, box_height = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), (0, 220, 255), 2)
        cv2.circle(annotated, centroid, 5, (0, 255, 0), -1)
    visible_segments = segments + ([trail] if trail else [])
    for segment in visible_segments:
        for start, end in zip(segment[-120:], segment[-119:]):
            cv2.line(annotated, start, end, (0, 255, 0), 2)
    cv2.putText(annotated, note, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return {"result": annotated, "mask": target_mask, "candidate_count": len(candidates),
            "centroid": centroid, "previous_centroid": previous_centroid, "crossed": crossed,
            "trail": trail, "segments": segments, "note": note, "candidates": candidates,
            "track_lost": track_lost}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Path to evidence video")
    parser.add_argument("--color", choices=list(COLOR_PRESETS.keys()), default="Red", help="Target color to track")
    parser.add_argument("--line-x-fraction", type=float, default=None, help="Doorway x-position as fraction of frame width; inspect video first")
    parser.add_argument("--min-area", type=float, default=100.0)
    parser.add_argument("--max-step-fraction", type=float, default=0.18)
    parser.add_argument("--crossing-direction", choices=("left_to_right", "right_to_left"), default="left_to_right")
    args = parser.parse_args()
    if args.line_x_fraction is not None and not 0 <= args.line_x_fraction <= 1:
        raise SystemExit("--line-x-fraction must be between 0 and 1")
    preset = COLOR_PRESETS.get(args.color, COLOR_PRESETS["Red"])
    video_path = args.video.expanduser()
    if not video_path.is_absolute() and not video_path.is_file():
        video_path = ROOT / video_path
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = None
    previous_centroid = None
    trail, segments, crossings = [], [], []
    frame_number = tracked_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if writer is None:
                path = OUTPUT_DIR / "tracked_corridor.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise SystemExit(f"Could not create output video: {path}")
            line_x = round(width * args.line_x_fraction) if args.line_x_fraction is not None else None
            result = process_tracking_frame(
                frame, previous_centroid, trail, segments, line_x, args.min_area, args.max_step_fraction,
                crossing_direction=args.crossing_direction,
                h_min=preset["h_min"], h_max=preset["h_max"],
                s_min=preset["s_min"], s_max=preset["s_max"],
                v_min=preset["v_min"], v_max=preset["v_max"],
                target_name=args.color,
            )
            previous_centroid, trail, segments = result["previous_centroid"], result["trail"], result["segments"]
            tracked_frames += int(result["centroid"] is not None)
            if result["crossed"]:
                crossings.append({"frame": frame_number, "seconds": frame_number / fps, "centroid": result["centroid"], "line_x": line_x})
            writer.write(result["result"])
            cv2.imshow("Source corridor video", frame)
            cv2.imshow("Candidate tracking", result["result"])
            frame_number += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    if frame_number == 0:
        raise SystemExit(f"No decodable frames in video: {video_path}")
    summary = {"source": str(video_path.resolve()), "frames_processed": frame_number, "frames_with_centroid": tracked_frames,
               "entrance_line_x_fraction": args.line_x_fraction, "crossings": crossings,
               "note": "Colour candidates are not identity tracking; crossings are evaluated only against a configured line."}
    summary_path = OUTPUT_DIR / "tracking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Processed {frame_number} frames; {tracked_frames} included a selected candidate; crossings: {len(crossings)}.")
    print(f"Saved video and summary under {OUTPUT_DIR}. Red color does not establish identity.")


if __name__ == "__main__":
    main()
