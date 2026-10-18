"""Instructor reference: continuity-aware red-candidate route reconstruction."""

import json
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workshop_code.vision import ASSETS, OUTPUTS, red_regions


def crossed_line(previous_x, current_x, line_x):
    return (previous_x < line_x <= current_x) or (previous_x > line_x >= current_x)


def process_video(path, line_fraction=0.60, crossing_test=crossed_line):
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video is missing or unreadable: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    output_dir = OUTPUTS / "level4"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "exit_tracking.mp4"
    writer = None
    frame_number = 0
    previous_centroid = None
    current_segment = []
    segments = []
    event = None
    candidate_frames = tracked_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            line_x = int(width * line_fraction)
            _, regions = red_regions(frame, minimum_area=100)
            annotated = frame.copy()
            cv2.line(annotated, (line_x, 0), (line_x, height - 1), (255, 180, 0), 2)
            candidates = [region for region in regions]
            candidate_frames += bool(candidates)
            selected = None
            if previous_centroid is not None and candidates:
                max_step = max(40.0, math.hypot(width, height) * 0.18)
                selected = min(candidates, key=lambda item: math.dist(previous_centroid, item["centroid"]))
                if math.dist(previous_centroid, selected["centroid"]) > max_step:
                    selected = None
            elif candidates:
                selected = max(candidates, key=lambda item: item["area"])

            for region in candidates:
                x, y, w, h = region["box"]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (120, 120, 120), 1)
            if selected is None:
                previous_centroid = None
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            else:
                centroid = selected["centroid"]
                tracked_frames += 1
                if previous_centroid is not None and event is None and crossing_test(previous_centroid[0], centroid[0], line_x):
                    event = {
                        "frame": frame_number,
                        "seconds": frame_number / fps if fps > 0 else None,
                        "centroid": list(centroid),
                        "line_x": line_x,
                        "time_basis": "clip-relative",
                    }
                current_segment.append(centroid)
                previous_centroid = centroid
                x, y, w, h = selected["box"]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 255), 2)
                cv2.circle(annotated, centroid, 5, (0, 255, 0), -1)
            visible_segments = segments + ([current_segment] if current_segment else [])
            for segment in visible_segments:
                for p1, p2 in zip(segment[-120:], segment[-119:]):
                    cv2.line(annotated, p1, p2, (0, 255, 0), 2)
            if writer is None:
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 25.0, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV could not create the annotated MP4 output.")
            writer.write(annotated)
            frame_number += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frame_number == 0:
        raise ValueError(f"No decodable frames in video: {path}")
    if current_segment:
        segments.append(current_segment)
    summary = {
        "source": str(path.resolve()),
        "frames_evaluated": frame_number,
        "frames_with_red_candidates": candidate_frames,
        "frames_with_continuous_centroid": tracked_frames,
        "trajectory_segments": segments,
        "exit_crossing": event,
        "annotated_video": str(output_path.relative_to(Path(__file__).resolve().parents[2]).as_posix()),
        "limitation": "HSV colour candidates may be distractors. Gaps or implausible centroid jumps break continuity. A clip-relative crossing does not identify the bag or carrier.",
    }
    (output_dir / "tracking_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    result = process_video(ASSETS / "level4_corridor_exit.mp4")
    crossing = result["exit_crossing"]
    print(f"Evaluated {result['frames_evaluated']} supplied video frames; tracked candidate frames: {result['frames_with_continuous_centroid']}.")
    print(f"Red-region exit crossing: {crossing if crossing else 'not supported by the observed centroid track'}.")
    print("Times are clip-relative unless supplied evidence establishes a wall-clock offset.")


if __name__ == "__main__":
    main()
