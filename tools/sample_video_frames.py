"""Sample supplied CCTV clips at a modest interval for optional annotation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Supplied CCTV video")
    parser.add_argument("--output", type=Path, default=Path("datasets/frames"))
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between extracted frames (default: 2)")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        parser.error(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count else 0
    args.output.mkdir(parents=True, exist_ok=True)
    timestamps: dict[str, str] = {}
    index = 0
    second = 0.0
    while not duration or second < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        name = f"frame_{index + 1:05d}.jpg"
        if not cv2.imwrite(str(args.output / name), frame):
            cap.release()
            parser.error(f"Could not write {args.output / name}")
        timestamps[name] = f"{second:.2f}s"
        index += 1
        second += args.interval
    cap.release()
    (args.output / "timestamps.json").write_text(json.dumps(timestamps, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {index} frames at {args.interval:g}s intervals to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
