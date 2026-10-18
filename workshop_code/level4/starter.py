"""Level 4 starter: follow a red region and record a line crossing once."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from workshop_code.vision import ASSETS
from workshop_code.level4.solution import process_video


def crossed_line(previous_x, current_x, line_x):
    # TODO: return True when the centroid moves from one side of the line to the other.
    return False


def main():
    event = process_video(ASSETS / "level4_corridor_exit.mp4", crossing_test=crossed_line)
    if event:
        print(f"Red-region crossing at clip time {event['seconds']:.2f}s (not a wall-clock time).")
    else:
        print("No crossing detected in the supplied clip.")


if __name__ == "__main__":
    main()
