"""Optional Ultralytics fine-tuning from a labeled YOLO-format dataset."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Ultralytics dataset YAML")
    parser.add_argument("--names", help="Optional comma-separated class names overriding names in the YAML")
    parser.add_argument("--weights", default="yolo11n.pt", help="Pretrained starting weights (default: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Ultralytics device; defaults to CPU")
    parser.add_argument("--output", type=Path, default=Path("models/custom"))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset YAML not found: {args.data}")
    if args.epochs < 1 or args.imgsz < 32:
        parser.error("--epochs must be positive and --imgsz must be at least 32")

    try:
        import yaml
        from ultralytics import YOLO
    except ImportError as exc:
        print("Training needs Ultralytics and PyYAML. Install project dependencies first.", file=sys.stderr)
        return 2

    try:
        config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Dataset YAML must contain a mapping.")
        if args.names:
            names = [name.strip() for name in args.names.split(",") if name.strip()]
            if not names:
                raise ValueError("--names did not contain any class names.")
            config["names"] = names
            config["nc"] = len(names)
            configured_root = config.get("path")
            dataset_root = Path(configured_root) if configured_root else args.data.parent.resolve()
            if not dataset_root.is_absolute():
                dataset_root = (args.data.parent / dataset_root).resolve()
            config["path"] = str(dataset_root)
            args.output.mkdir(parents=True, exist_ok=True)
            effective_yaml = args.output / "dataset.training.yaml"
            effective_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        else:
            effective_yaml = args.data.resolve()
        if not config.get("names"):
            raise ValueError("Add class names to the dataset YAML or pass --names.")

        model = YOLO(args.weights)
        project = args.output.resolve()
        model.train(
            data=str(effective_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            project=str(project),
            name="training",
            exist_ok=True,
        )
        best = project / "training" / "weights" / "best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"Training finished without best.pt at {best}")
        destination = project / "best.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, destination)
        print(f"Fine-tuned weights saved to {destination}")
        print("Set ARGUS_YOLO_WEIGHTS to this path, or place the file at models/custom/best.pt.")
        return 0
    except Exception as exc:
        print(f"YOLO fine-tuning failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
