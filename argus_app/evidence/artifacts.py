"""Helpers for saving genuine OpenCV outputs with notebook references."""

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def save_image(level: int, filename: str, image: np.ndarray) -> str:
    """Write an actual processed image under outputs/ and return its relative path."""
    if level not in (1, 2, 3, 4):
        raise ValueError("Level must be between 1 and 4.")
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("There is no processed image to save.")
    safe_name = Path(filename).name
    if not safe_name:
        raise ValueError("Output filename is empty.")
    destination = PROJECT_ROOT / "outputs" / f"level{level}" / safe_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"OpenCV could not save the processed output to {destination}.")
    return destination.relative_to(PROJECT_ROOT).as_posix()


def missing_manifest_assets() -> list[dict[str, str]]:
    """Return only evidence/record files declared by the manifest and absent on disk."""
    import json

    manifest_path = PROJECT_ROOT / "assets" / "evidence" / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_dir = PROJECT_ROOT / manifest.get("evidence_directory", "assets/evidence")
    records_dir = PROJECT_ROOT / manifest.get("records_directory", "assets/records")
    missing = []
    for item in manifest.get("assets", []):
        base = records_dir if item.get("kind") == "text_record" else evidence_dir
        if not (base / item.get("filename", "")).is_file():
            missing.append({
                "id": str(item.get("id", "—")),
                "filename": str(item.get("filename", "unknown")),
                "purpose": str(item.get("purpose", "")),
            })
    return missing


def documentary_records() -> list[dict[str, str]]:
    """Read only text records explicitly listed in the evidence manifest."""
    import json

    manifest_path = PROJECT_ROOT / "assets" / "evidence" / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_dir = PROJECT_ROOT / manifest.get("records_directory", "assets/records")
    records = []
    for item in manifest.get("assets", []):
        if item.get("kind") != "text_record":
            continue
        path = records_dir / item.get("filename", "")
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            content = f"Could not read this record: {exc}"
        records.append({"id": str(item.get("id", "—")), "filename": path.name, "path": str(path.resolve()), "content": content})
    return records
