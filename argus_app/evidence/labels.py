"""Canonical display labels for saved workshop evidence modules."""


_LEGACY_MODULE_LABELS = {
    "Level 2: Object Detection": "Level 2: The Missing Phone",
    "Level 3: Red Bag": "Level 3: Identify the Red Bag",
    "Level 4: Red Bag Tracking": "Level 4: Track the Red Bag",
    "Checkpoint Level 2: Reconstruct the Lab Scene": "Checkpoint Level 2: The Missing Phone",
    "Checkpoint Level 3: The Red Bag": "Checkpoint Level 3: Identify the Red Bag",
    "Checkpoint Level 4: Follow the Trail": "Checkpoint Level 4: Track the Red Bag",
}


def display_module_label(module: str) -> str:
    """Show canonical names for legacy entries without rewriting saved records."""
    return _LEGACY_MODULE_LABELS.get(str(module or ""), str(module or ""))
