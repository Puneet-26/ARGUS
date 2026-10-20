"""ARGUS evidence registry.

A simple registry class for storing lab results / evidence.  No content is
stored here by default; students or the UI can register entries as needed.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import tempfile
from typing import List, Optional


@dataclasses.dataclass
class EvidenceEntry:
    """A single piece of evidence recorded by a lab."""
    module: str
    kind: str  # e.g. "image", "text", "number", "detection", "tracking"
    value: object
    description: str = ""
    technique: str = ""
    observation: str = ""
    interpretation: str = ""
    uncertainty: str = "Low"  # Low, Medium, High
    timestamp: str = ""
    source_filename: str = ""
    source_path: str = ""
    output_path: str = ""
    source_timestamp: str = ""
    frame_number: Optional[int] = None
    unresolved_question: str = ""
    evidence_id: str = ""
    completion_mode: str = "independent"
    limitation: str = ""

    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.observation and self.description:
            self.observation = self.description


class EvidenceRegistry:
    """Simple in-memory registry of evidence entries."""

    def __init__(self) -> None:
        self._entries: List[EvidenceEntry] = []
        self.persistence_error = ""
        self.storage_path = self.default_storage_path()
        if self.storage_path.is_file():
            try:
                self.load(self.storage_path)
            except (OSError, ValueError, TypeError) as exc:
                self.persistence_error = f"Could not load saved notebook ({exc})"

    @staticmethod
    def default_storage_path() -> pathlib.Path:
        """Return the per-user application data location for case records."""
        if os.name == "nt":
            base = pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home() / "AppData" / "Local"))
        else:
            base = pathlib.Path(os.environ.get("XDG_DATA_HOME", pathlib.Path.home() / ".local" / "share"))
        return base / "ARGUS" / "evidence_notebook.json"

    def _persist(self) -> bool:
        """Persist the current notebook without leaving a partial JSON file."""
        temporary = None
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.storage_path.parent,
                prefix="evidence_notebook_", suffix=".tmp", delete=False,
            ) as stream:
                temporary = pathlib.Path(stream.name)
                json.dump([dataclasses.asdict(entry) for entry in self._entries], stream, indent=2, default=str)
                stream.write("\n")
            temporary.replace(self.storage_path)
            self.persistence_error = ""
            return True
        except (OSError, TypeError, ValueError) as exc:
            self.persistence_error = f"Could not save the notebook to {self.storage_path}: {exc}"
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            return False

    def add(self, entry: EvidenceEntry) -> bool:
        """Add and persist one entry, rolling back in memory if persistence fails."""
        if not isinstance(entry, EvidenceEntry):
            raise TypeError("Expected an EvidenceEntry instance.")
        if (
            not isinstance(entry.module, str) or not entry.module.strip()
            or not isinstance(entry.kind, str) or not entry.kind.strip()
        ):
            raise ValueError("Evidence entries need a module and kind.")
        self._entries.append(entry)
        if self._persist():
            return True
        self._entries.pop()
        return False

    def remove(self, index: int) -> bool:
        """Remove an entry, restoring it if the updated notebook cannot be saved."""
        if 0 <= index < len(self._entries):
            entry = self._entries.pop(index)
            if self._persist():
                return True
            self._entries.insert(index, entry)
            return False
        return False

    def clear(self) -> bool:
        """Clear all entries, restoring them if the empty notebook cannot be saved."""
        previous_entries = self._entries
        self._entries = []
        if self._persist():
            return True
        self._entries = previous_entries
        return False

    def all(self) -> List[EvidenceEntry]:
        """Return all entries (copy)."""
        return list(self._entries)

    def count(self) -> int:
        """Return total entries count."""
        return len(self._entries)

    def save(self, path: pathlib.Path) -> None:
        """Save entries to a JSON file."""
        data = [
            {
                "module": e.module,
                "kind": e.kind,
                "value": e.value,
                "description": e.description,
                "technique": e.technique,
                "observation": e.observation,
                "interpretation": e.interpretation,
                "uncertainty": e.uncertainty,
                "timestamp": e.timestamp,
                "source_filename": e.source_filename,
                "source_path": e.source_path,
                "output_path": e.output_path,
                "source_timestamp": e.source_timestamp,
                "frame_number": e.frame_number,
                "unresolved_question": e.unresolved_question,
                "evidence_id": e.evidence_id,
                "completion_mode": e.completion_mode,
                "limitation": e.limitation,
            }
            for e in self._entries
        ]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def load(self, path: pathlib.Path) -> None:
        """Load entries from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        self._entries = [
            EvidenceEntry(
                module=d.get("module", "General"),
                kind=d.get("kind", "text"),
                value=d.get("value", ""),
                description=d.get("description", ""),
                technique=d.get("technique", ""),
                observation=d.get("observation", d.get("description", "")),
                interpretation=d.get("interpretation", ""),
                uncertainty=d.get("uncertainty", "Low"),
                timestamp=d.get("timestamp", ""),
                source_filename=d.get("source_filename", ""),
                source_path=d.get("source_path", ""),
                output_path=d.get("output_path", ""),
                source_timestamp=d.get("source_timestamp", ""),
                frame_number=d.get("frame_number"),
                unresolved_question=d.get("unresolved_question", ""),
                evidence_id=d.get("evidence_id", ""),
                completion_mode=d.get("completion_mode", "independent"),
                limitation=d.get("limitation", ""),
            )
            for d in data
        ]


# Module-level singleton registry for convenience
registry = EvidenceRegistry()
