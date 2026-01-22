from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


@dataclass
class Event:
    """
    A published event the personas respond to.
    """

    title: str
    description: str
    country: Optional[str] = None
    title_en: Optional[str] = None
    description_en: Optional[str] = None
    if_red_tape: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)

    def serialize(self) -> str:
        """
        Flatten the event for use in prompt text.
        """
        country = f" ({self.country})" if self.country else ""
        summary = f"{self.title}{country}: {self.description}"

        parts = [summary]
        if self.if_red_tape:
            parts.append("Flag: red-tape scenario.")
        if self.metadata:
            meta = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
            parts.append(f"Metadata: {meta}")

        return " ".join(parts)


def load_events_from_yaml(path: str | Path) -> List[Event]:
    """
    Load a list of Event definitions from a YAML file.
    Each entry should contain: title (str), description (str).
    Optional: country, if_red_tape (bool), metadata (dict), title_en, description_en.
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        raw: Iterable[Dict] = yaml.safe_load(handle) or []

    events: List[Event] = []
    for entry in raw:
        events.append(
            Event(
                title=entry["title"],
                description=entry.get("description", ""),
                country=entry.get("country"),
                title_en=entry.get("title_en"),
                description_en=entry.get("description_en"),
                if_red_tape=bool(entry.get("if_red_tape", False)),
                metadata=entry.get("metadata", {}),
            )
        )
    return events
