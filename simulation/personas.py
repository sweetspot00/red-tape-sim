from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

import yaml

from .events import Event


@dataclass
class Persona:
    """
    Represents an agent persona with a reusable prompt template and traits.
    All attributes are treated as required in prompts; missing values are rendered as empty strings.
    """

    name: str
    country: str = ""
    gender: str = ""
    age: str = ""
    education: str = ""
    traits: Set[str] = field(default_factory=set)
    profession: str = ""
    background: str = ""
    family: str = ""
    political_ideology: str = ""
    political_party: str = ""
    prompt_template: str = (
        "You are {name}. {profile}. React to the event: {event}"
    )
    metadata: Dict[str, str] = field(default_factory=dict)
    history_attitude: Union[List[Dict[str, str]], List[str], str] = field(default_factory=list)

    def build_prompt(
        self,
        event: Event,
        context: Optional[Dict[str, str]] = None,
        include_optional: Optional[Set[str]] = None,
    ) -> str:
        """
        Render the persona prompt with the given event and context.
        All persona attributes are injected; include_optional is ignored for compatibility.
        """
        history_attitude = self.history_attitude
        if isinstance(history_attitude, list):
            # Flatten list of dicts or strings into a single string
            flattened = []
            for h in history_attitude:
                if isinstance(h, dict):
                    flattened.extend([str(v) for v in h.values()])
                else:
                    flattened.append(str(h))
            history_text = " ".join([t for t in flattened if t])
        else:
            history_text = str(history_attitude)

        def _val(text: str) -> str:
            return text if text else "unspecified"

        profile_parts = [
            f"Country: {_val(self.country)}",
            f"Gender: {_val(self.gender)}",
            f"Age: {_val(self.age)}",
            f"Education: {_val(self.education)}",
            f"Traits: {_val(', '.join(sorted(self.traits)) if self.traits else '')}",
            f"Profession: {_val(self.profession)}",
            f"Background: {_val(self.background)}",
            f"Family: {_val(self.family)}",
            f"Political Ideology: {_val(self.political_ideology)}",
            f"Political Party: {_val(self.political_party)}",
            f"History Attitude: {_val(history_text)}",
        ]
        profile_text = " | ".join(profile_parts)
        profile = f"Profile -> {profile_text}. " if profile_text else ""
        data = {
            "name": self.name,
            "profile": profile,
            "event": event.serialize(),
            "event_name": event.title,
            "event_description": event.description,
            "event_country": event.country or "",
            "event_if_red_tape": "yes" if event.if_red_tape else "no",
            **(context or {}),
        }
        return self.prompt_template.format(**data)


def load_personas(definitions: Sequence[Dict], country_prompts: Optional[Dict[str, str]] = None) -> List[Persona]:
    """
    Build Persona objects from serializable definitions.
    """
    personas: List[Persona] = []
    for entry in definitions:
        country = entry.get("country", "")
        prompt_template = entry.get("prompt_template")
        if not prompt_template and country_prompts:
            key = country.lower() if country is not None else ""
            prompt_template = country_prompts.get(key) or country_prompts.get("default")
        personas.append(
            Persona(
                name=entry["name"],
                country=country,
                gender=entry.get("gender", ""),
                age=entry.get("age", ""),
                education=entry.get("education", ""),
                traits=set(entry.get("traits", [])),
                profession=entry.get("profession", ""),
                background=entry.get("background", ""),
                family=entry.get("family", ""),
                political_ideology=entry.get("political_ideology", ""),
                political_party=entry.get("political_party", ""),
                prompt_template=prompt_template or Persona.prompt_template,
                history_attitude=entry.get("history_attitude", []),
            )
        )
    return personas


def load_personas_from_yaml(
    path: str | Path,
    *,
    country_prompts: Optional[Dict[str, str]] = None,
) -> List[Persona]:
    """
    Load persona definitions from a YAML file.
    Each entry should contain: name, country. Other attributes optional.
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        raw: Iterable[Dict] = yaml.safe_load(handle) or []
    return load_personas(raw, country_prompts=country_prompts)


def default_persona(name: str = "Default Agent") -> Persona:
    """
    Persona with all fields present; intended for fallback use.
    """
    return Persona(
        name=name,
        country="",
        gender="",
        age="",
        education="",
        traits=set(),
        profession="",
        background="",
        family="",
        political_ideology="",
        political_party="",
        history_attitude=[],
    )


def load_country_prompts_from_yaml(path: str | Path) -> Dict[str, str]:
    """
    Load country-level prompt templates keyed by country code/name.
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, str] = yaml.safe_load(handle) or {}
    return { (k or "").strip().lower(): v for k, v in raw.items() }
