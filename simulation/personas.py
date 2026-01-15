from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import yaml

from .events import Event

OPTIONAL_FIELDS = {
    "profession",
    "background",
    "family",
    "political_ideology",
    "political_party",
}


@dataclass
class Persona:
    """
    Represents an agent persona with a reusable prompt template and traits.
    Required attributes: name, country (can be blank for default agent).
    Optional: gender, age, education, profession, background, family, political_ideology, political_party, traits.
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
        "You are {name}. {profile}React to the event: {event}"
    )
    metadata: Dict[str, str] = field(default_factory=dict)

    def build_prompt(
        self,
        event: Event,
        context: Optional[Dict[str, str]] = None,
        include_optional: Optional[Set[str]] = None,
    ) -> str:
        """
        Render the persona prompt with the given event and optional context.
        include_optional controls which optional attributes are injected; defaults to none.
        """
        include_optional = include_optional or set()
        optional_data = {
            key: getattr(self, key, "")
            if key in include_optional and getattr(self, key, "")
            else ""
            for key in OPTIONAL_FIELDS
        }
        profile_parts = []
        for label, value in [
            ("Country", self.country),
            ("Gender", self.gender),
            ("Age", self.age),
            ("Education", self.education),
            ("Traits", ", ".join(sorted(self.traits)) if self.traits else ""),
        ]:
            if value:
                profile_parts.append(f"{label}: {value}")
        for key, value in optional_data.items():
            if value:
                profile_parts.append(f"{key.replace('_', ' ').title()}: {value}")
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
            prompt_template = country_prompts.get(country) or country_prompts.get("default")
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
                metadata=entry.get("metadata", {}),
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
    Persona with minimal attributes to react only to the event.
    """
    return Persona(name=name, country="", gender="", age="", education="", traits=set())


def load_country_prompts_from_yaml(path: str | Path) -> Dict[str, str]:
    """
    Load country-level prompt templates keyed by country code/name.
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, str] = yaml.safe_load(handle) or {}
    return raw
