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
    family: str = ""
    political_ideology: str = ""
    political_party: str = ""
    prompt_template: str = (
        "You are {name}. {profile}. React to the event: {event}"
    )
    history_attitude: Union[List[Dict[str, str]], List[str], str] = field(default_factory=list)
    power_distance: float = 0.0
    individualism: float = 0.0
    masculinity: float = 0.0
    uncertainty_avoidance: float = 0.0
    long_term_orientation: float = 0.0
    indulgence: float = 0.0

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

        profile_prompt = f"""
        You are {self.name} from {_val(self.country)}. You are a {_val(self.age)} years old {_val(self.gender)}. You have a {_val(self.education)} degree and work as a {_val(self.profession)}. 
        Your personality traits include {_val(', '.join(sorted(self.traits)) if self.traits else '')}. You are {_val(self.family)}. Your political ideology is {_val(self.political_ideology)} and you align with {_val(self.political_party)} party.
        You got some opinions and have some attitudes to your region's policy:  {_val(history_text)}.
        Culture factor wise, you got a score of {self.power_distance} in power distance, {self.individualism} in individualism, {self.masculinity} in masculinity, {self.uncertainty_avoidance} in uncertainty avoidance, {self.long_term_orientation} in long term orientation, and {self.indulgence} in indulgence.   
        Larger score denotes stronger tendency in that dimension.
        """

        data = {
            "name": self.name,
            "profile": profile_prompt,
            "event": event.serialize(),
            "event_name": event.title,
            "event_description": event.description,
            "event_country": event.country or "",
            # "event_if_red_tape": "yes" if event.if_red_tape else "no",
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
                family=entry.get("family", ""),
                political_ideology=entry.get("political_ideology", ""),
                political_party=entry.get("political_party", ""),
                prompt_template=prompt_template or Persona.prompt_template,
                history_attitude=entry.get("history_attitude", []),
                power_distance=float(entry.get("power_distance", 0.0)),
                individualism=float(entry.get("individualism", 0.0)),
                masculinity=float(entry.get("masculinity", 0.0)),
                uncertainty_avoidance=float(entry.get("uncertainty_avoidance", 0.0)),
                long_term_orientation=float(entry.get("long_term_orientation", 0.0)),
                indulgence=float(entry.get("indulgence", 0.0)),
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
        family="",
        political_ideology="",
        political_party="",
        history_attitude=[],
        power_distance=0.0,
        individualism=0.0,
        masculinity=0.0,
        uncertainty_avoidance=0.0,
        long_term_orientation=0.0,
        indulgence=0.0,
    )


def load_country_prompts_from_yaml(path: str | Path) -> Dict[str, str]:
    """
    Load country-level prompt templates keyed by country code/name.
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, str] = yaml.safe_load(handle) or {}
    return { (k or "").strip().lower(): v for k, v in raw.items() }
