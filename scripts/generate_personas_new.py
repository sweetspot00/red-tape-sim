"""
Generate personas for 北京, Hongkong, and Germany using the existing LLM client.
Outputs `data/personas_new.yaml` with 200 personas per region (600 total) and includes the fixed Mingyu_137 record provided by the user.

Environment:
- OPENAI_API_KEY (required for LLMClient)
- Optionally OPENAI_BASE_URL / OPENAI_MODEL
"""

import sys
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from threading import Lock

import yaml

# Import project modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from simulation.llm import LLMClient  # noqa: E402


OUTPUT_FILE = Path("data/personas_new.yaml")
TARGET_PER_REGION = 200
MAX_WORKERS = 6
LOG_LEVEL = os.getenv("PERSONA_LOG_LEVEL", "INFO").upper()  # allow override via env
file_lock = Lock()

FIXED_PERSONA: Dict[str, Any] = {
    "name": "Mingyu_137",
    "country": "北京",
    "gender": "female",
    "age": "62",
    "education": "master",
    "traits": ["pessimistic", "traditional", "urban"],
    "profession": "公务员",
    "family": "married",
    "political_ideology": "socialist",
    "political_party": "共产党",
    "history_attitude": "她认为中国在新冠疫情期间采取的策略是正确的，即居家隔离以避免病毒传播。她还认为中国应该在外交事务上采取强硬立场。",
}


@dataclass
class RegionConfig:
    country: str
    prefix: str
    history_language: str
    generate: int


REGIONS: List[RegionConfig] = [
    RegionConfig(country="北京", prefix="BJ", history_language="zh", generate=TARGET_PER_REGION - 1),
    RegionConfig(country="Hongkong", prefix="HK", history_language="zh", generate=TARGET_PER_REGION),
    RegionConfig(country="Germany", prefix="DE", history_language="en", generate=TARGET_PER_REGION),
]


class ProgressBar:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self._lock = Lock()

    def step(self) -> None:
        with self._lock:
            self.done += 1
            width = 40
            filled = int(width * self.done / self.total)
            bar = "#" * filled + "-" * (width - filled)
            print(f"\rGenerating personas: |{bar}| {self.done}/{self.total}", end="", flush=True)

    def finish(self) -> None:
        print()


def build_prompt(name: str, region: RegionConfig) -> str:
    history_hint = (
        "Write history_attitude in Simplified Chinese, 1-2 sentences focused on their view of recent policies or historical events relevant to the region."
        if region.history_language == "zh"
        else "Write history_attitude in concise English (1-2 sentences) about their view of recent policies or historical events relevant to the region."
    )
    return (
        "Create ONE persona as YAML with keys: name, country, gender, age, education, traits, profession, family, "
        "political_ideology, political_party, history_attitude.\n"
        f"- Use this exact name: {name}\n"
        f"- Set country exactly to: {region.country}\n"
        "- gender: male or female.\n"
        "- age: a string between 18 and 80.\n"
        "- education: one of middle_school, high_school, bachelor, master, phd.\n"
        "- traits: list of 3-5 adjectives\n"
        "- profession: realistic occupation for this region.\n"
        "- family: single, married, or partnered.\n"
        "- political_ideology: short label (e.g., conservative, progressive, socialist, nationalist, centrist).\n"
        "- political_party: plausible choice for the region;\n"
        f"- history_attitude: {history_hint}\n"
        "Return ONLY YAML for a single persona."
    )


def parse_persona(raw: str, name: str, country: str) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        data = {}

    if not isinstance(data, dict):
        data = {}

    # Ensure required keys exist and enforce provided name/country
    data.setdefault("name", name)
    data.setdefault("country", country)
    data.setdefault("gender", "female")
    data.setdefault("age", "30")
    data.setdefault("education", "bachelor")
    data.setdefault("traits", ["urban", "curious", "adaptable"])
    data.setdefault("profession", "office worker")
    data.setdefault("family", "single")
    data.setdefault("political_ideology", "centrist")
    data.setdefault("political_party", "prefer not to say")
    data.setdefault("history_attitude", "No history attitude provided.")

    # Normalize types
    data["age"] = str(data.get("age", "30"))
    traits = data.get("traits", [])
    if not isinstance(traits, list):
        traits = [str(traits)]
    data["traits"] = [str(t) for t in traits if t]
    data["history_attitude"] = str(data.get("history_attitude", "No history attitude provided."))

    return data


def write_initial_file() -> None:
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        yaml.safe_dump([FIXED_PERSONA], fh, sort_keys=False, allow_unicode=True)
    logging.info("Initialized %s with fixed persona %s", OUTPUT_FILE, FIXED_PERSONA["name"])


def append_persona_to_file(persona: Dict[str, Any]) -> None:
    with file_lock:
        with OUTPUT_FILE.open("a", encoding="utf-8") as fh:
            yaml.safe_dump([persona], fh, sort_keys=False, allow_unicode=True)


def generate_persona(client: LLMClient, region: RegionConfig, name: str, progress: ProgressBar) -> Dict[str, Any]:
    prompt = build_prompt(name, region)
    raw = client.generate(prompt)
    persona = parse_persona(raw, name, region.country)
    logging.info("Generated persona %s (%s): %s", persona.get("name"), persona.get("country"), persona)
    append_persona_to_file(persona)
    progress.step()
    return persona


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    client = LLMClient()
    write_initial_file()
    total_to_generate = sum(r.generate for r in REGIONS)
    progress = ProgressBar(total=total_to_generate)

    generated: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_meta = {}
        for region in REGIONS:
            for idx in range(region.generate):
                persona_name = f"{region.prefix}_{idx + 1:03d}"
                future = executor.submit(generate_persona, client, region, persona_name, progress)
                future_to_meta[future] = persona_name

        for future in as_completed(future_to_meta):
            persona = future.result()
            generated.append(persona)
    progress.finish()

    total_written = len(generated) + 1  # include fixed persona
    print(f"\nWrote {total_written} personas to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
