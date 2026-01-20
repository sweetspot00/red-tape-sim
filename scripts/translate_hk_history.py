"""
Translate `history_attitude` for Hongkong personas in data/personas_new.yaml to English.
Writes back after each translation (incremental persistence). Requires OPENAI_API_KEY (and optional OPENAI_BASE_URL / OPENAI_MODEL).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Make project modules available
sys.path.append(str(Path(__file__).resolve().parent.parent))
from simulation.llm import LLMClient  # noqa: E402


PERSONA_FILE = Path("data/personas_new.yaml")


def load_personas() -> List[Dict[str, Any]]:
    return yaml.safe_load(PERSONA_FILE.read_text(encoding="utf-8")) or []


def translate_text(client: LLMClient, text: str) -> str:
    prompt = (
        "Translate the following persona history attitude to English. "
        "Keep the meaning, return only the translated sentences, no YAML, no quotes.\n"
        f"Original: {text}"
    )
    return (client.generate(prompt, temperature=1) or "").strip()


def main() -> None:
    personas = load_personas()
    if not personas:
        print("No personas found.")
        return

    client = LLMClient()
    hk_indices = [i for i, p in enumerate(personas) if p.get("country") == "Hongkong"]
    if not hk_indices:
        print("No Hongkong personas found.")
        return

    for idx, persona_index in enumerate(hk_indices, start=1):
        entry = personas[persona_index]
        original = str(entry.get("history_attitude", "")).strip()
        translated = translate_text(client, original)
        entry["history_attitude"] = translated
        # Persist after each translation to avoid losing progress.
        PERSONA_FILE.write_text(
            yaml.safe_dump(personas, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"[{idx}/{len(hk_indices)}] Translated {entry.get('name')} and wrote to file.")

    print(f"Finished updating {len(hk_indices)} Hongkong personas in {PERSONA_FILE}.")


if __name__ == "__main__":
    main()
