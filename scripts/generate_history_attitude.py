"""
Generate history attitudes for personas using the existing LLM client.
Creates a NEW file (data/personas_with_history.yaml); does not modify data/personas.yaml.
Adds `history_attitude` as a list of two plain sentences per persona (no nested dicts).

Environment:
- OPENAI_API_KEY (required for LLMClient)
- Optionally OPENAI_BASE_URL / OPENAI_MODEL
"""

import asyncio
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Import project modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from simulation.llm import LLMClient  # noqa: E402


PERSONA_FILE = Path("data/personas.yaml")
OUTPUT_FILE = Path("data/personas_with_history.yaml")
FIELDS_KEY = "history_attitude"
BATCH_SIZE = 5


def load_personas() -> List[Dict[str, Any]]:
    return yaml.safe_load(PERSONA_FILE.read_text(encoding="utf-8")) or []


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def build_prompt(entry: Dict[str, Any]) -> str:
    name = entry.get("name", "")
    country = entry.get("country", "unknown country")
    traits = entry.get("traits", [])
    traits_text = ", ".join(traits) if traits else "no specific traits listed"
    return (
        "For the persona below, invent TWO past events that are strongly tied to their country/region "
        "and summarize their attitude toward each. Return YAML with key `history_attitude`, "
        "a list of two plain sentences (strings), no nested dicts.\n"
        f"persona_name: {name}\n"
        f"country: {country}\n"
        f"traits: {traits_text}\n"
    )


async def translate_batch(client: LLMClient, entries: List[Dict[str, Any]]) -> List[List[str]]:
    loop = asyncio.get_running_loop()
    tasks = []
    for entry in entries:
        prompt = build_prompt(entry)
        tasks.append(loop.run_in_executor(None, client.generate, prompt))
    results = await asyncio.gather(*tasks)
    outputs: List[List[str]] = []
    for raw in results:
        try:
            data = yaml.safe_load(raw) or {}
            hist = data.get(FIELDS_KEY, [])
            if isinstance(hist, list):
                hist = [str(x) for x in hist]
            else:
                hist = []
        except yaml.YAMLError:
            hist = []
        outputs.append(hist)
    return outputs


async def main():
    personas = load_personas()
    if not personas:
        print("No personas found.")
        return

    client = LLMClient()
    updated = copy.deepcopy(personas)

    for batch in chunked(list(range(len(personas))), BATCH_SIZE):
        entries = [personas[i] for i in batch]
        histories = await translate_batch(client, entries)
        for idx, hist in zip(batch, histories):
            if hist:
                updated[idx][FIELDS_KEY] = hist

    OUTPUT_FILE.write_text(
        yaml.safe_dump(updated, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Wrote {len(personas)} personas to {OUTPUT_FILE} with {FIELDS_KEY}.")


if __name__ == "__main__":
    asyncio.run(main())
