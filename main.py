import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from simulation import (
    EventSimulation,
    LLMClient,
    load_country_prompts_from_yaml,
    load_events_from_yaml,
    load_personas_from_yaml,
)


def demo():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    country_prompts = load_country_prompts_from_yaml("data/country_prompts.yaml")
    personas = load_personas_from_yaml("data/personas.yaml", country_prompts=country_prompts)
    events = load_events_from_yaml("data/events.yaml")
    if not events:
        raise SystemExit("No events found in data/events.yaml")

    llm = LLMClient()  # configure via env vars: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    sim = EventSimulation(personas, llm=llm)
    all_reactions = defaultdict(list)

    for event in events:
        reactions = sim.publish_event(
            event,
            countries={(event.country or "") , ""},
        )
        all_reactions[event.title] = reactions
        stats = EventSimulation.summarize_emotions(reactions)

        print(f"\n=== Event: {event.title} ===")
        for r in reactions:
            print(f"{r.persona.name} ({r.persona.country or 'N/A'}) -> {r.emotion}")
        print("Counts:", stats["counts"])
        print("Distribution:", stats["distribution"])

    plot_stats(all_reactions)


def plot_stats(all_reactions: dict):
    """
    Plot emotion counts per event, broken out by country (stacked bars).
    """
    emotions = [
        "peace",
        "anger",
        "contempt",
        "fear",
        "disgust",
        "joy",
        "sadness",
        "surprise",
        "confusion",
        "frustration",
    ]

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    for title, reactions in all_reactions.items():
        country_set = {r.persona.country or "Unknown" for r in reactions}
        countries = sorted(country_set)

        # Build matrix: emotions x countries
        matrix = {emotion: {country: 0 for country in countries} for emotion in emotions}
        for r in reactions:
            emotion = r.emotion if r.emotion in emotions else "confusion"
            country = r.persona.country or "Unknown"
            matrix[emotion][country] += 1

        # Filter out emotions with zero total counts
        filtered_emotions = [e for e in emotions if sum(matrix[e].values()) > 0]
        if not filtered_emotions:
            logging.info("No emotions to plot for %s", title)
            continue

        plt.figure(figsize=(10, 5))
        colors = plt.cm.tab20.colors
        group_width = 0.8
        width = group_width / max(1, len(countries))
        offsets = [(i - (len(countries) - 1) / 2) * width for i in range(len(countries))]
        x_positions = list(range(len(filtered_emotions)))

        for idx, country in enumerate(countries):
            values = [matrix[emotion][country] for emotion in filtered_emotions]
            bars = [x + offsets[idx] for x in x_positions]
            plt.bar(bars, values, width=width, color=colors[idx % len(colors)], label=country)

        plt.title(f"Emotion counts by country for {title}")
        plt.ylabel("Count")
        plt.xticks(x_positions, filtered_emotions, rotation=30, ha="right")
        plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        outfile = output_dir / f"{title.lower().replace(' ', '_')}_emotions_by_country.png"
        plt.savefig(outfile)
        plt.close()
        logging.info("Saved figure %s", outfile)


if __name__ == "__main__":
    demo()
