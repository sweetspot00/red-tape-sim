import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Set

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from simulation import (
    EventSimulation,
    LLMClient,
    load_country_prompts_from_yaml,
    load_events_from_yaml,
    load_personas_from_yaml,
)


@st.cache_data
def load_data():
    country_prompts = load_country_prompts_from_yaml("data/country_prompts.yaml")
    personas = load_personas_from_yaml("data/personas.yaml", country_prompts=country_prompts)
    events = load_events_from_yaml("data/events.yaml")
    return personas, events


def normalize_country(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def run_simulation_async(
    event_idx: int,
    countries: Set[str],
    include_optional: Set[str],
    on_progress: Optional[Callable] = None,
):
    personas, events = load_data()
    if event_idx >= len(events):
        st.error("Invalid event selection.")
        return None, None, None

    event = events[event_idx]
    try:
        async def runner():
            llm = LLMClient()
            sim = EventSimulation(personas, llm=llm)
            reactions = await sim.publish_event_async(
                event,
                countries=countries,
                include_optional_persona_fields=include_optional,
                on_result=on_progress,
            )
            stats = EventSimulation.summarize_emotions(reactions)
            return reactions, stats, event

        return asyncio.run(runner())
    except Exception as exc:  # surface errors instead of hanging silently
        st.error(f"Simulation failed: {exc}")
        return None, None, None


def run_test_simulation(on_progress: Optional[Callable] = None):
    personas, events = load_data()
    if not events:
        st.error("No events available.")
        return None, None, None
    event = events[0]
    small_personas = personas[:2]
    try:
        async def runner():
            llm = LLMClient()
            sim = EventSimulation(small_personas, llm=llm)
            reactions = await sim.publish_event_async(
                event,
                countries={p.country for p in small_personas} | {""},
                on_result=on_progress,
            )
            stats = EventSimulation.summarize_emotions(reactions)
            return reactions, stats, event

        return asyncio.run(runner())
    except Exception as exc:
        st.error(f"Test simulation failed: {exc}")
        return None, None, None


def render_profiles(personas):
    profile_rows = []
    for p in personas:
        profile_rows.append(
            {
                "name": p.name,
                "country": p.country or "Default",
                "gender": p.gender,
                "age": p.age,
                "education": p.education,
                "traits": ", ".join(sorted(p.traits)) if p.traits else "",
                "profession": p.profession,
                "background": p.background,
                "family": p.family,
                "political_ideology": p.political_ideology,
                "political_party": p.political_party,
                "history_attitude": " | ".join(p.history_attitude) if p.history_attitude else "",
            }
        )
    st.dataframe(pd.DataFrame(profile_rows))


def render_reactions(reactions):
    rows = []
    for r in reactions:
        rows.append(
            {
                "persona": r.persona.name,
                "country": r.persona.country or "Default",
                "emotion": r.emotion,
                "comment": r.comment,
            }
        )
    st.dataframe(pd.DataFrame(rows))


def render_stats(stats):
    if not stats or not stats["counts"]:
        st.info("No reactions to summarize.")
        return
    counts = stats["counts"]
    df = pd.DataFrame({"emotion": list(counts.keys()), "count": list(counts.values())})
    df = df.sort_values("count", ascending=False)
    st.bar_chart(df.set_index("emotion"))


EMOTION_ORDER = [
    "joy",
    "anger",
    "fear",
    "contempt",
    "disgust",
    "sadness",
    "surprise",
    "peace",
    "confusion",
]


def render_emotion_circle(reactions):
    if not reactions:
        st.info("No reactions to visualize.")
        return

    # Aggregate counts per country per emotion
    counts = defaultdict(lambda: defaultdict(int))
    for r in reactions:
        country = r.persona.country or "Default"
        counts[country][r.emotion] += 1

    theta = np.linspace(0, 2 * np.pi, len(EMOTION_ORDER), endpoint=False)

    for country, emo_counts in counts.items():
        values = [emo_counts.get(e, 0) for e in EMOTION_ORDER]
        # Close the loop for plotting
        angles = np.concatenate([theta, theta[:1]])
        vals_closed = np.concatenate([values, values[:1]])

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(angles, vals_closed, linewidth=2, color="#333333")
        ax.fill(angles, vals_closed, alpha=0.1, color="#333333")
        ax.set_xticks(theta)
        ax.set_xticklabels(EMOTION_ORDER)
        ax.set_yticklabels([])  # declutter radial labels
        ax.set_title(f"Emotions for {country}", fontsize=14, pad=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    st.title("Persona Reaction Dashboard")
    personas, events = load_data()

    st.sidebar.header("Simulation Controls")
    event_names = [e.title for e in events]
    event_idx = st.sidebar.selectbox("Event", range(len(event_names)), format_func=lambda i: event_names[i])

    # Determine target countries for the selected event (case-insensitive)
    target_countries = {normalize_country(events[event_idx].country), ""}

    st.subheader("Personas (filtered for event country)")
    filtered_personas = [p for p in personas if normalize_country(p.country) in target_countries]
    render_profiles(filtered_personas)

    st.subheader("Run Simulation")
    col1, col2 = st.columns(2)
    run_clicked = col1.button("Run (LLM)")
    test_clicked = col2.button("Test (Mock)")

    if run_clicked:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        rows: List[dict] = []

        def on_progress(reaction):
            rows.append(
                {
                    "persona": reaction.persona.name,
                    "country": reaction.persona.country or "Default",
                    "emotion": reaction.emotion,
                    "comment": reaction.comment,
                }
            )
            df = pd.DataFrame(rows)
            progress_placeholder.dataframe(df)
            status_placeholder.info(f"Received {len(rows)} reactions...")

        with st.spinner("Running simulation with LLM..."):
            reactions, stats, event = run_simulation_async(
                event_idx, target_countries, set(), on_progress=on_progress
            )
        if reactions is not None:
            st.success("Completed LLM simulation.")
            st.write(f"Event: **{event.title}** ({event.country or 'N/A'})")
            st.write(event.description)
            st.subheader("Reactions")
            render_reactions(reactions)
            st.subheader("Emotion Counts")
            render_stats(stats)
            st.subheader("Emotion Circle (per country)")
            render_emotion_circle(reactions)

    if test_clicked:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        rows: List[dict] = []

        def on_progress(reaction):
            rows.append(
                {
                    "persona": reaction.persona.name,
                    "country": reaction.persona.country or "Default",
                    "emotion": reaction.emotion,
                    "comment": reaction.comment,
                }
            )
            df = pd.DataFrame(rows)
            progress_placeholder.dataframe(df)
            status_placeholder.info(f"Received {len(rows)} reactions (test)...")

        with st.spinner("Running mock simulation..."):
            reactions, stats, event = run_test_simulation(on_progress=on_progress)
        if reactions is not None:
            st.success("Completed mock simulation.")
            st.write(f"Event: **{event.title}** ({event.country or 'N/A'}) [MOCK]")
            st.write(event.description)
            st.subheader("Reactions")
            render_reactions(reactions)
            st.subheader("Emotion Counts")
            render_stats(stats)
            st.subheader("Emotion Circle (per country)")
            render_emotion_circle(reactions)

    if not (run_clicked or test_clicked):
        st.info("Configure options and click Run (LLM) or Test (Mock) to simulate.")


if __name__ == "__main__":
    main()
