import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Set

import pandas as pd
import streamlit as st

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


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    st.title("Persona Reaction Dashboard")
    personas, events = load_data()

    st.sidebar.header("Simulation Controls")
    event_names = [e.title for e in events]
    event_idx = st.sidebar.selectbox("Event", range(len(event_names)), format_func=lambda i: event_names[i])

    optional_fields = st.sidebar.multiselect(
        "Optional persona fields to include in prompt",
        ["profession", "background", "family", "political_ideology", "political_party", "traits"],
    )

    # Determine target countries for the selected event
    target_countries = {events[event_idx].country or "", ""}

    st.subheader("Personas (filtered for event country)")
    filtered_personas = [p for p in personas if (p.country or "") in target_countries]
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
                event_idx, target_countries, set(optional_fields), on_progress=on_progress
            )
        if reactions is not None:
            st.success("Completed LLM simulation.")
            st.write(f"Event: **{event.title}** ({event.country or 'N/A'})")
            st.write(event.description)
            st.subheader("Reactions")
            render_reactions(reactions)
            st.subheader("Emotion Counts")
            render_stats(stats)

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

    if not (run_clicked or test_clicked):
        st.info("Configure options and click Run (LLM) or Test (Mock) to simulate.")


if __name__ == "__main__":
    main()
