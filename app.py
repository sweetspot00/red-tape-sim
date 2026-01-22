import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Set, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import streamlit as st

from simulation import (
    EventSimulation,
    LLMClient,
    Event,
    Persona,
    load_country_prompts_from_yaml,
    load_events_from_yaml,
    load_personas_from_yaml,
)
from simulation.model import Reaction

HISTORY_DIR = Path("data/sim_history")


@st.cache_data
def load_data():
    country_prompts = load_country_prompts_from_yaml("data/country_prompts.yaml")
    personas = load_personas_from_yaml("data/personas.yaml", country_prompts=country_prompts)
    events = load_events_from_yaml("data/events.yaml")
    return personas, events


def normalize_country(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _emotion_to_str(emotion) -> str:
    if isinstance(emotion, dict):
        try:
            return json.dumps(emotion, ensure_ascii=False)
        except Exception:
            return str(emotion)
    return "" if emotion is None else str(emotion)


def target_countries_for_event(event: Event) -> Set[str]:
    return {normalize_country(event.country), ""}


def run_simulations(
    events_to_run: List[Event],
    countries: Set[str],
    include_optional: Set[str],
    model: str,
    on_progress: Optional[Callable] = None,
):
    """
    Run LLM simulations sequentially for the provided events.
    """
    results = []
    for event in events_to_run:
        reactions, stats, resolved_event = run_simulation_async(
            event, countries, include_optional, model, on_progress=on_progress
        )
        if reactions is not None:
            results.append((resolved_event, reactions, stats))
    return results


def run_simulation_async(
    event_or_idx: Union[int, Event],
    countries: Set[str],
    include_optional: Set[str],
    model: str,
    on_progress: Optional[Callable] = None,
):
    personas, events = load_data()
    if isinstance(event_or_idx, int):
        if event_or_idx >= len(events):
            st.error("Invalid event selection.")
            return None, None, None
        event = events[event_or_idx]
    else:
        event = event_or_idx
    try:
        async def runner():
            llm = LLMClient(model=model)
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


def run_test_simulation(
    events_to_run: List[Event],
    countries: Set[str],
    on_progress: Optional[Callable] = None,
):
    personas, _ = load_data()
    if not events_to_run:
        st.error("No events available.")
        return []
    small_personas = personas[:2]
    rng = np.random.default_rng(0)
    results = []
    for event in events_to_run:
        reactions = []
        for persona in small_personas:
            if countries and normalize_country(persona.country) not in countries:
                continue
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
            ]
            if persona.country.lower() == "germany":
                emotions.append("frustration")
            probs = rng.dirichlet(np.ones(len(emotions)))
            emo_dict = {emo: float(p) for emo, p in zip(emotions, probs)}
            reaction = Reaction(persona=persona, emotion=emo_dict, comment="mock")
            reactions.append(reaction)
            if on_progress:
                on_progress(reaction)
        stats = EventSimulation.summarize_emotions(reactions)
        results.append((event, reactions, stats))
    return results


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
                "family": p.family,
                "political_ideology": p.political_ideology,
                "political_party": p.political_party,
                "history_attitude": " | ".join(p.history_attitude) if p.history_attitude else "",
                "power_distance": p.power_distance,
                "individualism": p.individualism,
                "masculinity": p.masculinity,
                "uncertainty_avoidance": p.uncertainty_avoidance,
                "long_term_orientation": p.long_term_orientation,
                "indulgence": p.indulgence,
            }
        )
    st.dataframe(pd.DataFrame(profile_rows))


def _format_top_emotions(emotion_dict: dict, top_n: int = 3) -> str:
    if not isinstance(emotion_dict, dict):
        return ""
    items = []
    for k, v in emotion_dict.items():
        try:
            items.append((k, float(v)))
        except (TypeError, ValueError):
            continue
    items = sorted(items, key=lambda kv: kv[1], reverse=True)[:top_n]
    return ", ".join(f"{k}: {float(v):.2f}" for k, v in items)


def render_reactions(reactions):
    rows = []
    for r in reactions:
        top3 = _format_top_emotions(r.emotion, top_n=3)
        raw = (
            json.dumps(r.emotion, ensure_ascii=False)
            if isinstance(r.emotion, dict)
            else str(r.emotion)
        )
        rows.append(
            {
                "persona": r.persona.name,
                "country": r.persona.country or "Default",
                "emotions_raw": raw,
                "top_3_emotions": top3,
                "comment": r.comment,
            }
        )
    st.dataframe(pd.DataFrame(rows))


def _reaction_to_dict(reaction: Reaction) -> dict:
    return {
        "persona": reaction.persona.name,
        "country": reaction.persona.country or "Default",
        "emotion": reaction.emotion,
        "comment": reaction.comment,
    }


def _event_to_dict(event: Event) -> dict:
    return {
        "title": event.title,
        "description": event.description,
        "country": event.country,
        "if_red_tape": event.if_red_tape,
        "metadata": event.metadata,
    }


def _dict_to_event(data: dict) -> Event:
    return Event(
        title=data.get("title", ""),
        description=data.get("description", ""),
        country=data.get("country"),
        if_red_tape=bool(data.get("if_red_tape", False)),
        metadata=data.get("metadata", {}) or {},
    )


def _dicts_to_reactions(items: List[dict]) -> List[Reaction]:
    reactions: List[Reaction] = []
    for item in items:
        persona = Persona(name=item.get("persona", ""), country=item.get("country", ""))
        reactions.append(
            Reaction(
                persona=persona,
                emotion=item.get("emotion"),
                comment=item.get("comment", ""),
            )
        )
    return reactions


EMOTION_BASE_ORDER = [
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


def _emotion_order(keys) -> List[str]:
    order: List[str] = []
    seen = set()
    for emo in EMOTION_BASE_ORDER:
        if emo in keys and emo not in seen:
            order.append(emo)
            seen.add(emo)
    for emo in sorted(keys):
        if emo not in seen:
            order.append(emo)
    return order


def _collect_emotion_values(reactions: List[Reaction]) -> dict[str, List[float]]:
    values: dict[str, List[float]] = {}
    for reaction in reactions:
        if not isinstance(reaction.emotion, dict):
            continue
        for emo, val in reaction.emotion.items():
            try:
                score = float(val)
            except (TypeError, ValueError):
                continue
            values.setdefault(emo, []).append(score)
    return {k: v for k, v in values.items() if v}


def _plot_emotion_bar(emotions: dict, title: str):
    order = _emotion_order(emotions.keys())
    values = [emotions.get(e, 0) for e in order]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(order, values, color="#6a8aff", edgecolor="#1f3b73", alpha=0.9)
    ax.set_ylim(0, max(values + [0.01]) * 1.1)
    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    st.pyplot(fig)


def _plot_emotion_radar(emotions: dict, title: str, color: str = "#6a8aff"):
    order = _emotion_order(emotions.keys())
    if not order:
        return
    values = [emotions.get(e, 0) for e in order]
    theta = np.linspace(0, 2 * np.pi, len(order), endpoint=False)
    angles = np.concatenate([theta, theta[:1]])
    vals_closed = np.concatenate([values, values[:1]])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
    ax.plot(angles, vals_closed, linewidth=2, color=color)
    ax.fill(angles, vals_closed, alpha=0.2, color=color)
    ax.set_xticks(theta)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    st.pyplot(fig)


def render_emotion_dot_whisker(reactions: List[Reaction], label: str):
    values_by_emotion = _collect_emotion_values(reactions)
    if not values_by_emotion:
        st.info(f"No emotion data for {label}.")
        return
    order = _emotion_order(values_by_emotion.keys())
    ordered_values = [values_by_emotion[e] for e in order]
    fig_height = max(3.5, 0.45 * len(order) + 1.5)

    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.boxplot(
        ordered_values,
        vert=False,
        labels=order,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops={"facecolor": "#dfe7ff", "edgecolor": "#1f3b73"},
        medianprops={"color": "#1f3b73"},
        whiskerprops={"color": "#1f3b73"},
        capprops={"color": "#1f3b73"},
        meanprops={"color": "#f58518", "linewidth": 2},
    )

    rng = np.random.default_rng(0)
    for idx, emo in enumerate(order, start=1):
        vals = values_by_emotion[emo]
        jitter = rng.normal(loc=0, scale=0.04, size=len(vals))
        y = np.full(len(vals), idx) + jitter
        ax.scatter(
            vals,
            y,
            color="#4c78a8",
            alpha=0.6,
            s=25,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel("Emotion score")
    ax.set_ylabel("Emotion")
    ax.set_title(f"Emotion dot-and-whisker — {label}", fontsize=12, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    st.pyplot(fig)


def render_emotion_dot_whisker_comparison(
    base_reactions: List[Reaction],
    red_reactions: List[Reaction],
    label: str = "Red vs Non red-tape",
):
    base_values = _collect_emotion_values(base_reactions)
    red_values = _collect_emotion_values(red_reactions)
    keys = set(base_values) | set(red_values)
    if not keys:
        st.info("No emotion data to compare.")
        return
    order = _emotion_order(keys)
    fig_height = max(3.8, 0.5 * len(order) + 1.6)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    boxprops_base = {"facecolor": "#dfe7ff", "edgecolor": "#1f3b73"}
    boxprops_red = {"facecolor": "#ffe0cb", "edgecolor": "#9c3d00"}

    base_data, base_pos = [], []
    red_data, red_pos = [], []
    jitter_scale = 0.04
    rng = np.random.default_rng(0)

    for idx, emo in enumerate(order, start=1):
        vals_base = base_values.get(emo)
        if vals_base:
            base_data.append(vals_base)
            base_pos.append(idx + 0.15)
        vals_red = red_values.get(emo)
        if vals_red:
            red_data.append(vals_red)
            red_pos.append(idx - 0.15)

    if base_data:
        ax.boxplot(
            base_data,
            vert=False,
            positions=base_pos,
            showmeans=True,
            meanline=True,
            patch_artist=True,
            boxprops=boxprops_base,
            medianprops={"color": "#1f3b73"},
            whiskerprops={"color": "#1f3b73"},
            capprops={"color": "#1f3b73"},
            meanprops={"color": "#1f3b73", "linewidth": 2},
        )
    if red_data:
        ax.boxplot(
            red_data,
            vert=False,
            positions=red_pos,
            showmeans=True,
            meanline=True,
            patch_artist=True,
            boxprops=boxprops_red,
            medianprops={"color": "#9c3d00"},
            whiskerprops={"color": "#9c3d00"},
            capprops={"color": "#9c3d00"},
            meanprops={"color": "#9c3d00", "linewidth": 2},
        )

    for idx, emo in enumerate(order, start=1):
        vals_base = base_values.get(emo, [])
        if vals_base:
            jitter = rng.normal(loc=0, scale=jitter_scale, size=len(vals_base))
            y = np.full(len(vals_base), idx + 0.15) + jitter
            ax.scatter(
                vals_base,
                y,
                color="#4c78a8",
                alpha=0.6,
                s=22,
                edgecolors="white",
                linewidths=0.5,
            )
        vals_red = red_values.get(emo, [])
        if vals_red:
            jitter = rng.normal(loc=0, scale=jitter_scale, size=len(vals_red))
            y = np.full(len(vals_red), idx - 0.15) + jitter
            ax.scatter(
                vals_red,
                y,
                color="#f58518",
                alpha=0.6,
                s=22,
                edgecolors="white",
                linewidths=0.5,
            )

    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order)
    ax.set_xlabel("Emotion score")
    ax.set_ylabel("Emotion")
    ax.set_title(f"Emotion dot-and-whisker comparison — {label}", fontsize=12, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    legend_handles = [
        Line2D([0], [0], color="#1f3b73", lw=2, label="Non red-tape"),
        Line2D([0], [0], color="#9c3d00", lw=2, label="Red-tape"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    st.pyplot(fig)


def render_emotion_means(stats, label: str):
    if not stats or not stats.get("means"):
        st.info(f"No emotion data for {label}.")
        return
    means = stats["means"]
    _plot_emotion_bar(means, f"Mean emotion probabilities — {label}")
    _plot_emotion_radar(means, f"Mean emotion radar — {label}", color="#6a8aff")


def render_emotion_difference(base_stats, red_stats):
    if not base_stats or not red_stats:
        st.info("Need both baseline and red-tape stats to show differences.")
        return
    base = base_stats.get("means", {})
    red = red_stats.get("means", {})
    keys = set(base) | set(red)
    if not keys:
        st.info("No overlapping emotions to compare.")
        return
    order = _emotion_order(keys)
    diffs = {emo: red.get(emo, 0) - base.get(emo, 0) for emo in order}
    abs_diffs = {emo: abs(val) for emo, val in diffs.items()}

    # Signed bar to show direction of change
    fig, ax = plt.subplots(figsize=(8, 3.5))
    vals = [diffs[e] for e in order]
    colors = ["#d6604d" if v < 0 else "#4daf7c" for v in vals]
    ax.bar(order, vals, color=colors, edgecolor="#333333", alpha=0.9)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_title("Emotion mean differences (Red - Non-red)", fontsize=12, pad=10)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    st.pyplot(fig)

    # Radar with magnitude of variance (absolute differences)
    _plot_emotion_radar(abs_diffs, "Emotion variance (|Red - Non-red|)", color="#ff8c42")


def render_emotion_dual_radar(base_stats, red_stats):
    if not base_stats or not red_stats:
        return
    base = base_stats.get("means", {}) or {}
    red = red_stats.get("means", {}) or {}
    keys = set(base) | set(red)
    if not keys:
        return
    order = _emotion_order(keys)
    base_vals = [base.get(e, 0) for e in order]
    red_vals = [red.get(e, 0) for e in order]
    theta = np.linspace(0, 2 * np.pi, len(order), endpoint=False)
    angles = np.concatenate([theta, theta[:1]])
    base_closed = np.concatenate([base_vals, base_vals[:1]])
    red_closed = np.concatenate([red_vals, red_vals[:1]])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5.5, 5.5))
    ax.plot(angles, base_closed, linewidth=2, color="#4c78a8", label="Non red-tape")
    ax.fill(angles, base_closed, alpha=0.15, color="#4c78a8")
    ax.plot(angles, red_closed, linewidth=2, color="#f58518", label="Red-tape")
    ax.fill(angles, red_closed, alpha=0.12, color="#f58518")
    ax.set_xticks(theta)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Emotion radar: Non red-tape vs Red-tape", fontsize=13, pad=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    st.pyplot(fig)


def render_result_pair(results: List[tuple], mock: bool = False):
    """
    Render two events side by side with reactions and emotion stats.
    """
    base_result = results[0] if len(results) >= 1 else None
    red_result = results[1] if len(results) >= 2 else None

    col_nonred, col_red = st.columns(2, gap="large")

    if base_result:
        event, reactions, stats = base_result
        with col_nonred:
            st.markdown("### Non red-tape" + (" [MOCK]" if mock else ""))
            st.write(f"**{event.title}** ({event.country or 'N/A'})")
            st.caption(event.description)
            st.markdown("**Reactions**")
            render_reactions(reactions)
            st.markdown("**Emotion Statistics**")
            render_emotion_means(stats, event.title)
            render_emotion_dot_whisker(reactions, event.title)
    else:
        st.warning("Non red-tape event did not run.")

    if red_result:
        event, reactions, stats = red_result
        with col_red:
            st.markdown("### Red-tape" + (" [MOCK]" if mock else ""))
            st.write(f"**{event.title}** ({event.country or 'N/A'})")
            st.caption(event.description)
            st.markdown("**Reactions**")
            render_reactions(reactions)
            st.markdown("**Emotion Statistics**")
            render_emotion_means(stats, event.title)
            render_emotion_dot_whisker(reactions, event.title)
    else:
        st.warning("Red-tape event did not run.")

    if base_result and red_result:
        st.subheader("Emotion Dot-and-Whisker Comparison" + (" [MOCK]" if mock else ""))
        render_emotion_dot_whisker_comparison(base_result[1], red_result[1])
        st.subheader("Emotion Differences (Red - Non-red)" + (" [MOCK]" if mock else ""))
        render_emotion_difference(base_result[2], red_result[2])
        st.subheader("Emotion Radar Comparison" + (" [MOCK]" if mock else ""))
        render_emotion_dual_radar(base_result[2], red_result[2])


def save_simulation_run(results: List[tuple], mode: str, model: Optional[str] = None) -> Path:
    """
    Persist the current simulation results to disk for later viewing.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "id": run_id,
        "mode": mode,
        "saved_at": run_id,
        "model": model,
        "results": [],
    }
    for event, reactions, stats in results:
        payload["results"].append(
            {
                "event": _event_to_dict(event),
                "reactions": [_reaction_to_dict(r) for r in reactions],
                "stats": stats,
            }
        )
    path = HISTORY_DIR / f"{run_id}_{mode}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_simulation_run(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.error(f"Failed to load history {path.name}: {exc}")
        return None


def list_history_files() -> List[Path]:
    if not HISTORY_DIR.exists():
        return []
    return sorted(HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def history_label(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        saved_at = payload.get("saved_at") or path.stem
        first_event = ""
        results = payload.get("results") or []
        if results and isinstance(results[0], dict):
            evt = results[0].get("event", {})
            if isinstance(evt, dict):
                first_event = evt.get("title") or ""
        model = payload.get("model")
        label_parts = [p for p in (first_event, saved_at) if p]
        label = " — ".join(label_parts) if label_parts else path.name
        if model:
            label = f"{label} (model: {model})"
        return label
    except Exception:
        pass
    return path.name


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    st.title("Persona Reaction Dashboard")
    personas, events = load_data()

    tabs = st.tabs(["Run Simulation", "History"])

    with tabs[0]:
        st.sidebar.header("Simulation Controls")
        model_choices = [
            "azure/gpt-5",
            "azure/gpt-4o",
            "azure/gpt-4o-mini",
            "gemini/ggap/gemini-2.5-pro",
            "gemini/ggap/gemini-2.5-flash-lite",
            "gemini/ggap/gemini-2.5-flash",
            "gemini/ggap/gemini-3-pro",
        ]
        selected_model = st.sidebar.selectbox("LLM model", model_choices, index=1)
        non_red_events = [e for e in events if not e.if_red_tape]
        red_tape_events = [e for e in events if e.if_red_tape]

        if not non_red_events:
            st.error("No non-red-tape events available.")
            return

        base_idx = st.sidebar.selectbox(
            "Non red-tape event",
            range(len(non_red_events)),
            format_func=lambda i: non_red_events[i].title,
        )
        base_event = non_red_events[base_idx]

        # Prefer red-tape events in the same country as the base event
        country_matched_red = [
            e for e in red_tape_events if normalize_country(e.country) == normalize_country(base_event.country)
        ]
        available_red = country_matched_red or red_tape_events
        if not available_red:
            st.sidebar.warning("No red-tape events available. Add one to run paired simulation.")
            red_event = None
        else:
            red_idx = st.sidebar.selectbox(
                "Red-tape event",
                range(len(available_red)),
                format_func=lambda i: available_red[i].title,
            )
            red_event = available_red[red_idx]

        selected_events = [e for e in (base_event, red_event) if e]
        target_countries: Set[str] = set()
        for event in selected_events:
            target_countries |= target_countries_for_event(event)

        st.caption(f"Using model: {selected_model}")

        st.subheader("Personas (filtered for selected countries)")
        filtered_personas = [p for p in personas if normalize_country(p.country) in target_countries]
        render_profiles(filtered_personas)

        st.subheader("Run Simulation")
        col1, col2 = st.columns(2)
        run_clicked = col1.button("Run (LLM)")
        test_clicked = col2.button("Test (Mock)")

        if run_clicked:
            if len(selected_events) < 2:
                st.error("Please select both a non red-tape and a red-tape event.")
            else:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                rows: List[dict] = []

                def on_progress(reaction):
                    rows.append(
                        {
                            "persona": reaction.persona.name,
                            "country": reaction.persona.country or "Default",
                            "emotion": _emotion_to_str(reaction.emotion),
                        }
                    )
                    df = pd.DataFrame(rows)
                    progress_placeholder.dataframe(df)
                    status_placeholder.info(f"Received {len(rows)} reactions across events...")

            with st.spinner("Running simulation with LLM..."):
                results = run_simulations(
                    selected_events,
                    target_countries,
                    set(),
                    selected_model,
                    on_progress=on_progress,
                )
            if results:
                save_path = save_simulation_run(results, mode="llm", model=selected_model)
                st.success(f"Completed LLM simulation for selected pair. Saved to {save_path.name}")
                st.caption(f"Model used: {selected_model}")
                render_result_pair(results, mock=False)

        if test_clicked:
            if len(selected_events) < 2:
                st.error("Please select both a non red-tape and a red-tape event.")
            else:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                rows: List[dict] = []

                def on_progress(reaction):
                    rows.append(
                        {
                            "persona": reaction.persona.name,
                            "country": reaction.persona.country or "Default",
                            "emotion": _emotion_to_str(reaction.emotion),
                            "comment": reaction.comment,
                        }
                    )
                    df = pd.DataFrame(rows)
                    progress_placeholder.dataframe(df)
                    status_placeholder.info(f"Received {len(rows)} reactions across events (test)...")

            with st.spinner("Running mock simulation..."):
                results = run_test_simulation(selected_events, target_countries, on_progress=on_progress)
            if results:
                save_path = save_simulation_run(results, mode="mock", model="mock")
                st.success(f"Completed mock simulation for selected pair. Saved to {save_path.name}")
                st.caption("Model used: mock")
                render_result_pair(results, mock=True)

        if not (run_clicked or test_clicked):
            st.info("Configure options and click Run (LLM) or Test (Mock) to simulate.")

    with tabs[1]:
        st.subheader("Simulation History")
        files = list_history_files()
        if not files:
            st.info("No saved simulations yet. Run one to populate history.")
        else:
            labels = [history_label(f) for f in files]
            choice = st.selectbox("Saved runs", range(len(files)), format_func=lambda i: labels[i])
            selected_file = files[choice]
            loaded = load_simulation_run(selected_file)
            if loaded:
                st.caption(f"Loaded {selected_file.name}")
                model_used = loaded.get("model")
                if model_used:
                    st.caption(f"Model used: {model_used}")
                result_entries = []
                for entry in loaded.get("results", []):
                    event_dict = entry.get("event", {})
                    event = _dict_to_event(event_dict)
                    reactions = _dicts_to_reactions(entry.get("reactions", []))
                    stats = entry.get("stats", {})
                    result_entries.append((event, reactions, stats))
                if result_entries:
                    render_result_pair(result_entries, mock=(loaded.get("mode") == "mock"))
                else:
                    st.warning("Selected file has no results to display.")


if __name__ == "__main__":
    main()
