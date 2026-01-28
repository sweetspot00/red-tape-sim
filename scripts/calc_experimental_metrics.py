#!/usr/bin/env python
"""Compute average emotion scores and Jaccard@3 for Experimental rewards (red tape)."""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import yaml

HUMAN_TOP3 = {
    "Germany": ["anger", "frustration", "confusion"],
    "China": ["fear", "joy", "surprise"],
    "HongKong": ["contemp", "disgust", "surprise"],
}

TARGET_COUNTRIES = set(HUMAN_TOP3.keys())
ENGLISH_EXCLUDED_COUNTRIES = {"Germany", "China"}
ALLOWED_MODEL_SUBSTRINGS = ("gpt-5", "gpt-4o", "gemini-3-pro", "deepseek")


def load_experimental_red_tape_titles(path: str) -> Tuple[Dict[str, set], Dict[str, set]]:
    """Return (titles, english_titles) for Experimental rewards (red tape)."""
    with open(path, "r", encoding="utf-8") as f:
        events = yaml.safe_load(f)

    titles_by_country: Dict[str, set] = defaultdict(set)
    english_titles_by_country: Dict[str, set] = defaultdict(set)
    for ev in events:
        if not ev.get("if_red_tape"):
            continue
        country = ev.get("country")
        if country not in TARGET_COUNTRIES:
            continue

        title = ev.get("title", "")
        title_en = ev.get("title_en", "")

        def is_english_title(text: str) -> bool:
            return bool(text) and all(ord(c) < 128 for c in text) and any(c.isalpha() for c in text)

        # Heuristic: only include Experimental rewards events.
        if (
            "Experimental" in title
            or "实验奖金" in title
            or "Versuchsteilnahme" in title
            or "Experimental" in title_en
            or "participation" in title_en
        ):
            if title and not (country in ENGLISH_EXCLUDED_COUNTRIES and is_english_title(title)):
                titles_by_country[country].add(title)
            if title_en and country not in ENGLISH_EXCLUDED_COUNTRIES:
                titles_by_country[country].add(title_en)
            if title_en and country in ENGLISH_EXCLUDED_COUNTRIES:
                english_titles_by_country[country].add(title_en)

    return titles_by_country, english_titles_by_country


def iter_history_results(paths: Iterable[str]):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("model", "unknown")
        mode = data.get("mode", "unknown")
        for result in data.get("results", []):
            yield path, model, mode, result


def normalize_label(label: str) -> str:
    label = label.strip().lower()
    if label == "contemp":
        return "contempt"
    return label


def add_emotions(
    totals: Dict[str, float], counts: Dict[str, int], emotion: Dict
) -> int:
    added = 0
    for k, v in emotion.items():
        if isinstance(v, (int, float)):
            key = normalize_label(str(k))
            totals[key] += float(v)
            counts[key] += 1
            added += 1
    return added


def jaccard_at_3(human: List[str], model: List[str]) -> float:
    h = set(normalize_label(x) for x in human[:3])
    m = set(normalize_label(x) for x in model[:3])
    if not h and not m:
        return 0.0
    return len(h & m) / len(h | m)


def main() -> None:
    titles_by_country, english_titles_by_country = load_experimental_red_tape_titles(
        "data/events.yaml"
    )
    paths = sorted(glob.glob("data/sim_history/*.json"))

    # Group by (model, country, event title)
    totals: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    meta_counts = defaultdict(
        lambda: {"results": 0, "reactions": 0, "numeric_reactions": 0, "error_reactions": 0}
    )
    # Per-file grouping
    file_totals: Dict[Tuple[str, str, str, str], Dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    file_counts: Dict[Tuple[str, str, str, str], Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    file_meta_counts = defaultdict(
        lambda: {"results": 0, "reactions": 0, "numeric_reactions": 0, "error_reactions": 0}
    )
    file_jaccards: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)

    for path, model, mode, result in iter_history_results(paths):
        model_lower = (model or "").lower()
        is_default_agent = (mode or "").lower() == "default_agent"
        if not is_default_agent and not any(sub in model_lower for sub in ALLOWED_MODEL_SUBSTRINGS):
            continue
        if not any(sub in (model or "").lower() for sub in ALLOWED_MODEL_SUBSTRINGS):
            continue

        event = result.get("event", {})
        country = event.get("country")
        title = event.get("title")
        if country not in TARGET_COUNTRIES:
            continue
        if not event.get("if_red_tape"):
            continue
        if title not in titles_by_country.get(country, set()):
            if not (
                is_default_agent
                and title in english_titles_by_country.get(country, set())
            ):
                continue

        model_label = f"default_agent/{model}" if is_default_agent else model
        key = (model_label, country, title)
        file_key = (path, model_label, country, title)
        meta_counts[key]["results"] += 1
        file_meta_counts[file_key]["results"] += 1

        for reaction in result.get("reactions", []):
            emotion = reaction.get("emotion")
            if isinstance(emotion, dict):
                added = add_emotions(totals[key], counts[key], emotion)
                added_file = add_emotions(file_totals[file_key], file_counts[file_key], emotion)
                meta_counts[key]["reactions"] += 1
                file_meta_counts[file_key]["reactions"] += 1
                if added > 0:
                    meta_counts[key]["numeric_reactions"] += 1
                else:
                    meta_counts[key]["error_reactions"] += 1
                if added_file > 0:
                    file_meta_counts[file_key]["numeric_reactions"] += 1
                else:
                    file_meta_counts[file_key]["error_reactions"] += 1
            else:
                meta_counts[key]["reactions"] += 1
                meta_counts[key]["error_reactions"] += 1
                file_meta_counts[file_key]["reactions"] += 1
                file_meta_counts[file_key]["error_reactions"] += 1

    # Print results
    # Per-file results
    for key in sorted(file_totals.keys()):
        path, model, country, title = key
        print(f"\n-- {path} | {model} | {country} | {title} --")
        print(
            "files: {results} results, {reactions} reactions "
            "({numeric} numeric, {errors} error-only)".format(
                results=file_meta_counts[key]["results"],
                reactions=file_meta_counts[key]["reactions"],
                numeric=file_meta_counts[key]["numeric_reactions"],
                errors=file_meta_counts[key]["error_reactions"],
            )
        )
        averages = {
            emo: (file_totals[key][emo] / file_counts[key][emo])
            for emo in file_totals[key]
            if file_counts[key][emo] > 0
        }
        top3 = [e for e, _ in sorted(averages.items(), key=lambda x: (-x[1], x[0]))[:3]]
        jac = jaccard_at_3(HUMAN_TOP3[country], top3)
        print(f"top3: {top3}")
        print(f"jaccard@3: {jac:.4f}")
        file_jaccards[(model, country, title)].append(jac)

    # Aggregate results (jaccard averaged over files)
    for key in sorted(totals.keys()):
        model, country, title = key
        print(f"\n== {model} | {country} | {title} ==")
        print(
            "files: {results} results, {reactions} reactions "
            "({numeric} numeric, {errors} error-only)".format(
                results=meta_counts[key]["results"],
                reactions=meta_counts[key]["reactions"],
                numeric=meta_counts[key]["numeric_reactions"],
                errors=meta_counts[key]["error_reactions"],
            )
        )

        # Average per emotion (combined)
        averages = {
            emo: (totals[key][emo] / counts[key][emo])
            for emo in totals[key]
            if counts[key][emo] > 0
        }
        for emo, avg in sorted(averages.items(), key=lambda x: (-x[1], x[0])):
            print(f"{emo:>12}: {avg:.4f}")

        file_jacs = file_jaccards.get(key, [])
        if file_jacs:
            avg_jac = sum(file_jacs) / len(file_jacs)
            print(f"jaccard@3_avg_files: {avg_jac:.4f}")
        else:
            print("jaccard@3_avg_files: 0.0000")


if __name__ == "__main__":
    main()
