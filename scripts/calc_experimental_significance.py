#!/usr/bin/env python
"""Significance test between non-red and red tape Experimental rewards events.

Uses paired per-file mean emotion differences with a permutation sign-flip test.
"""
from __future__ import annotations

import glob
import json
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import yaml

ALLOWED_MODEL_SUBSTRINGS = ("gpt-5", "gpt-4o", "gemini-3-pro", "deepseek")
ENGLISH_EXCLUDED_COUNTRIES = {"Germany", "China"}
RNG_SEED = 42
PERMUTATION_ITERATIONS = 2000
SIGRATE_Q = 0.95


def is_english_title(text: str) -> bool:
    return bool(text) and all(ord(c) < 128 for c in text) and any(c.isalpha() for c in text)


def load_experimental_titles(path: str) -> Tuple[Dict[str, set], Dict[str, set]]:
    """Return (non_red_titles, red_titles) for Experimental rewards."""
    with open(path, "r", encoding="utf-8") as f:
        events = yaml.safe_load(f)

    non_red: Dict[str, set] = defaultdict(set)
    red: Dict[str, set] = defaultdict(set)
    english_titles: Dict[str, set] = defaultdict(set)

    for ev in events:
        country = ev.get("country")
        if not country:
            continue
        title = ev.get("title", "")
        title_en = ev.get("title_en", "")
        is_red = bool(ev.get("if_red_tape"))

        if not (
            "Experimental" in title
            or "实验奖金" in title
            or "Versuchsteilnahme" in title
            or "Experimental" in title_en
            or "participation" in title_en
        ):
            continue

        if country in ENGLISH_EXCLUDED_COUNTRIES:
            if title and not is_english_title(title):
                (red if is_red else non_red)[country].add(title)
            if title_en:
                english_titles[country].add(title_en)
        else:
            if title:
                (red if is_red else non_red)[country].add(title)
            if title_en:
                (red if is_red else non_red)[country].add(title_en)

    return non_red, red, english_titles


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


def compute_mean_emotions(reactions: List[dict]) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for rx in reactions:
        emotion = rx.get("emotion")
        if not isinstance(emotion, dict):
            continue
        for k, v in emotion.items():
            if isinstance(v, (int, float)):
                key = normalize_label(str(k))
                totals[key] += float(v)
                counts[key] += 1
    return {k: totals[k] / counts[k] for k in totals if counts[k] > 0}


def sign_flip_pvalue(diffs: List[float], rng: random.Random, iters: int) -> float:
    if not diffs:
        return 1.0
    obs = abs(sum(diffs) / len(diffs))
    if obs == 0:
        return 1.0
    greater = 0
    for _ in range(iters):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if abs(s / len(diffs)) >= obs:
            greater += 1
    return (greater + 1) / (iters + 1)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    if q <= 0:
        return values_sorted[0]
    if q >= 1:
        return values_sorted[-1]
    idx = (len(values_sorted) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(values_sorted) - 1)
    frac = idx - lo
    return values_sorted[lo] * (1 - frac) + values_sorted[hi] * frac


def main() -> None:
    non_red_titles, red_titles, english_titles = load_experimental_titles("data/events.yaml")
    paths = sorted(glob.glob("data/sim_history/*.json"))

    # Collect per-file means
    # file_key: (path, model, country)
    per_file_nonred: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    per_file_red: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    for path, model, mode, result in iter_history_results(paths):
        mode_lower = (mode or "").lower()
        model_lower = (model or "").lower()
        is_default_agent = mode_lower == "default_agent"
        if not is_default_agent and not any(sub in model_lower for sub in ALLOWED_MODEL_SUBSTRINGS):
            continue
        if mode_lower not in {"llm", "default_agent"}:
            continue

        event = result.get("event", {})
        country = event.get("country")
        title = event.get("title")
        if not country or not title:
            continue

        model_label = f"default_agent/{model}" if is_default_agent else model
        key = (path, model_label, country)
        if title in non_red_titles.get(country, set()):
            per_file_nonred[key] = compute_mean_emotions(result.get("reactions", []))
        elif title in red_titles.get(country, set()):
            per_file_red[key] = compute_mean_emotions(result.get("reactions", []))
        elif is_default_agent and title in english_titles.get(country, set()):
            # Allow English Experimental rewards titles for default_agent runs
            if "red tape" in title.lower():
                per_file_red[key] = compute_mean_emotions(result.get("reactions", []))
            else:
                per_file_nonred[key] = compute_mean_emotions(result.get("reactions", []))

    # Build paired diffs per model/country/emotion
    diffs: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    per_group_files: Dict[Tuple[str, str], List[Tuple[str, Dict[str, float], Dict[str, float]]]] = (
        defaultdict(list)
    )
    for key in per_file_nonred.keys() & per_file_red.keys():
        path, model, country = key
        nonred = per_file_nonred[key]
        red = per_file_red[key]
        per_group_files[(model, country)].append((path, nonred, red))
        for emotion in set(nonred.keys()) | set(red.keys()):
            if emotion in nonred and emotion in red:
                diffs[(model, country, emotion)].append(red[emotion] - nonred[emotion])

    rng = random.Random(RNG_SEED)

    # Print per-emotion stats
    for (model, country, emotion) in sorted(diffs.keys()):
        values = diffs[(model, country, emotion)]
        if not values:
            continue
        mean_diff = sum(values) / len(values)
        pval = sign_flip_pvalue(values, rng, PERMUTATION_ITERATIONS)
        print(
            f"{model} | {country} | {emotion} | n={len(values)} | "
            f"mean_diff(red-nonred)={mean_diff:.4f} | p≈{pval:.4f}"
        )

    # SigRate95 across emotions per model/country
    for (model, country) in sorted(per_group_files.keys()):
        files = per_group_files[(model, country)]
        emotions = sorted(
            {
                e
                for _, nonred, red in files
                for e in set(nonred.keys()) | set(red.keys())
                if e in nonred and e in red
            }
        )
        if not emotions:
            continue

        # Observed deltas per emotion
        observed = {}
        for e in emotions:
            vals = []
            for _, nonred, red in files:
                vals.append(red[e] - nonred[e])
            observed[e] = sum(vals) / len(vals)

        # Null distribution of |Δ| by shuffling condition within file
        null_abs = []
        for _ in range(PERMUTATION_ITERATIONS):
            perm_deltas = {e: [] for e in emotions}
            for _, nonred, red in files:
                flip = rng.random() < 0.5
                for e in emotions:
                    if flip:
                        perm_deltas[e].append(nonred[e] - red[e])
                    else:
                        perm_deltas[e].append(red[e] - nonred[e])
            for e in emotions:
                null_abs.append(abs(sum(perm_deltas[e]) / len(perm_deltas[e])))

        tau95 = percentile(null_abs, SIGRATE_Q)
        sig_count = sum(1 for e in emotions if abs(observed[e]) > tau95)
        sigrate = sig_count / len(emotions)
        print(
            f"SigRate95 | {model} | {country} | "
            f"tau95={tau95:.4f} | emotions={len(emotions)} | sigrate={sigrate:.4f}"
        )


if __name__ == "__main__":
    main()
