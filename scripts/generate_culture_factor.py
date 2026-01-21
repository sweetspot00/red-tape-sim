import random
from pathlib import Path
import yaml

"""
Generate per-person Hofstede-like cultural scores for a small set of countries.
We assume a "stable culture" spread, so we use a modest std dev and clip to [0, 100].
"""

ATTRIBUTES = [
    "power_distance",
    "individualism",
    "masculinity",
    "uncertainty_avoidance",
    "long_term_orientation",
    "indulgence",
]

# Country means from user-provided targets
COUNTRY_MEANS = {
    "China": [80, 43, 66, 30, 77, 24],
    "Germany": [35, 79, 66, 65, 57, 40],
    "Hong Kong": [68, 50, 57, 29, 93, 17],
}

PEOPLE_PER_COUNTRY = 200
# Stable culture: keep variance modest; adjust if you want more/less spread
STD_DEV = 16 

random.seed(42)


def truncated_gaussian(mean: float, std_dev: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """Sample from a gaussian and clip to bounds to avoid unrealistic scores."""
    value = random.gauss(mean, std_dev)
    return max(lower, min(upper, value))


def generate_country_scores(country: str, means: list[float]) -> list[dict]:
    """Generate PEOPLE_PER_COUNTRY rows of cultural scores for a single country."""
    scores = []
    for i in range(1, PEOPLE_PER_COUNTRY + 1):
        entry = {"id": f"{country.replace(' ', '')}_{i:03d}"}
        for attr, mean in zip(ATTRIBUTES, means):
            entry[attr] = round(truncated_gaussian(mean, STD_DEV), 2)
        scores.append(entry)
    return scores


def main() -> None:
    dataset = {
        "metadata": {
            "attributes": ATTRIBUTES,
            "std_dev": STD_DEV,
            "people_per_country": PEOPLE_PER_COUNTRY,
            "note": "Scores sampled from gaussian with clipping to [0, 100] to avoid outliers.",
        },
        "countries": {},
    }

    for country, means in COUNTRY_MEANS.items():
        dataset["countries"][country] = generate_country_scores(country, means)

    output_path = Path("data/culture_factors.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
