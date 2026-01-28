import random
import yaml
from pathlib import Path

random.seed(42)

# Target distribution (sum = 300)
CITY_COUNTS = {
    "HongKong": 95,
    "China": 75,
    "Germany": 70,          # "country" field will be Germany, with city-like traits via tags
    "San Francisco": 25,
    "Mumbai": 20,
    "London": 15,
}

GENDERS = ["female", "male", "non-binary"]

EDUCATION = ["high_school", "associate", "bachelor", "master", "phd"]
EDU_WEIGHTS = [0.10, 0.08, 0.42, 0.32, 0.08]

FAMILIES = ["single", "married", "partnered", "divorced", "widowed"]
FAM_WEIGHTS = [0.48, 0.30, 0.14, 0.06, 0.02]

PROFESSIONS = [
    "software engineer", "data scientist", "product manager", "designer",
    "teacher", "nurse", "doctor", "lawyer", "accountant", "consultant",
    "mechanical engineer", "civil engineer", "researcher", "entrepreneur",
    "marketing specialist", "sales", "chef", "artist", "journalist",
    "student", "finance analyst", "trader", "policy analyst", "social worker",
]
PROF_WEIGHTS = [
    0.18, 0.08, 0.07, 0.05,
    0.05, 0.04, 0.04, 0.04, 0.04, 0.05,
    0.04, 0.04, 0.05, 0.04,
    0.04, 0.03, 0.02, 0.02, 0.03,
    0.03, 0.04, 0.02, 0.02, 0.02
]

TRAITS_POOL = [
    "optimistic", "pessimistic", "introverted", "extroverted", "urban", "rural",
    "ambitious", "family-oriented", "risk-averse", "risk-tolerant", "tech-savvy",
    "traditional", "cosmopolitan", "environmentalist", "pragmatic", "creative",
    "disciplined", "empathetic", "competitive", "community-minded"
]

IDEOLOGIES = [
    "nationalist", "liberal", "conservative", "socialist", "centrist",
    "progressive", "libertarian", "environmentalist"
]

# Party labels are sensitive; keep them as self-declared strings, varied by region
PARTIES_BY_REGION = {
    "China": ["Chinese Communist Party", "non-partisan", "prefer not to say"],
    "HongKong": ["non-partisan", "pro-establishment", "pro-democracy", "prefer not to say"],
    "Germany": ["SPD", "CDU/CSU", "Greens", "FDP", "Left", "AfD", "non-partisan", "prefer not to say"],
    "San Francisco": ["Democratic Party", "Republican Party", "Independent", "prefer not to say"],
    "Mumbai": ["BJP", "INC", "AAP", "Shiv Sena", "non-partisan", "prefer not to say"],
    "London": ["Labour", "Conservative", "Liberal Democrats", "Green", "SNP", "non-partisan", "prefer not to say"],
}

# Names (simple curated pools to avoid external deps)
NAMES = {
    "China": ["Lucy", "Yating", "Mingyu", "Chenxi", "Jiawei", "Haoran", "Zihan", "Yue", "Tianqi", "Rui", "Xinyi", "Shuai"],
    "HongKong": ["Ka-Yan", "Wing", "Chloe", "Jason", "Michelle", "Oscar", "Vivian", "Eason", "Suki", "Ivan", "Kelly", "Aaron"],
    "Germany": ["Lena", "Jonas", "Mia", "Noah", "Hannah", "Leon", "Sofia", "Finn", "Lea", "Paul", "Emma", "Ben"],
    "San Francisco": ["Ava", "Ethan", "Sophia", "Mason", "Olivia", "Liam", "Isabella", "Noah", "Amelia", "Lucas"],
    "Mumbai": ["Aarav", "Anaya", "Vihaan", "Isha", "Arjun", "Diya", "Kabir", "Meera", "Rohan", "Saanvi"],
    "London": ["Harry", "Olivia", "Jack", "Amelia", "Charlie", "Emily", "George", "Sophie", "Arthur", "Grace"],
}

def weighted_choice(items, weights):
    return random.choices(items, weights=weights, k=1)[0]

def make_traits(region):
    base = set(random.sample(TRAITS_POOL, k=random.randint(2, 4)))
    # Regional flavor nudges
    if region in ("China", "HongKong", "San Francisco", "London"):
        base.add("urban")
    if region == "San Francisco":
        base.add("tech-savvy")
        if random.random() < 0.25: base.add("progressive")
    if region == "HongKong":
        base.add("cosmopolitan")
    if region == "Germany":
        if random.random() < 0.35: base.add("disciplined")
        if random.random() < 0.25: base.add("environmentalist")
    if region == "Mumbai":
        if random.random() < 0.30: base.add("community-minded")
    return sorted(base)

def make_agent(i, region):
    gender = random.choices(GENDERS, weights=[0.48, 0.48, 0.04], k=1)[0]
    age = str(random.randint(20, 62))
    edu = weighted_choice(EDUCATION, EDU_WEIGHTS)
    fam = weighted_choice(FAMILIES, FAM_WEIGHTS)
    prof = weighted_choice(PROFESSIONS, PROF_WEIGHTS)
    traits = make_traits(region)

    ideology = random.choices(
        IDEOLOGIES,
        weights=[0.18, 0.16, 0.14, 0.10, 0.18, 0.12, 0.06, 0.06],
        k=1
    )[0]

    party = random.choice(PARTIES_BY_REGION[region])

    # pick a name and disambiguate duplicates
    name = random.choice(NAMES[region])
    agent = {
        "name": f"{name}_{i:03d}",
        "country": region,
        "gender": gender,
        "age": age,
        "education": edu,
        "traits": traits,
        "profession": prof,
        "family": fam,
        "political_ideology": ideology,
        "political_party": party,
    }
    return agent

agents = []
idx = 1
for region, count in CITY_COUNTS.items():
    for _ in range(count):
        agents.append(make_agent(idx, region))
        idx += 1

random.shuffle(agents)

output_path = Path("data/personas.yaml")
with output_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(agents, f, sort_keys=False, allow_unicode=True)

