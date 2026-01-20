from typing import Iterable, List, Optional, Set

from .personas import Persona


def _norm_country(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def filter_personas(
    personas: Iterable[Persona],
    *,
    countries: Optional[Set[str]] = None,
    traits: Optional[Set[str]] = None,
) -> List[Persona]:
    """
    Return personas that match any of the supplied countries and contain all requested traits.
    Passing None leaves a dimension unfiltered.
    """
    filtered: List[Persona] = []
    norm_countries = {_norm_country(c) for c in countries} if countries is not None else None
    for persona in personas:
        if norm_countries is not None and _norm_country(persona.country) not in norm_countries:
            continue
        if traits is not None and not traits.issubset(persona.traits):
            continue
        filtered.append(persona)
    return filtered
