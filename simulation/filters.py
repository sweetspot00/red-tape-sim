from typing import Iterable, List, Optional, Set

from .personas import Persona


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
    for persona in personas:
        if countries is not None and persona.country not in countries:
            continue
        if traits is not None and not traits.issubset(persona.traits):
            continue
        filtered.append(persona)
    return filtered
