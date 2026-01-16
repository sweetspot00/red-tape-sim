import asyncio
import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, List, Optional

from .events import Event
from .filters import filter_personas
from .llm import LLMClient
from .personas import Persona

logger = logging.getLogger(__name__)


@dataclass
class Reaction:
    persona: Persona
    comment: str
    emotion: str


class PersonaAgent:
    """
    Simple agent that turns a persona into an LLM prompt and returns the model output.
    """

    def __init__(self, persona: Persona, llm: LLMClient):
        self.persona = persona
        self.llm = llm

    def react_to_event(
        self,
        event: Event,
        context=None,
        include_optional_persona_fields: Optional[set[str]] = None,
    ) -> str:
        prompt = self.persona.build_prompt(
            event, context, include_optional=include_optional_persona_fields
        )
        logger.debug("Prompt for '%s': %s", self.persona.name, prompt)
        comment, emotion = self.llm.generate_reaction(prompt)
        return Reaction(persona=self.persona, comment=comment, emotion=emotion)


class EventSimulation:
    """
    Broadcasts events to persona agents and collects LLM-generated reactions.
    """

    def __init__(self, personas: Iterable[Persona], llm: Optional[LLMClient] = None):
        self.personas: List[Persona] = list(personas)
        self.llm = llm or LLMClient()
        self.agents: List[PersonaAgent] = [
            PersonaAgent(persona=p, llm=self.llm) for p in self.personas
        ]
        logger.debug("Initialized EventSimulation with %d personas", len(self.personas))

    def publish_event(
        self,
        event: Event,
        *,
        countries: Optional[set[str]] = None,
        traits: Optional[set[str]] = None,
        context=None,
        include_optional_persona_fields: Optional[set[str]] = None,
    ) -> List[Reaction]:
        targets = filter_personas(self.personas, countries=countries, traits=traits)
        reactions: List[Reaction] = []
        for agent in self.agents:
            if agent.persona in targets:
                logger.info("Broadcasting event '%s' to persona '%s'", event.title, agent.persona.name)
                reactions.append(
                    agent.react_to_event(
                        event, context, include_optional_persona_fields=include_optional_persona_fields
                    )
                )
        logger.info("Collected %d reactions for event '%s'", len(reactions), event.title)
        return reactions

    async def publish_event_async(
        self,
        event: Event,
        *,
        countries: Optional[set[str]] = None,
        traits: Optional[set[str]] = None,
        context=None,
        include_optional_persona_fields: Optional[set[str]] = None,
        on_result: Optional[Callable[[Reaction], None]] = None,
    ) -> List[Reaction]:
        """
        Async version using thread pool for concurrent LLM calls.
        Calls on_result(reaction) as each completes.
        """
        targets = filter_personas(self.personas, countries=countries, traits=traits)
        loop = asyncio.get_running_loop()
        tasks = []
        for agent in self.agents:
            if agent.persona in targets:
                fn = partial(
                    agent.react_to_event,
                    event,
                    context,
                    include_optional_persona_fields,
                )
                tasks.append(loop.run_in_executor(None, fn))

        reactions: List[Reaction] = []
        for coro in asyncio.as_completed(tasks):
            reaction = await coro
            reactions.append(reaction)
            if on_result:
                on_result(reaction)
        logger.info("Collected %d reactions for event '%s' (async)", len(reactions), event.title)
        return reactions

    @staticmethod
    def summarize_emotions(reactions: List[Reaction]) -> dict:
        """
        Count emotions across reactions.
        """
        counts: dict[str, int] = {}
        for reaction in reactions:
            counts[reaction.emotion] = counts.get(reaction.emotion, 0) + 1
        total = len(reactions) or 1
        distribution = {k: v / total for k, v in counts.items()}
        return {"counts": counts, "distribution": distribution, "total": len(reactions)}
