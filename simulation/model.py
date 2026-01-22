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
    comment: str = ""  # placeholder for future text responses
    emotion: dict | None = None  # emotion probabilities keyed by emotion name


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
        logger.debug("Built prompt for '%s': %s", self.persona.name, prompt)
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
        if self.persona.country.lower() == "germany":
            emotions.append("frustration")
        try:
            emotion_dict, status = self.llm.generate_reaction(prompt, emotions=emotions)
            if status != "success":
                logger.warning(
                    "LLM returned status '%s' for persona '%s'; keeping raw content.",
                    status,
                    self.persona.name,
                )
        except Exception as exc:
            logger.warning("Failed to generate reaction for '%s': %s", self.persona.name, exc)
            emotion_dict = {"error": str(exc)}
        return Reaction(persona=self.persona, emotion=emotion_dict)


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
                try:
                    reactions.append(
                        agent.react_to_event(
                            event, context, include_optional_persona_fields=include_optional_persona_fields
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Reaction failed for persona '%s' on event '%s': %s",
                        agent.persona.name,
                        event.title,
                        exc,
                    )
                    reactions.append(
                        Reaction(persona=agent.persona, emotion={"error": str(exc)}, comment="error")
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
                def safe_react(agent=agent):
                    try:
                        return agent.react_to_event(
                            event,
                            context,
                            include_optional_persona_fields,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Async reaction failed for persona '%s' on event '%s': %s",
                            agent.persona.name,
                            event.title,
                            exc,
                        )
                        return Reaction(persona=agent.persona, emotion={"error": str(exc)}, comment="error")

                tasks.append(loop.run_in_executor(None, safe_react))

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
        Aggregate emotion probabilities across reactions and compute per-emotion means.
        """
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for reaction in reactions:
            if not isinstance(reaction.emotion, dict):
                continue
            for emo, val in reaction.emotion.items():
                try:
                    score = float(val)
                except (TypeError, ValueError):
                    continue
                totals[emo] = totals.get(emo, 0.0) + score
                counts[emo] = counts.get(emo, 0) + 1

        means = {emo: (totals[emo] / counts[emo]) for emo in totals}
        return {
            "means": means,
            "totals": totals,
            "counts": counts,
            "num_reactions": len(reactions),
        }
