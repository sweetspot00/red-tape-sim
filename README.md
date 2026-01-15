# red-tape-sim

Persona-driven event simulation scaffold that uses an LLM to generate reactions. Define persona prompts and events as data, filter agents by country/traits, and broadcast events to get LLM responses.

## Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start
```
export OPENAI_API_KEY=...
# optional: export OPENAI_BASE_URL=...  export OPENAI_MODEL=gpt-4o-mini
python main.py
```

## Core ideas
- Personas live in `simulation/personas.py` and can be serialized/deserialized for easy customization or user input.
- Country prompts can be shared: add templates in `data/country_prompts.yaml`, and personas from the same country will reuse them unless they set their own `prompt_template`.
- Prompts live with personas; edit `prompt_template` strings directly (in code or YAML) to change how each persona speaks.
- Filtering happens via `filter_personas` so you can slice by country or traits before broadcasting.
- `EventSimulation` pairs personas with `LLMClient` and sends prompts for each published `Event`.
- Each reaction now captures a single-word emotion from: peace, anger, contempt, fear, disgust, joy, sadness, surprise, confusion, frustration; use `EventSimulation.summarize_emotions` to get counts and distribution.
- Personas require: name and country (country may be blank for a generic agent). Optional: gender, age, education, profession, background, family, political_ideology, political_party, traits. Default prompts include only provided fields; pass `include_optional_persona_fields` to `publish_event` to add optional ones.

## Editing personas/events in YAML
- Persona definitions: `data/personas.yaml` (name, country, optional gender/age/education/traits and optional fields above, prompt_template, metadata).
- Event definitions: `data/events.yaml` (title, country, description, optional metadata, optional `non_red_tape` and `red_tape` step lists).
- Run `python main.py` after editing; the demo loads from those files.

Programmatic load:
```python
from simulation import (
    EventSimulation,
    LLMClient,
    load_country_prompts_from_yaml,
    load_personas_from_yaml,
    load_events_from_yaml,
)

country_prompts = load_country_prompts_from_yaml("data/country_prompts.yaml")
personas = load_personas_from_yaml("data/personas.yaml", country_prompts=country_prompts)
events = load_events_from_yaml("data/events.yaml")
sim = EventSimulation(personas, llm=LLMClient())
reactions = sim.publish_event(events[0], countries={"US", "CN"}, traits={"tech"})
stats = EventSimulation.summarize_emotions(reactions)
```

## Customizing personas
```python
from simulation import Persona, Event, EventSimulation, LLMClient

personas = [
    Persona(
        name="Sofia",
        traits={"optimistic", "urban"},
        country="ES",
        prompt_template="You are {name} from {country}, traits: {traits}. Event: {event}",
    ),
    Persona(name="Tunde", country="NG", traits={"rural", "entrepreneur"}),
]

sim = EventSimulation(personas, llm=LLMClient())
event = Event(title="Festival", description="A new arts festival is announced.")
reactions = sim.publish_event(
    event,
    countries={"ES"},
    include_optional_persona_fields={"profession"},
)
```

## Filtering
```python
from simulation import filter_personas

filtered = filter_personas(personas, countries={"ES", "NG"}, traits={"optimistic"})

## Including optional persona fields in prompts
reactions = sim.publish_event(
    events[0],
    include_optional_persona_fields={"profession", "background"},
)
```
