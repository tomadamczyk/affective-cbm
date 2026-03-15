import json
import logging
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    CONCEPT_CATEGORIES,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    NUM_CONCEPTS,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)

CONCEPTS_DIR = PROCESSED_DATA_DIR / "concepts"
CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)
CONCEPTS_FILE = CONCEPTS_DIR / "concepts.json"

SEED_CATEGORIES: dict[str, list[str]] = CONCEPT_CATEGORIES
EMOTION_CATEGORIES = ["positive", "negative", "neutral"]

SYSTEM_PROMPT = (
    "You are an expert in affective neuroscience, computational linguistics, "
    "and emotion recognition.  Your task is to generate fine-grained concepts "
    "that characterize the emotional, cognitive, and linguistic properties of "
    "text.  These concepts will be used as an interpretable bottleneck layer "
    "in a multimodal EEG-text classification model."
)


def _build_category_prompt(emotion_category: str, target_count: int) -> str:
    seed_list = "\n".join(
        f"  - {cat}: {', '.join(items)}"
        for cat, items in SEED_CATEGORIES.items()
    )
    return f"""Generate exactly {target_count} fine-grained affective concepts that characterize text expressing **{emotion_category}** sentiment.

Each concept should be a measurable feature that could plausibly be detected both in text (via NLP) and in brain activity (via EEG).

Seed categories to cover (generate concepts spanning ALL of these):
{seed_list}

For each concept return a JSON object with these fields:
- "concept_name": snake_case identifier (e.g. "high_valence", "nostalgic_sadness")
- "category": one of {list(SEED_CATEGORIES.keys())}
- "description": 1-2 sentence description of what this concept captures
- "associated_emotions": list of fine-grained emotion labels this concept relates to (e.g. ["joy", "contentment"])

Return ONLY a JSON array of objects, no extra text or markdown fences."""


def _get_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY environment variable is not set. "
            "Add it to your .env file or export it in your shell."
        )
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _call_deepseek(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def _parse_concepts(raw_text: str) -> list[dict]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    text = text.strip()

    try:
        concepts = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM response as JSON: %s", exc)
        raise

    if not isinstance(concepts, list):
        raise ValueError(f"Expected a JSON array, got {type(concepts).__name__}")
    return concepts


_VALID_CATEGORIES = set(SEED_CATEGORIES.keys())


def _validate_concept(concept: dict) -> bool:
    required = {"concept_name", "category", "description", "associated_emotions"}
    if not required.issubset(concept.keys()):
        return False
    if not isinstance(concept["associated_emotions"], list):
        return False
    return True


def _deduplicate(concepts: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for c in concepts:
        name = c["concept_name"]
        if name not in seen:
            seen.add(name)
            unique.append(c)
    return unique


def _assign_emotion_category(concept: dict, emotion_category: str) -> dict:
    concept["emotion_category"] = emotion_category
    return concept


def generate_concepts(
    force: bool = False,
    output_path: Path | None = None,
) -> list[dict]:
    save_path = output_path or CONCEPTS_FILE

    if save_path.exists() and not force:
        logger.info("Loading cached concepts from %s", save_path)
        with open(save_path) as fh:
            concepts = json.load(fh)
        logger.info("Loaded %d concepts from cache.", len(concepts))
        return concepts

    logger.info("Generating concepts via DeepSeek (%s) ...", DEEPSEEK_MODEL)
    client = _get_client()

    per_category = NUM_CONCEPTS // len(EMOTION_CATEGORIES)
    remainder = NUM_CONCEPTS % len(EMOTION_CATEGORIES)

    all_concepts: list[dict] = []

    for idx, emotion_cat in enumerate(EMOTION_CATEGORIES):
        target = per_category + (1 if idx < remainder else 0)
        prompt = _build_category_prompt(emotion_cat, target)

        logger.info("Requesting %d concepts for '%s' category ...", target, emotion_cat)
        raw = _call_deepseek(client, prompt)
        batch = _parse_concepts(raw)
        batch = [_assign_emotion_category(c, emotion_cat) for c in batch]
        batch = [c for c in batch if _validate_concept(c)]
        logger.info("Received %d valid concepts for '%s'.", len(batch), emotion_cat)
        all_concepts.extend(batch)

    all_concepts = _deduplicate(all_concepts)

    if len(all_concepts) > NUM_CONCEPTS:
        all_concepts = all_concepts[:NUM_CONCEPTS]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as fh:
        json.dump(all_concepts, fh, indent=2)
    logger.info("Saved %d concepts to %s", len(all_concepts), save_path)

    return all_concepts
