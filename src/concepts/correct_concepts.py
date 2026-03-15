import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DATA_DIR, SENTIMENT_LABELS

logger = logging.getLogger(__name__)

CONCEPTS_DIR = PROCESSED_DATA_DIR / "concepts"
CONCEPTS_FILE = CONCEPTS_DIR / "concepts.json"
COMBINED_SCORES_FILE = CONCEPTS_DIR / "combined_concept_scores.npy"
CORRECTED_SCORES_FILE = CONCEPTS_DIR / "corrected_concept_scores.npy"

_LABEL_TO_NAME: dict[int, str] = {v: k for k, v in SENTIMENT_LABELS.items()}


def _load_concepts(path: Path | None = None) -> list[dict]:
    p = path or CONCEPTS_FILE
    if not p.exists():
        raise FileNotFoundError(f"Concepts file not found at {p}. Run generate_concepts.py first.")
    with open(p) as fh:
        return json.load(fh)


def combine_scores(
    text_scores: np.ndarray,
    eeg_scores: np.ndarray,
    text_weight: float = 0.5,
    save: bool = True,
    output_path: Path | None = None,
) -> np.ndarray:
    if text_scores.shape != eeg_scores.shape:
        raise ValueError(
            f"Score matrix shapes must match: text={text_scores.shape}, eeg={eeg_scores.shape}"
        )

    eeg_weight = 1.0 - text_weight
    combined = (text_weight * text_scores + eeg_weight * eeg_scores).astype(np.float32)

    if save:
        path = output_path or COMBINED_SCORES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, combined)

    return combined


def _concept_conflicts_with_label(concept: dict, label_name: str) -> bool:
    concept_cat = concept.get("emotion_category")
    if concept_cat is None:
        return False
    if concept_cat == "neutral":
        return False
    return concept_cat != label_name


def correct_concepts(
    scores: np.ndarray,
    labels: np.ndarray,
    concepts: list[dict] | None = None,
    save: bool = True,
    output_path: Path | None = None,
) -> np.ndarray:
    if concepts is None:
        concepts = _load_concepts()

    n_samples, n_concepts = scores.shape
    corrected = scores.copy().astype(np.float32)

    label_values = sorted(_LABEL_TO_NAME.keys())
    conflict_masks: dict[int, np.ndarray] = {}
    for lv in label_values:
        label_name = _LABEL_TO_NAME[lv]
        mask = np.array(
            [_concept_conflicts_with_label(c, label_name) for c in concepts],
            dtype=bool,
        )
        conflict_masks[lv] = mask

    zeroed_total = 0
    for i in range(n_samples):
        label = int(labels[i])
        mask = conflict_masks.get(label)
        if mask is None:
            continue
        corrected[i, mask] = 0.0
        zeroed_total += int(mask.sum())

    logger.info(
        "ACC complete: zeroed %d concept-sample entries (%.1f%% of total).",
        zeroed_total,
        100.0 * zeroed_total / max(n_samples * n_concepts, 1),
    )

    if save:
        path = output_path or CORRECTED_SCORES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, corrected)

    return corrected
