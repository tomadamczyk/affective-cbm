import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DEVICE,
    EEG_BANDS,
    N_NEURAL_CONCEPTS,
    NEURAL_CONCEPT_NAMES,
    NUM_CONCEPTS,
    PROCESSED_DATA_DIR,
    TEXT_MODEL_NAME,
)

logger = logging.getLogger(__name__)

CONCEPTS_DIR = PROCESSED_DATA_DIR / "concepts"
CONCEPTS_FILE = CONCEPTS_DIR / "concepts.json"
TEXT_SCORES_FILE = CONCEPTS_DIR / "text_concept_scores.npy"
EEG_SCORES_FILE = CONCEPTS_DIR / "eeg_concept_scores.npy"

BAND_INDEX = {name: idx for idx, name in enumerate(EEG_BANDS.keys())}

_FRONTAL = list(range(0, 20))
_POSTERIOR = list(range(50, 110))
_ALL = list(range(128))
_F3, _F4 = 3, 7


def _mean_power(eeg: np.ndarray, channels: list[int], bands: list[str]) -> float:
    band_idxs = [BAND_INDEX[b] for b in bands]
    return float(eeg[np.ix_(channels, band_idxs)].mean())


def _frontal_alpha_asymmetry(eeg: np.ndarray) -> float:
    alpha_bands = ["alpha1", "alpha2"]
    bidxs = [BAND_INDEX[b] for b in alpha_bands]
    f4_alpha = float(eeg[_F4, bidxs].mean())
    f3_alpha = float(eeg[_F3, bidxs].mean())
    return f4_alpha - f3_alpha


EEG_CONCEPT_EXTRACTORS: dict[str, Any] = {
    "valence": _frontal_alpha_asymmetry,
    "positive_valence": _frontal_alpha_asymmetry,
    "negative_valence": lambda eeg: -_frontal_alpha_asymmetry(eeg),
    "arousal": lambda eeg: _mean_power(eeg, _ALL, ["beta1", "beta2"]),
    "high_arousal": lambda eeg: _mean_power(eeg, _ALL, ["beta1", "beta2"]),
    "low_arousal": lambda eeg: -_mean_power(eeg, _ALL, ["beta1", "beta2"]),
    "cognitive_load": lambda eeg: _mean_power(eeg, _FRONTAL, ["theta1", "theta2"]),
    "cognitive_overload": lambda eeg: _mean_power(eeg, _FRONTAL, ["theta1", "theta2"]),
    "engagement": lambda eeg: (
        _mean_power(eeg, _FRONTAL, ["theta1", "theta2"])
        / max(_mean_power(eeg, _FRONTAL, ["beta1", "beta2"]), 1e-8)
    ),
    "relaxation": lambda eeg: _mean_power(eeg, _POSTERIOR, ["alpha1", "alpha2"]),
    "dominance": lambda eeg: _mean_power(eeg, _FRONTAL, ["beta1", "beta2"]),
    "boredom": lambda eeg: _mean_power(eeg, _ALL, ["alpha1", "alpha2"]),
    "frustration": lambda eeg: _mean_power(eeg, _FRONTAL, ["beta2"]),
    "surprise": lambda eeg: _mean_power(eeg, _ALL, ["gamma1", "gamma2"]),
    "confusion": lambda eeg: _mean_power(eeg, _FRONTAL, ["theta1", "theta2"]),
}


def _load_concepts(path: Path | None = None) -> list[dict]:
    p = path or CONCEPTS_FILE
    if not p.exists():
        raise FileNotFoundError(f"Concepts file not found at {p}. Run generate_concepts.py first.")
    with open(p) as fh:
        return json.load(fh)


def _mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embs = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
    summed = (token_embs * mask_expanded).sum(dim=1)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def _encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 64,
) -> np.ndarray:
    all_embs: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
        output = model(**encoded)
        embs = _mean_pooling(output, encoded["attention_mask"])
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def _cosine_scores(sentence_embs: np.ndarray, concept_embs: np.ndarray) -> np.ndarray:
    sim = sentence_embs @ concept_embs.T
    return (sim + 1.0) / 2.0


def _normalise_01(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    mn = arr.min(axis=axis, keepdims=True)
    mx = arr.max(axis=axis, keepdims=True)
    denom = mx - mn
    denom = np.where(denom < 1e-12, 1.0, denom)
    return (arr - mn) / denom


def score_text_concepts(
    sentences: list[str],
    concepts: list[dict] | None = None,
    model_name: str = TEXT_MODEL_NAME,
    save: bool = True,
    output_path: Path | None = None,
) -> np.ndarray:
    if concepts is None:
        concepts = _load_concepts()

    n_concepts = len(concepts)
    logger.info("Scoring %d sentences against %d concepts using '%s' ...",
                len(sentences), n_concepts, model_name)

    concept_texts = [
        f"{c['concept_name'].replace('_', ' ')}: {c['description']}"
        for c in concepts
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)

    concept_embs = _encode_texts(concept_texts, tokenizer, model)
    sentence_embs = _encode_texts(sentences, tokenizer, model)

    scores = _cosine_scores(sentence_embs, concept_embs)

    mu = scores.mean(axis=0, keepdims=True)
    std = scores.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    scores = 1.0 / (1.0 + np.exp(-(scores - mu) / std))

    if save:
        path = output_path or TEXT_SCORES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, scores)

    return scores


def score_eeg_concepts(
    eeg_features: np.ndarray,
    concepts: list[dict] | None = None,
    text_scores: np.ndarray | None = None,
    save: bool = True,
    output_path: Path | None = None,
) -> np.ndarray:
    if concepts is None:
        concepts = _load_concepts()

    n_samples = eeg_features.shape[0]
    n_concepts = len(concepts)

    scores = np.zeros((n_samples, n_concepts), dtype=np.float32)

    eeg_derived_count = 0
    for c_idx, concept in enumerate(concepts):
        name = concept["concept_name"].lower()
        extractor = EEG_CONCEPT_EXTRACTORS.get(name)

        if extractor is None:
            for key, ext_fn in EEG_CONCEPT_EXTRACTORS.items():
                if key in name or name in key:
                    extractor = ext_fn
                    break

        if extractor is not None:
            raw_vals = np.array(
                [extractor(eeg_features[i]) for i in range(n_samples)],
                dtype=np.float32,
            )
            scores[:, c_idx] = raw_vals
            eeg_derived_count += 1

    scores = _normalise_01(scores, axis=0).astype(np.float32)

    if save:
        path = output_path or EEG_SCORES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, scores)

    return scores


_P3, _P4 = 52, 58
_CENTRAL = list(range(20, 50))
_TEMPORAL = list(range(110, 128))
_LEFT_TEMPORAL = list(range(110, 119))
_OCCIPITAL = list(range(80, 110))
_FZ = 0


def _extract_neural_concepts(eeg: np.ndarray) -> np.ndarray:
    alpha_bands = [BAND_INDEX["alpha1"], BAND_INDEX["alpha2"]]
    theta_bands = [BAND_INDEX["theta1"], BAND_INDEX["theta2"]]
    beta_bands = [BAND_INDEX["beta1"], BAND_INDEX["beta2"]]
    gamma_bands = [BAND_INDEX["gamma1"], BAND_INDEX["gamma2"]]
    n_ch = eeg.shape[0]

    def _safe_ch(idx):
        return min(idx, n_ch - 1)

    f3, f4 = _safe_ch(_F3), _safe_ch(_F4)
    p3, p4 = _safe_ch(_P3), _safe_ch(_P4)
    fz = _safe_ch(_FZ)

    def _region_mean(channels, bands):
        valid = [c for c in channels if c < n_ch]
        if not valid:
            return 0.0
        return float(eeg[np.ix_(valid, bands)].mean())

    scores = np.zeros(N_NEURAL_CONCEPTS, dtype=np.float64)
    scores[0] = eeg[f4, alpha_bands].mean() - eeg[f3, alpha_bands].mean()
    scores[1] = eeg[p4, alpha_bands].mean() - eeg[p3, alpha_bands].mean()
    scores[2] = _region_mean(_FRONTAL, theta_bands)
    parietal = [c for c in range(50, 80) if c < n_ch]
    scores[3] = _region_mean(parietal, alpha_bands)
    scores[4] = _region_mean(_CENTRAL, beta_bands)
    scores[5] = _region_mean(_TEMPORAL, gamma_bands)
    theta_mean = _region_mean(list(range(n_ch)), theta_bands)
    beta_mean = _region_mean(list(range(n_ch)), beta_bands)
    scores[6] = theta_mean / max(beta_mean, 1e-8)
    alpha_mean = _region_mean(list(range(n_ch)), alpha_bands)
    scores[7] = alpha_mean / max(theta_mean, 1e-8)
    scores[8] = float(eeg[fz, theta_bands].mean())
    scores[9] = -_region_mean(_POSTERIOR, alpha_bands)
    gamma_mean = _region_mean(list(range(n_ch)), gamma_bands)
    scores[10] = beta_mean / max(gamma_mean, 1e-8)
    scores[11] = _region_mean(_LEFT_TEMPORAL, gamma_bands)
    scores[12] = alpha_mean
    scores[13] = eeg[f4, beta_bands].mean() - eeg[f3, beta_bands].mean()
    scores[14] = _region_mean(_OCCIPITAL, alpha_bands)

    return scores


def extract_neural_concept_scores(
    eeg_features: np.ndarray,
    save: bool = True,
    output_path: Path | None = None,
) -> np.ndarray:
    n_samples = eeg_features.shape[0]

    raw_scores = np.array(
        [_extract_neural_concepts(eeg_features[i]) for i in range(n_samples)],
        dtype=np.float64,
    )

    mu = raw_scores.mean(axis=0, keepdims=True)
    std = raw_scores.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    scores = 1.0 / (1.0 + np.exp(-(raw_scores - mu) / std))
    scores = scores.astype(np.float32)

    if save:
        path = output_path or (CONCEPTS_DIR / "neural_concept_scores.npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, scores)

    return scores
