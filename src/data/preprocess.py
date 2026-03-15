import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch

try:
    from config import (
        DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
        EEG_BANDS, N_EEG_BANDS, PROCESSED_DATA_DIR, RAW_DATA_DIR,
        SENTIMENT_LABELS, ZUCO_N_CHANNELS,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import (
        DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
        EEG_BANDS, N_EEG_BANDS, PROCESSED_DATA_DIR, RAW_DATA_DIR,
        SENTIMENT_LABELS, ZUCO_N_CHANNELS,
    )

logger = logging.getLogger(__name__)

BAND_FIELD_MAP = {
    "theta1": "mean_t1", "theta2": "mean_t2",
    "alpha1": "mean_a1", "alpha2": "mean_a2",
    "beta1":  "mean_b1", "beta2":  "mean_b2",
    "gamma1": "mean_g1", "gamma2": "mean_g2",
}
BAND_NAMES: List[str] = list(EEG_BANDS.keys())


def _read_string(f: h5py.File, ref) -> str:
    obj = f[ref] if isinstance(ref, h5py.Reference) else ref
    if isinstance(obj, h5py.Dataset):
        data = obj[()]
    else:
        data = np.array(obj)
    data = np.array(data).flatten()
    try:
        return "".join(chr(int(c)) for c in data)
    except (ValueError, TypeError):
        return str(data)


def _read_band_vector(f: h5py.File, ref) -> Optional[np.ndarray]:
    try:
        obj = f[ref] if isinstance(ref, h5py.Reference) else ref
        arr = np.array(obj, dtype=np.float64).flatten()
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


_POS_WORDS = frozenset(
    "good great excellent amazing wonderful fantastic awesome love happy joy "
    "beautiful brilliant best terrific superb perfect delightful warm "
    "outstanding remarkable magnificent impressive inspiring heartwarming "
    "charming pleasant enjoyable satisfying positive uplifting".split()
)
_NEG_WORDS = frozenset(
    "bad terrible awful horrible worst hate sad angry boring dull "
    "disappointing poor ugly disgusting painful miserable dreadful "
    "atrocious appalling abysmal wretched lousy mediocre fail lacking "
    "annoying frustrating negative depressing offensive".split()
)


def _heuristic_sentiment(text: str) -> int:
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    pos_count = len(tokens & _POS_WORDS)
    neg_count = len(tokens & _NEG_WORDS)
    if pos_count > neg_count:
        return SENTIMENT_LABELS["positive"]
    elif neg_count > pos_count:
        return SENTIMENT_LABELS["negative"]
    return SENTIMENT_LABELS["neutral"]


def _model_sentiment(texts: List[str]) -> List[int]:
    from transformers import pipeline as hf_pipeline

    classifier = hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1, truncation=True, max_length=512,
    )

    labels = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = classifier(batch)
        for r in results:
            score = r["score"]
            label_str = r["label"]
            if score < 0.7:
                labels.append(SENTIMENT_LABELS["neutral"])
            elif label_str == "POSITIVE":
                labels.append(SENTIMENT_LABELS["positive"])
            else:
                labels.append(SENTIMENT_LABELS["negative"])
    return labels


def _deepseek_sentiment(texts: List[str], batch_size: int = 25) -> List[int]:
    from openai import OpenAI

    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    labels: List[int] = []

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]
        numbered = "\n".join(f"{idx + 1}. {sent}" for idx, sent in enumerate(batch))

        prompt = (
            "Classify the sentiment of each sentence below as exactly one of: "
            "positive, negative, or neutral.\n\n"
            "Sentences:\n"
            f"{numbered}\n\n"
            "Return ONLY a JSON array of objects, one per sentence, in order. "
            'Each object must have "index" (1-based) and "sentiment" '
            '(one of "positive", "negative", "neutral"). '
            "No other text or explanation."
        )

        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0, max_tokens=4096,
        )
        raw_content = response.choices[0].message.content.strip()

        if raw_content.startswith("```"):
            lines = raw_content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw_content = "\n".join(lines).strip()

        parsed = json.loads(raw_content)

        for item in parsed:
            sentiment = item.get("sentiment", "").lower().strip()
            if sentiment in SENTIMENT_LABELS:
                labels.append(SENTIMENT_LABELS[sentiment])
            else:
                labels.append(SENTIMENT_LABELS["neutral"])

    return labels


def load_sst_lookup(sst_path: Path) -> Dict[str, int]:
    import pandas as pd
    df = pd.read_csv(sst_path, sep="\t" if sst_path.suffix == ".tsv" else ",")
    df.columns = [c.strip().lower() for c in df.columns]
    if "sentence" not in df.columns or "label" not in df.columns:
        return {}
    mapping: Dict[str, int] = {}
    for _, row in df.iterrows():
        raw_label = int(row["label"])
        if raw_label <= 1:
            label = SENTIMENT_LABELS["negative"]
        elif raw_label >= 3:
            label = SENTIMENT_LABELS["positive"]
        else:
            label = SENTIMENT_LABELS["neutral"]
        mapping[str(row["sentence"]).strip()] = label
    return mapping


def process_mat_file(mat_path: Path) -> List[Dict[str, Any]]:
    logger.info("Processing %s ...", mat_path.name)
    sentences: List[Dict[str, Any]] = []

    try:
        with h5py.File(str(mat_path), "r") as f:
            if "sentenceData" not in f:
                return sentences

            sd = f["sentenceData"]
            n_sents = sd["content"].shape[0]
            content_refs = np.array(sd["content"]).flatten()

            band_refs = {}
            for band_name, field_name in BAND_FIELD_MAP.items():
                if field_name in sd:
                    band_refs[band_name] = np.array(sd[field_name]).flatten()

            if len(band_refs) < N_EEG_BANDS:
                return sentences

            for i in range(n_sents):
                text = _read_string(f, content_refs[i]).strip()
                if len(text) < 2:
                    continue

                bands = []
                valid = True
                for band_name in BAND_NAMES:
                    vec = _read_band_vector(f, band_refs[band_name][i])
                    if vec is None:
                        valid = False
                        break
                    bands.append(vec)

                if not valid or len(bands) != N_EEG_BANDS:
                    continue

                min_ch = min(b.shape[0] for b in bands)
                eeg_features = np.stack([b[:min_ch] for b in bands], axis=-1)

                sentences.append({"text": text, "eeg_features": eeg_features})

    except Exception as exc:
        logger.error("Failed to read %s: %s", mat_path, exc, exc_info=True)

    return sentences


def preprocess_zuco(
    raw_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    sst_path: Optional[Path] = None,
    task_filter: str = "TSR",
) -> Path:
    raw_dir = raw_dir or RAW_DATA_DIR
    out_dir = out_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(raw_dir.rglob("*.mat"))
    if task_filter:
        mat_files = [p for p in mat_files if task_filter.lower() in str(p).lower()]

    if not mat_files:
        logger.error("No .mat files found in %s.", raw_dir)
        sys.exit(1)

    sst_lookup: Optional[Dict[str, int]] = None
    if sst_path and sst_path.exists():
        sst_lookup = load_sst_lookup(sst_path)

    all_sentences: List[Dict[str, Any]] = []
    for mat_path in mat_files:
        sentences = process_mat_file(mat_path)
        all_sentences.extend(sentences)

    if not all_sentences:
        logger.error("No sentences extracted.")
        sys.exit(1)

    text_to_entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sent in all_sentences:
        text_to_entries[sent["text"]].append(sent)

    unique_sentences: List[Dict[str, Any]] = []
    for text, entries in text_to_entries.items():
        eeg_list = [e["eeg_features"] for e in entries]
        min_ch = min(e.shape[0] for e in eeg_list)
        stacked = np.stack([np.nan_to_num(e[:min_ch], nan=0.0) for e in eeg_list])
        avg_eeg = np.nanmean(stacked, axis=0)
        avg_eeg = np.nan_to_num(avg_eeg, nan=0.0)
        unique_sentences.append({"text": text, "eeg_features": avg_eeg})

    all_texts = [s["text"] for s in unique_sentences]
    if sst_lookup:
        for sent in unique_sentences:
            if sent["text"] in sst_lookup:
                sent["label"] = sst_lookup[sent["text"]]
            else:
                sent["label"] = _heuristic_sentiment(sent["text"])
    else:
        labeled = False
        try:
            deepseek_labels = _deepseek_sentiment(all_texts)
            for sent, lbl in zip(unique_sentences, deepseek_labels):
                sent["label"] = lbl
            labeled = True
        except Exception:
            pass

        if not labeled:
            try:
                model_labels = _model_sentiment(all_texts)
                for sent, lbl in zip(unique_sentences, model_labels):
                    sent["label"] = lbl
                labeled = True
            except Exception:
                pass

        if not labeled:
            for sent in unique_sentences:
                sent["label"] = _heuristic_sentiment(sent["text"])

    texts: List[str] = []
    eeg_tensors: List[torch.Tensor] = []
    labels: List[int] = []

    for sent in unique_sentences:
        feat = sent["eeg_features"]
        n_elec = feat.shape[0]
        if n_elec < ZUCO_N_CHANNELS:
            pad = np.zeros((ZUCO_N_CHANNELS - n_elec, N_EEG_BANDS))
            feat = np.concatenate([feat, pad], axis=0)
        elif n_elec > ZUCO_N_CHANNELS:
            feat = feat[:ZUCO_N_CHANNELS, :]

        texts.append(sent["text"])
        eeg_tensors.append(torch.tensor(feat, dtype=torch.float32))
        labels.append(sent["label"])

    eeg_tensor = torch.stack(eeg_tensors)
    eeg_tensor = torch.nan_to_num(eeg_tensor, nan=0.0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    mean = eeg_tensor.mean(dim=0, keepdim=True)
    std = eeg_tensor.std(dim=0, keepdim=True).clamp(min=1e-8)
    eeg_tensor = (eeg_tensor - mean) / std
    eeg_tensor = torch.nan_to_num(eeg_tensor, nan=0.0)

    out_file = out_dir / f"zuco_{task_filter.replace('-', '_')}.pt"
    payload = {
        "texts": texts,
        "eeg_features": eeg_tensor,
        "labels": labels_tensor,
        "eeg_mean": mean.squeeze(0),
        "eeg_std": std.squeeze(0),
        "band_names": BAND_NAMES,
        "n_channels": ZUCO_N_CHANNELS,
        "n_bands": N_EEG_BANDS,
    }

    torch.save(payload, out_file)
    logger.info("Saved processed data to %s", out_file)
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ZuCo 2.0 .mat files into .pt tensors.")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sst-path", type=Path, default=None)
    parser.add_argument("--task", type=str, default="TSR")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    preprocess_zuco(raw_dir=args.raw_dir, out_dir=args.out_dir, sst_path=args.sst_path, task_filter=args.task)
