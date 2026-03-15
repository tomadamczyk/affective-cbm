#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BATCH_SIZE, CBL_LEARNING_RATE, DEVICE, EMBEDDING_DIM, MAX_GRAD_NORM,
    NEURAL_CONCEPT_NAMES, NUM_CLASSES, NUM_CONCEPTS, NUM_EPOCHS_CBL,
    PREDICTOR_L1_LAMBDA, PREDICTOR_LEARNING_RATE, PROCESSED_DATA_DIR,
    RANDOM_SEED, RESULTS_DIR, SENTIMENT_LABELS, WARMUP_STEPS, WEIGHT_DECAY,
)
from src.training.train_cbl import CBLModel
from src.training.train_predictor import SparseLinearPredictor
from src.training.utils import (
    EarlyStopping, WarmupScheduler, compute_metrics, save_checkpoint, set_seed,
)

logger = logging.getLogger(__name__)

CV_DIR = RESULTS_DIR / "cv"

MODEL_CONFIGS = {
    "text_only": "TextOnlyCBLModel",
    "eeg_only": "EEGOnlyCBLModel",
    "full_concat": "CBLModel_concat",
    "full_cross_modulation": "CBLModel_cross_modulation",
    "full_gated": "CBLModel_gated",
}


class TextOnlyCBLModel(nn.Module):
    def __init__(self, num_concepts: int = NUM_CONCEPTS):
        super().__init__()
        from src.models.text_encoder import TextEncoder
        from src.models.concept_bottleneck import ConceptBottleneckLayer

        self.text_encoder = TextEncoder(freeze_backbone=True)
        self.cbl = ConceptBottleneckLayer(input_dim=EMBEDDING_DIM, num_concepts=num_concepts)
        self.prediction_head = nn.Linear(num_concepts, NUM_CLASSES)

    def forward(self, eeg_input, input_ids, attention_mask):
        text_emb = self.text_encoder(input_ids, attention_mask)
        concept_activations = self.cbl(text_emb)
        logits = self.prediction_head(concept_activations)
        return concept_activations, logits


class EEGOnlyCBLModel(nn.Module):
    def __init__(self, num_concepts: int = NUM_CONCEPTS):
        super().__init__()
        from src.models.eeg_encoder import EEGEncoder
        from src.models.concept_bottleneck import ConceptBottleneckLayer

        self.eeg_encoder = EEGEncoder()
        self.cbl = ConceptBottleneckLayer(input_dim=EMBEDDING_DIM, num_concepts=num_concepts)
        self.prediction_head = nn.Linear(num_concepts, NUM_CLASSES)

    def forward(self, eeg_input, input_ids, attention_mask):
        eeg_emb = self.eeg_encoder(eeg_input)
        concept_activations = self.cbl(eeg_emb)
        logits = self.prediction_head(concept_activations)
        return concept_activations, logits


def load_experiment_data(use_synthetic: bool = False):
    pt_path = PROCESSED_DATA_DIR / "zuco_TSR.pt"

    if not pt_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {pt_path}. Run preprocessing first."
        )

    pt_data = torch.load(pt_path, map_location="cpu", weights_only=False)
    texts = pt_data["texts"]
    eeg_features = pt_data["eeg_features"].numpy()
    labels = pt_data["labels"].numpy()

    concepts_dir = PROCESSED_DATA_DIR / "concepts"
    corrected_path = concepts_dir / "corrected_concept_scores.npy"

    if corrected_path.exists():
        corrected_scores = np.load(corrected_path)
    else:
        from src.concepts.generate_concepts import generate_concepts
        from src.concepts.score_concepts import score_text_concepts, score_eeg_concepts
        from src.concepts.correct_concepts import combine_scores, correct_concepts

        concepts = generate_concepts()
        text_scores = score_text_concepts(texts, concepts)
        eeg_scores = score_eeg_concepts(eeg_features, concepts, text_scores=text_scores)
        combined = combine_scores(text_scores, eeg_scores)
        corrected_scores = correct_concepts(combined, labels, concepts)

    neural_path = concepts_dir / "neural_concept_scores.npy"
    if neural_path.exists():
        neural_scores = np.load(neural_path)
    else:
        from src.concepts.score_concepts import extract_neural_concept_scores
        neural_scores = extract_neural_concept_scores(eeg_features)

    all_concept_scores = np.concatenate([corrected_scores, neural_scores], axis=1)

    concepts_file = concepts_dir / "concepts.json"
    if concepts_file.exists():
        with open(concepts_file) as f:
            concepts = json.load(f)
        concept_names = [c["concept_name"] for c in concepts]
    else:
        concept_names = [f"concept_{i}" for i in range(corrected_scores.shape[1])]
    all_concept_names = concept_names + list(NEURAL_CONCEPT_NAMES)

    n_text_concepts = corrected_scores.shape[1]
    n_neural = neural_scores.shape[1]

    return (texts, eeg_features, labels, all_concept_scores,
            all_concept_names, n_text_concepts, n_neural)


@torch.no_grad()
def inference_ablation(
    model: nn.Module,
    test_loader: DataLoader,
    experiment_name: str,
    device: str | torch.device = DEVICE,
) -> dict:
    if not isinstance(model, CBLModel):
        return {}

    model.eval()
    model.to(device)

    conditions = {
        "both": lambda eeg, ids, mask: model.forward_with_ablation(eeg, ids, mask),
        "text_only": lambda eeg, ids, mask: model.forward_with_ablation(None, ids, mask),
        "eeg_only": lambda eeg, ids, mask: model.forward_with_ablation(eeg, None, None),
    }

    results = {}
    for condition_name, forward_fn in conditions.items():
        all_preds, all_labels = [], []
        for batch in test_loader:
            eeg, input_ids, attention_mask, _, labels = [b.to(device) for b in batch[:5]]
            _, logits = forward_fn(eeg, input_ids, attention_mask)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        metrics = compute_metrics(
            all_labels, all_preds, NUM_CLASSES,
            label_names=list(SENTIMENT_LABELS.keys()),
        )
        results[condition_name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_f1": metrics["per_class_f1"],
        }

    return results


def tokenize_all_data(texts: list[str]):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    return encoded["input_ids"], encoded["attention_mask"]


def create_fold_loaders(
    full_dataset: TensorDataset,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_labels = labels[train_idx]
    sub_train_idx, sub_val_idx = next(sss.split(train_idx, train_labels))

    actual_train_idx = train_idx[sub_train_idx]
    actual_val_idx = train_idx[sub_val_idx]

    train_ds = Subset(full_dataset, actual_train_idx.tolist())
    val_ds = Subset(full_dataset, actual_val_idx.tolist())
    test_ds = Subset(full_dataset, test_idx.tolist())

    train_labels_tensor = torch.tensor(labels[actual_train_idx], dtype=torch.long)
    class_counts = torch.bincount(train_labels_tensor, minlength=NUM_CLASSES).float()
    sample_weights = (1.0 / class_counts.clamp(min=1.0))[train_labels_tensor]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=sampler),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )


def create_model(
    model_name: str, total_concepts: int, n_text_concepts: int, n_neural: int
) -> nn.Module:
    if model_name == "text_only":
        return TextOnlyCBLModel(num_concepts=total_concepts)
    elif model_name == "eeg_only":
        return EEGOnlyCBLModel(num_concepts=total_concepts)
    elif model_name == "full_concat":
        return CBLModel(fusion_mode="concat", num_concepts=n_text_concepts, n_neural_concepts=n_neural)
    elif model_name == "full_cross_modulation":
        return CBLModel(fusion_mode="cross_modulation", num_concepts=n_text_concepts, n_neural_concepts=n_neural)
    elif model_name == "full_gated":
        return CBLModel(fusion_mode="gated", num_concepts=n_text_concepts, n_neural_concepts=n_neural)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_fold_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    fold_idx: int,
    num_epochs: int = NUM_EPOCHS_CBL,
    learning_rate: float = CBL_LEARNING_RATE,
    device: str | torch.device = DEVICE,
    patience: int = 10,
) -> tuple[dict, nn.Module]:
    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    total_steps = num_epochs * len(train_loader)
    scheduler = WarmupScheduler(optimizer, WARMUP_STEPS, total_steps)

    concept_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stopping = EarlyStopping(patience=patience, mode="max")

    best_f1 = -float("inf")
    best_epoch = 0
    best_state = None
    tag = f"fold{fold_idx}/{model_name}"

    for epoch in range(1, num_epochs + 1):
        alpha = 0.1 + 0.9 * (epoch - 1) / max(1, num_epochs - 1)

        model.train()
        for batch in tqdm(train_loader, desc=f"  [{tag}] epoch {epoch}", leave=False):
            eeg, input_ids, attention_mask, concept_targets, labels_batch = [
                b.to(device) for b in batch[:5]
            ]
            optimizer.zero_grad()
            concept_preds, logits = model(eeg, input_ids, attention_mask)
            loss = concept_criterion(concept_preds, concept_targets) + alpha * cls_criterion(logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                eeg, input_ids, attention_mask, concept_targets, labels_batch = [
                    b.to(device) for b in batch[:5]
                ]
                _, logits = model(eeg, input_ids, attention_mask)
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(labels_batch.cpu().tolist())

        val_metrics = compute_metrics(all_labels, all_preds, NUM_CLASSES)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ckpt_dir = CV_DIR / f"fold_{fold_idx}" / model_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(ckpt_dir / "best_model.pt", epoch=epoch, model=model, metrics=val_metrics)

        if early_stopping.step(val_metrics["macro_f1"]):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_concept_acts, all_gate_values = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            eeg, input_ids, attention_mask, concept_targets, labels_batch = [
                b.to(device) for b in batch[:5]
            ]
            concept_acts, logits = model(eeg, input_ids, attention_mask)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels_batch.cpu().tolist())
            all_concept_acts.append(concept_acts.cpu())

            if isinstance(model, CBLModel) and model.fusion_mode == "gated":
                gate_vals = model.get_gate_values()
                if gate_vals is not None:
                    all_gate_values.append(gate_vals.cpu())

    test_metrics = compute_metrics(all_labels, all_preds, NUM_CLASSES, label_names=list(SENTIMENT_LABELS.keys()))
    concept_activations = torch.cat(all_concept_acts, dim=0)

    result = {
        "model_name": model_name, "fold": fold_idx, "best_epoch": best_epoch,
        "best_val_f1": best_f1, "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_per_class_f1": test_metrics["per_class_f1"],
        "concept_activations": concept_activations.numpy(),
    }

    if all_gate_values:
        gate_tensor = torch.cat(all_gate_values, dim=0)
        result["gate_values"] = gate_tensor.numpy()
        result["mean_text_gate"] = float(gate_tensor[:, 0].mean())
        result["mean_eeg_gate"] = float(gate_tensor[:, 1].mean())

    if isinstance(model, CBLModel):
        result["inference_ablation"] = inference_ablation(model, test_loader, tag, device=device)

    return result, model


def train_predictor_on_activations(
    concept_acts: np.ndarray,
    labels: np.ndarray,
    num_epochs: int = 30,
    l1_lambda: float = PREDICTOR_L1_LAMBDA,
    device: str | torch.device = DEVICE,
) -> SparseLinearPredictor:
    n_concepts = concept_acts.shape[1]
    predictor = SparseLinearPredictor(num_concepts=n_concepts).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=PREDICTOR_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    dataset = TensorDataset(
        torch.tensor(concept_acts, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    predictor.train()
    for _ in range(num_epochs):
        for batch in loader:
            acts, labs = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            logits = predictor(acts)
            loss = criterion(logits, labs) + l1_lambda * predictor.l1_penalty()
            loss.backward()
            optimizer.step()

    predictor.eval()
    return predictor


def run_cv_experiments(
    model_names: list[str] | None = None,
    num_epochs: int = NUM_EPOCHS_CBL,
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> dict:
    set_seed(seed)

    if model_names is None:
        model_names = list(MODEL_CONFIGS.keys())

    (texts, eeg_features, labels, all_concept_scores,
     all_concept_names, n_text_concepts, n_neural) = load_experiment_data()

    total_concepts = all_concept_scores.shape[1]
    logger.info("Data: %d samples, %d concepts (text=%d + neural=%d)",
                len(texts), total_concepts, n_text_concepts, n_neural)

    input_ids, attention_mask = tokenize_all_data(texts)

    eeg_tensor = (
        torch.tensor(np.array(eeg_features), dtype=torch.float32)
        if not isinstance(eeg_features, torch.Tensor) else eeg_features.float()
    )
    concept_tensor = (
        torch.tensor(np.array(all_concept_scores), dtype=torch.float32)
        if not isinstance(all_concept_scores, torch.Tensor) else all_concept_scores.float()
    )
    labels_array = np.array(labels)
    labels_tensor = torch.tensor(labels_array, dtype=torch.long)

    full_dataset = TensorDataset(eeg_tensor, input_ids, attention_mask, concept_tensor, labels_tensor)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    all_fold_results: dict[str, list[dict]] = {name: [] for name in model_names}
    all_predictor_weights: dict[str, list[np.ndarray]] = {name: [] for name in model_names}
    all_gate_values: list[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(labels_array)), labels_array)
    ):
        logger.info("\nFOLD %d/%d  (train=%d, test=%d)", fold_idx + 1, n_folds, len(train_idx), len(test_idx))

        train_loader, val_loader, test_loader = create_fold_loaders(
            full_dataset, labels_array, train_idx, test_idx, seed=seed + fold_idx,
        )

        for model_name in model_names:
            logger.info("\n--- %s (fold %d) ---", model_name, fold_idx)

            model = create_model(model_name, total_concepts, n_text_concepts, n_neural)
            result, trained_model = train_fold_model(
                model, train_loader, val_loader, test_loader,
                model_name=model_name, fold_idx=fold_idx, num_epochs=num_epochs, device=DEVICE,
            )

            train_concept_acts, train_labels_list = [], []
            trained_model.eval()
            with torch.no_grad():
                for batch in train_loader:
                    eeg, ids, mask = [b.to(DEVICE) for b in batch[:3]]
                    acts, _ = trained_model(eeg, ids, mask)
                    train_concept_acts.append(acts.cpu())
                    train_labels_list.append(batch[4])

            predictor = train_predictor_on_activations(
                torch.cat(train_concept_acts).numpy(),
                torch.cat(train_labels_list).numpy(),
            )
            all_predictor_weights[model_name].append(predictor.get_weight_matrix().numpy())

            if "gate_values" in result:
                all_gate_values.append(result["gate_values"])

            result_clean = {
                k: v for k, v in result.items()
                if k not in ("concept_activations", "gate_values", "predictor_weights")
            }
            all_fold_results[model_name].append(result_clean)

            del model, trained_model, predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary: dict[str, dict] = {}
    for model_name in model_names:
        folds = all_fold_results[model_name]
        accs = [f["test_accuracy"] for f in folds]
        f1s = [f["test_macro_f1"] for f in folds]

        summary[model_name] = {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "per_fold": folds,
        }

        per_class: dict[str, dict] = {}
        for cls_name in SENTIMENT_LABELS.keys():
            cls_f1s = [f["test_per_class_f1"].get(cls_name, 0.0) for f in folds]
            per_class[cls_name] = {"f1_mean": float(np.mean(cls_f1s)), "f1_std": float(np.std(cls_f1s))}
        summary[model_name]["per_class_f1"] = per_class

        if folds[0].get("inference_ablation"):
            abl_summary: dict[str, dict] = {}
            for condition in ["both", "text_only", "eeg_only"]:
                cond_f1s = [
                    f["inference_ablation"][condition]["macro_f1"]
                    for f in folds if condition in f.get("inference_ablation", {})
                ]
                if cond_f1s:
                    abl_summary[condition] = {
                        "macro_f1_mean": float(np.mean(cond_f1s)),
                        "macro_f1_std": float(np.std(cond_f1s)),
                    }
            summary[model_name]["inference_ablation"] = abl_summary

        logger.info("%-15s  Acc: %.3f +/- %.3f  F1: %.3f +/- %.3f",
                    model_name, summary[model_name]["accuracy_mean"],
                    summary[model_name]["accuracy_std"],
                    summary[model_name]["macro_f1_mean"],
                    summary[model_name]["macro_f1_std"])

    stats: dict[str, dict] = {}
    if "full_gated" in model_names and "text_only" in model_names:
        gated_f1s = [f["test_macro_f1"] for f in all_fold_results["full_gated"]]
        text_f1s = [f["test_macro_f1"] for f in all_fold_results["text_only"]]
        if len(gated_f1s) == len(text_f1s) and len(gated_f1s) >= 2:
            t_stat, p_value = ttest_rel(gated_f1s, text_f1s)
            diffs = np.array(gated_f1s) - np.array(text_f1s)
            cohens_d = float(np.mean(diffs) / max(np.std(diffs, ddof=1), 1e-10))
            stats["gated_vs_text_only"] = {
                "t_statistic": float(t_stat), "p_value": float(p_value),
                "cohens_d": cohens_d, "significant": bool(p_value < 0.05),
            }

    if "full_concat" in model_names and "text_only" in model_names:
        concat_f1s = [f["test_macro_f1"] for f in all_fold_results["full_concat"]]
        text_f1s = [f["test_macro_f1"] for f in all_fold_results["text_only"]]
        if len(concat_f1s) == len(text_f1s) and len(concat_f1s) >= 2:
            t_stat, p_value = ttest_rel(concat_f1s, text_f1s)
            diffs = np.array(concat_f1s) - np.array(text_f1s)
            cohens_d = float(np.mean(diffs) / max(np.std(diffs, ddof=1), 1e-10))
            stats["concat_vs_text_only"] = {
                "t_statistic": float(t_stat), "p_value": float(p_value),
                "cohens_d": cohens_d, "significant": bool(p_value < 0.05),
            }

    gate_analysis: dict = {}
    if all_gate_values:
        all_gates = np.concatenate(all_gate_values, axis=0)
        gate_analysis = {
            "mean_text_gate": float(all_gates[:, 0].mean()),
            "std_text_gate": float(all_gates[:, 0].std()),
            "mean_eeg_gate": float(all_gates[:, 1].mean()),
            "std_eeg_gate": float(all_gates[:, 1].std()),
        }

    concept_importance: dict[str, list[tuple[str, float]]] = {}
    for model_name in model_names:
        if all_predictor_weights[model_name]:
            avg_weights = np.mean([np.abs(w) for w in all_predictor_weights[model_name]], axis=0)
            importance = avg_weights.sum(axis=0)
            n_concepts_model = importance.shape[0]
            names = (
                all_concept_names[:n_concepts_model]
                if n_concepts_model <= len(all_concept_names)
                else [(all_concept_names[i] if i < len(all_concept_names) else f"concept_{i}")
                      for i in range(n_concepts_model)]
            )
            ranked = sorted(zip(names, importance.tolist()), key=lambda x: x[1], reverse=True)
            concept_importance[model_name] = ranked[:20]

    _generate_cv_plots(summary, concept_importance, gate_analysis, stats, model_names)

    output = {
        "n_folds": n_folds, "num_epochs": num_epochs, "seed": seed,
        "model_summary": summary, "statistical_tests": stats,
        "gate_analysis": gate_analysis,
        "concept_importance": {k: [(name, float(score)) for name, score in v]
                               for k, v in concept_importance.items()},
    }

    CV_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CV_DIR / "cv_results.json"

    def _serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        return obj

    with open(save_path, "w") as f:
        json.dump(_serialize(output), f, indent=2, default=str)
    logger.info("All results saved to %s", save_path)

    return output


def _generate_cv_plots(summary, concept_importance, gate_analysis, stats, model_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plot_dir = CV_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    palette = sns.color_palette("muted")

    names = [n for n in model_names if n in summary]
    if names:
        f1_means = [summary[n]["macro_f1_mean"] for n in names]
        f1_stds = [summary[n]["macro_f1_std"] for n in names]
        acc_means = [summary[n]["accuracy_mean"] for n in names]
        acc_stds = [summary[n]["accuracy_std"] for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(names))
        display_names = [n.replace("_", "\n") for n in names]

        axes[0].bar(x, acc_means, yerr=acc_stds, capsize=5, color=palette[:len(names)], edgecolor="white")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(display_names, fontsize=9)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Model Comparison: Accuracy")
        axes[0].set_ylim(0, 1)

        axes[1].bar(x, f1_means, yerr=f1_stds, capsize=5, color=palette[:len(names)], edgecolor="white")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(display_names, fontsize=9)
        axes[1].set_ylabel("Macro F1")
        axes[1].set_title("Model Comparison: Macro F1")
        axes[1].set_ylim(0, 1)

        if "gated_vs_text_only" in stats and "text_only" in names and "full_gated" in names:
            p = stats["gated_vs_text_only"]["p_value"]
            sig_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            i1 = names.index("text_only")
            i2 = names.index("full_gated")
            y_max = max(f1_means[i1] + f1_stds[i1], f1_means[i2] + f1_stds[i2]) + 0.05
            axes[1].plot([i1, i1, i2, i2], [y_max, y_max + 0.02, y_max + 0.02, y_max], "k-", lw=1)
            axes[1].text((i1 + i2) / 2, y_max + 0.025, sig_str, ha="center", fontsize=8)

        fig.suptitle("5-Fold Cross-Validation Results", fontsize=14)
        fig.tight_layout()
        fig.savefig(plot_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    imp_model = (
        "full_gated" if "full_gated" in concept_importance
        else (list(concept_importance.keys())[0] if concept_importance else None)
    )
    if imp_model and concept_importance.get(imp_model):
        top_concepts = concept_importance[imp_model][:15]
        names_c = [c[0][:30] for c in top_concepts]
        scores_c = [c[1] for c in top_concepts]

        fig, ax = plt.subplots(figsize=(10, 7))
        y_pos = np.arange(len(names_c))
        ax.barh(y_pos, scores_c, color=palette[0], edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names_c, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |Weight| (across folds)")
        ax.set_title(f"Top Concept Importance ({imp_model})")
        fig.tight_layout()
        fig.savefig(plot_dir / "concept_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if gate_analysis:
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ["EEG", "Text"]
        means_g = [gate_analysis["mean_eeg_gate"], gate_analysis["mean_text_gate"]]
        stds_g = [gate_analysis["std_eeg_gate"], gate_analysis["std_text_gate"]]
        ax.bar(categories, means_g, yerr=stds_g, capsize=8,
               color=[palette[0], palette[1]], edgecolor="white", width=0.5)
        ax.set_ylabel("Gate Weight")
        ax.set_title("Gated Fusion: EEG vs Text Gate Weights")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(plot_dir / "gate_weights.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    abl_models = [n for n in model_names if summary.get(n, {}).get("inference_ablation")]
    if abl_models:
        fig, ax = plt.subplots(figsize=(10, 6))
        conditions = ["both", "text_only", "eeg_only"]
        x = np.arange(len(conditions))
        width = 0.8 / len(abl_models)
        for i, mn in enumerate(abl_models):
            abl = summary[mn]["inference_ablation"]
            means_a = [abl.get(c, {}).get("macro_f1_mean", 0) for c in conditions]
            stds_a = [abl.get(c, {}).get("macro_f1_std", 0) for c in conditions]
            offset = (i - len(abl_models) / 2 + 0.5) * width
            ax.bar(x + offset, means_a, width, yerr=stds_a, capsize=3,
                   label=mn, color=palette[i], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel("Macro F1")
        ax.set_title("Inference-Time Ablation")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(plot_dir / "inference_ablation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="5-Fold CV Experiments for RQ1-RQ3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()), default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS_CBL)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    CV_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(str(CV_DIR / "cv_experiments.log"))],
    )

    run_cv_experiments(model_names=args.models, num_epochs=args.epochs, n_folds=args.n_folds, seed=args.seed)
