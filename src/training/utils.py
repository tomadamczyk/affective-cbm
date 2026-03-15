import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def step(self, metric: float) -> bool:
        if self._is_improvement(metric):
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class WarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


def compute_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    num_classes: int = 3,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes))
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes))
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    if label_names is None:
        label_names = [str(i) for i in range(num_classes)]

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "per_class_f1": {name: float(v) for name, v in zip(label_names, per_f1)},
        "per_class_precision": {name: float(v) for name, v in zip(label_names, per_prec)},
        "per_class_recall": {name: float(v) for name, v in zip(label_names, per_rec)},
        "confusion_matrix": cm,
    }


def save_checkpoint(
    path: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    metrics: dict | None = None,
    **extra: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"epoch": epoch, "model_state_dict": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if metrics is not None:
        state["metrics"] = metrics
    state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    device: str | torch.device = "cpu",
) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state


def log_metrics(
    writer: SummaryWriter,
    metrics: dict[str, Any],
    step: int,
    prefix: str = "val",
) -> None:
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    writer.add_scalar(f"{prefix}/{key}/{sub_key}", sub_value, step)
        elif isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}", value, step)
