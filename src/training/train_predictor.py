import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import NUM_CLASSES, NUM_CONCEPTS


class SparseLinearPredictor(nn.Module):
    def __init__(
        self,
        num_concepts: int | None = None,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        num_concepts = num_concepts or NUM_CONCEPTS
        super().__init__()
        self.linear = nn.Linear(num_concepts, num_classes, bias=False)

    def forward(self, concept_activations: torch.Tensor) -> torch.Tensor:
        return self.linear(concept_activations)

    def l1_penalty(self) -> torch.Tensor:
        return self.linear.weight.abs().sum()

    def get_concept_importance(self) -> dict[str, float]:
        weights = self.linear.weight.detach().cpu()
        importance = weights.abs().sum(dim=0)
        return {
            f"concept_{i}": float(importance[i])
            for i in range(importance.size(0))
        }

    def get_weight_matrix(self) -> torch.Tensor:
        return self.linear.weight.detach().cpu()

    def sparsity_ratio(self, threshold: float = 1e-3) -> float:
        weights = self.linear.weight.detach().cpu()
        return float((weights.abs() < threshold).float().mean())
