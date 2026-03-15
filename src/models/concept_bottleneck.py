import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import EMBEDDING_DIM, NUM_CONCEPTS, CONCEPT_CATEGORIES


def _build_concept_names(num_concepts: int | None = None) -> list[str]:
    n = num_concepts or NUM_CONCEPTS
    names: list[str] = []
    for category, concepts in CONCEPT_CATEGORIES.items():
        for concept in concepts:
            names.append(f"{category}/{concept}")
    while len(names) < n:
        names.append(f"latent/concept_{len(names)}")
    return names[:n]


class ConceptBottleneckLayer(nn.Module):
    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        num_concepts: int = NUM_CONCEPTS,
        concept_names: list[str] | None = None,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.input_dim = input_dim

        if concept_names is not None:
            self.concept_names = list(concept_names)
        else:
            self.concept_names = _build_concept_names(num_concepts)
        while len(self.concept_names) < num_concepts:
            self.concept_names.append(f"latent/concept_{len(self.concept_names)}")
        self.concept_names = self.concept_names[:num_concepts]

        self.activation = nn.ReLU()
        self.linear = nn.Linear(input_dim, num_concepts)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(h))

    def get_concept_weights(self) -> dict[str, torch.Tensor]:
        weights = self.linear.weight.detach()
        return {
            name: weights[i]
            for i, name in enumerate(self.concept_names)
        }

    def top_concepts(
        self,
        activations: torch.Tensor,
        k: int = 10,
    ) -> list[list[tuple[str, float]]]:
        results = []
        for sample in activations:
            topk = torch.topk(sample, k=min(k, self.num_concepts))
            results.append([
                (self.concept_names[idx], val.item())
                for val, idx in zip(topk.values, topk.indices)
            ])
        return results
