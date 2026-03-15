import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    EMBEDDING_DIM,
    N_NEURAL_CONCEPTS,
    NUM_CLASSES,
    NUM_CONCEPTS,
)
from src.models.eeg_encoder import EEGEncoder
from src.models.text_encoder import TextEncoder
from src.models.concept_bottleneck import ConceptBottleneckLayer


class ConceptLevelGating(nn.Module):
    def __init__(self, n_text_concepts: int, n_neural_concepts: int) -> None:
        super().__init__()
        total = n_text_concepts + n_neural_concepts
        self.gate_network = nn.Sequential(
            nn.Linear(total, max(total // 2, 4)),
            nn.ReLU(),
            nn.Linear(max(total // 2, 4), 2),
            nn.Softmax(dim=-1),
        )
        self.n_text = n_text_concepts
        self.n_neural = n_neural_concepts

    def forward(
        self, text_concepts: torch.Tensor, neural_concepts: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([text_concepts, neural_concepts], dim=-1)
        gate = self.gate_network(combined)
        g_text = gate[:, 0:1]
        g_neural = gate[:, 1:2]
        self._last_gate = gate.detach()
        return torch.cat([
            text_concepts * g_text,
            neural_concepts * g_neural,
        ], dim=-1)


class ConceptCrossModulation(nn.Module):
    def __init__(self, n_text_concepts: int, n_neural_concepts: int) -> None:
        super().__init__()
        self.neural_to_text = nn.Linear(n_neural_concepts, n_text_concepts)
        self.text_to_neural = nn.Linear(n_text_concepts, n_neural_concepts)
        self.sigmoid = nn.Sigmoid()
        self.n_text = n_text_concepts
        self.n_neural = n_neural_concepts

    def forward(
        self, text_concepts: torch.Tensor, neural_concepts: torch.Tensor,
    ) -> torch.Tensor:
        text_gate = self.sigmoid(self.neural_to_text(neural_concepts))
        neural_gate = self.sigmoid(self.text_to_neural(text_concepts))
        return torch.cat([
            text_concepts * text_gate,
            neural_concepts * neural_gate,
        ], dim=-1)


class CBLModel(nn.Module):
    def __init__(
        self,
        fusion_mode: str = "concat",
        num_concepts: int | None = None,
        n_neural_concepts: int = 0,
        use_temporal: bool = False,
    ) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode
        self.num_concepts = num_concepts or NUM_CONCEPTS
        self.n_neural_concepts = n_neural_concepts
        self.total_concepts = self.num_concepts + n_neural_concepts

        self.eeg_encoder = EEGEncoder()
        self.text_encoder = TextEncoder(freeze_backbone=True)

        self.text_cbl = ConceptBottleneckLayer(
            input_dim=EMBEDDING_DIM,
            num_concepts=self.num_concepts,
        )

        if n_neural_concepts > 0:
            from config import NEURAL_CONCEPT_NAMES
            self.neural_cbl = ConceptBottleneckLayer(
                input_dim=EMBEDDING_DIM,
                num_concepts=n_neural_concepts,
                concept_names=list(NEURAL_CONCEPT_NAMES),
            )
        else:
            self.neural_cbl = None

        if n_neural_concepts > 0 and fusion_mode == "gated":
            self.concept_fusion = ConceptLevelGating(
                self.num_concepts, n_neural_concepts,
            )
        elif n_neural_concepts > 0 and fusion_mode == "cross_modulation":
            self.concept_fusion = ConceptCrossModulation(
                self.num_concepts, n_neural_concepts,
            )
        else:
            self.concept_fusion = None

        self.prediction_head = nn.Linear(self.total_concepts, NUM_CLASSES)

    def _fuse_concepts(
        self, text_concepts: torch.Tensor, neural_concepts: torch.Tensor | None,
    ) -> torch.Tensor:
        if neural_concepts is None:
            return text_concepts
        if self.concept_fusion is not None:
            return self.concept_fusion(text_concepts, neural_concepts)
        return torch.cat([text_concepts, neural_concepts], dim=-1)

    def forward(
        self,
        eeg_input: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temporal_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eeg_emb = self.eeg_encoder(eeg_input)
        text_emb = self.text_encoder(input_ids, attention_mask)

        text_concepts = self.text_cbl(text_emb)
        neural_concepts = self.neural_cbl(eeg_emb) if self.neural_cbl is not None else None

        concept_activations = self._fuse_concepts(text_concepts, neural_concepts)
        logits = self.prediction_head(concept_activations)
        return concept_activations, logits

    def forward_with_ablation(
        self,
        eeg_input: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        temporal_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device

        if input_ids is not None and attention_mask is not None:
            text_emb = self.text_encoder(input_ids, attention_mask)
            text_concepts = self.text_cbl(text_emb)
        else:
            batch_size = eeg_input.size(0) if eeg_input is not None else 1
            text_concepts = torch.zeros(batch_size, self.num_concepts, device=device)

        if self.neural_cbl is not None:
            if eeg_input is not None:
                eeg_emb = self.eeg_encoder(eeg_input)
                neural_concepts = self.neural_cbl(eeg_emb)
            else:
                batch_size = input_ids.size(0) if input_ids is not None else 1
                neural_concepts = torch.zeros(batch_size, self.n_neural_concepts, device=device)
        else:
            neural_concepts = None

        concept_activations = self._fuse_concepts(text_concepts, neural_concepts)
        logits = self.prediction_head(concept_activations)
        return concept_activations, logits

    def get_gate_values(self) -> torch.Tensor | None:
        if self.fusion_mode == "gated" and self.concept_fusion is not None:
            if hasattr(self.concept_fusion, '_last_gate'):
                return self.concept_fusion._last_gate
        return None
