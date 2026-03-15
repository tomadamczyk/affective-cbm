import torch
import torch.nn as nn
from transformers import RobertaModel

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import TEXT_MODEL_NAME, EMBEDDING_DIM, DROPOUT


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = TEXT_MODEL_NAME,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = DROPOUT,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = RobertaModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        embedding = self.projection(cls_hidden)
        return embedding
