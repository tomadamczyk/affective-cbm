import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import EEG_INPUT_CHANNELS, N_EEG_BANDS, EMBEDDING_DIM, DROPOUT


class SentenceLevelEEG(nn.Module):
    def __init__(
        self,
        n_channels: int = EEG_INPUT_CHANNELS,
        n_bands: int = N_EEG_BANDS,
        output_dim: int = EMBEDDING_DIM,
        dropout: float = DROPOUT,
        F1: int = 16,
        D: int = 2,
        F2: int = 32,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands

        self.spectral_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, n_bands), padding=(0, n_bands // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 3), padding=(0, 1), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(F2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.spectral_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.projection(x)
        return x


class EEGEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int = EEG_INPUT_CHANNELS,
        n_bands: int = N_EEG_BANDS,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = DROPOUT,
        F1: int = 16,
        D: int = 2,
        F2: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.sentence_branch = SentenceLevelEEG(
            n_channels=n_channels, n_bands=n_bands, output_dim=embedding_dim,
            dropout=dropout, F1=F1, D=D, F2=F2,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.sentence_branch(x)
