import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer

try:
    from config import (
        BATCH_SIZE, DEVICE, N_EEG_BANDS, PROCESSED_DATA_DIR, RANDOM_SEED,
        TEST_SPLIT, TEXT_MODEL_NAME, TRAIN_SPLIT, VAL_SPLIT, ZUCO_N_CHANNELS,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config import (
        BATCH_SIZE, DEVICE, N_EEG_BANDS, PROCESSED_DATA_DIR, RANDOM_SEED,
        TEST_SPLIT, TEXT_MODEL_NAME, TRAIN_SPLIT, VAL_SPLIT, ZUCO_N_CHANNELS,
    )

logger = logging.getLogger(__name__)


class ZuCoDataset(Dataset):
    def __init__(
        self,
        pt_path: Optional[Path] = None,
        tokenizer_name: str = TEXT_MODEL_NAME,
        max_length: int = 128,
    ) -> None:
        if pt_path is None:
            pt_path = PROCESSED_DATA_DIR / "zuco_TSR.pt"
        self.pt_path = Path(pt_path)

        data = torch.load(self.pt_path, map_location="cpu", weights_only=False)

        self.texts: List[str] = data["texts"]
        self.eeg_features: torch.Tensor = data["eeg_features"]
        self.labels: torch.Tensor = data["labels"]

        _, n_ch, n_b = self.eeg_features.shape
        if n_ch != ZUCO_N_CHANNELS:
            self.eeg_features = self._fix_channels(self.eeg_features)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        encoded = self.tokenizer(
            self.texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        self.input_ids: torch.Tensor = encoded["input_ids"]
        self.attention_mask: torch.Tensor = encoded["attention_mask"]

    @staticmethod
    def _fix_channels(tensor: torch.Tensor) -> torch.Tensor:
        _, n_ch, n_b = tensor.shape
        if n_ch < ZUCO_N_CHANNELS:
            pad = torch.zeros(tensor.shape[0], ZUCO_N_CHANNELS - n_ch, n_b)
            tensor = torch.cat([tensor, pad], dim=1)
        elif n_ch > ZUCO_N_CHANNELS:
            tensor = tensor[:, :ZUCO_N_CHANNELS, :]
        return tensor

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "eeg_features": self.eeg_features[idx],
            "label": self.labels[idx],
            "text": self.texts[idx],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "eeg_features": torch.stack([b["eeg_features"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "text": [b["text"] for b in batch],
    }


def get_splits(
    pt_path: Optional[Path] = None,
    tokenizer_name: str = TEXT_MODEL_NAME,
    max_length: int = 128,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    seed: int = RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = ZuCoDataset(pt_path=pt_path, tokenizer_name=tokenizer_name, max_length=max_length)

    n = len(dataset)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    train_ds = Subset(dataset, indices[:n_train])
    val_ds = Subset(dataset, indices[n_train: n_train + n_val])
    test_ds = Subset(dataset, indices[n_train + n_val:])

    shared_kwargs = dict(
        batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=(DEVICE != "cpu"),
    )

    return (
        DataLoader(train_ds, shuffle=True, **shared_kwargs),
        DataLoader(val_ds, shuffle=False, **shared_kwargs),
        DataLoader(test_ds, shuffle=False, **shared_kwargs),
    )
