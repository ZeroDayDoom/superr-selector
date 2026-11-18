"""Shared PyTorch training loop utilities (CPU-friendly, seeded)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 20260131
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(X: torch.Tensor, y: Dict[str, torch.Tensor], batch_size: int,
                shuffle: bool) -> DataLoader:
    tensors = [X] + [y[k] for k in y]
    ds = TensorDataset(*tensors)
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    epochs: int,
    lr: float = 1e-3,
) -> Dict[str, List[float]]:
    """Multi-task training: total loss = sum of named head losses."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in loaders["train"]:
            batch = [b.to(DEVICE) for b in batch]
            x, targets = batch[0], batch[1:]
            opt.zero_grad()
            loss = model.total_loss(x, targets)
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.size(0)
        history["train_loss"].append(running / len(loaders["train"].dataset))

        model.eval()
        running = 0.0
        with torch.no_grad():
            for batch in loaders["val"]:
                batch = [b.to(DEVICE) for b in batch]
                x, targets = batch[0], batch[1:]
                loss = model.total_loss(x, targets)
                running += float(loss.item()) * x.size(0)
        history["val_loss"].append(running / len(loaders["val"].dataset))
    return history


@torch.no_grad()
def predict_heads(model: nn.Module, X: torch.Tensor,
                  batch_size: int = 8192) -> Dict[str, np.ndarray]:
    model.eval()
    outs: Dict[str, List[np.ndarray]] = {}
    for i in range(0, len(X), batch_size):
        heads = model.forward_heads(X[i:i + batch_size].to(DEVICE))
        for k, v in heads.items():
            outs.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}
