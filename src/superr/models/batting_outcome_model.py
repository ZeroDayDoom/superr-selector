"""Batting outcome model: deep-learning training glue + orchestration helpers.

Contains the tensor preparation, MLP/GRU training and prediction functions
used by the Phase-6 runner for BOTH batting and bowling multi-task models.
The tabular models live in expected_runs / expected_wicket /
bowling_outcome_model; this module is the PyTorch counterpart.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from superr.models.deep_learning.batting_mlp import OutcomeMLP
from superr.models.deep_learning.sequence_model import (
    SEQ_FEATURES,
    SEQ_LENGTH,
    OutcomeGRU,
    build_sequences,
)
from superr.models.deep_learning.trainer import DEVICE, set_seed

CAT_CODES = {}  # populated per fit: column -> {level: code}


def _cat_codes(train: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict]:
    return {c: {lvl: i for i, lvl in enumerate(sorted(train[c].astype(str).unique()))}
            for c in cat_cols}


def _encode_cats(df: pd.DataFrame, codes: Dict[str, Dict]) -> np.ndarray:
    cols = []
    for c, mapping in codes.items():
        cols.append(df[c].astype(str).map(mapping).fillna(0).astype(int).values)
    return np.stack(cols, axis=1) if cols else np.zeros((len(df), 0), dtype=np.int64)


def _num_matrix(df: pd.DataFrame, num_cols: List[str],
                means: pd.Series, stds: pd.Series) -> np.ndarray:
    X = df[num_cols].astype(float).copy()
    X = X.fillna(means)
    return ((X - means) / stds.replace(0, 1)).values.astype(np.float32)


def prepare_deep_data(
    train: pd.DataFrame, eval_frames: Dict[str, pd.DataFrame],
    bucket_col: str, buckets: List[str], event_defs: Dict[str, str],
    num_cols: List[str], cat_cols: List[str],
) -> Tuple[Dict, Dict[str, Dict]]:
    """Return tensors + preprocessing artifacts for MLP training."""
    codes = _cat_codes(train, cat_cols)
    num_means = train[num_cols].astype(float).mean()
    num_stds = train[num_cols].astype(float).std().fillna(1.0)
    bucket_index = {b: i for i, b in enumerate(buckets)}

    def tensors(frame: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_num = torch.tensor(_num_matrix(frame, num_cols, num_means, num_stds))
        x_cat = torch.tensor(_encode_cats(frame, codes))
        bt = torch.tensor(frame[bucket_col].map(bucket_index).fillna(0).astype(int).values)
        return x_num, x_cat, bt

    def _event(frame: pd.DataFrame, col: str) -> torch.Tensor:
        v = frame[col].astype("boolean").fillna(False).astype(bool).values
        return torch.tensor(v, dtype=torch.float32)

    train_t = tensors(train)
    y_train = {"bucket": train_t[2]}
    for name, col in event_defs.items():
        y_train[name] = _event(train, col)
    eval_t = {}
    for split, frame in eval_frames.items():
        xt = tensors(frame)
        yv = {"bucket": xt[2]}
        for name, col in event_defs.items():
            yv[name] = _event(frame, col)
        eval_t[split] = (xt, yv)

    artifacts = {
        "codes": codes, "num_means": num_means, "num_stds": num_stds,
        "bucket_index": bucket_index,
    }
    return {"train": train_t, "y_train": y_train, "eval": eval_t}, artifacts


class DeepDataset(torch.utils.data.Dataset):
    def __init__(self, x_num, x_cat, y: Dict[str, torch.Tensor]):
        self.x_num, self.x_cat = x_num, x_cat
        self.keys = list(y.keys())
        self.tensors = [y[k] for k in self.keys]

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, i):
        return (self.x_num[i], self.x_cat[i]) + tuple(t[i] for t in self.tensors)


def train_deep_mlp(
    data: Dict, artifacts: Dict, buckets: List[str],
    event_defs: Dict[str, str], event_pos_weights: Dict[str, float],
    num_cols: List[str], cat_cols: List[str],
    epochs: int = 12, batch_size: int = 4096, lr: float = 1e-3,
) -> Tuple[OutcomeMLP, Dict[str, List[float]]]:
    """Train the multi-task MLP; returns (model, history)."""
    set_seed()
    x_num, x_cat, bt = data["train"]
    y = data["y_train"]
    train_ds = DeepDataset(x_num, x_cat, y)
    g = torch.Generator().manual_seed(20260131)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                               shuffle=True, generator=g)
    model = OutcomeMLP(
        n_numeric=len(num_cols),
        cat_cardinalities=[len(artifacts["codes"][c]) for c in cat_cols],
        buckets=buckets, bce_heads=event_pos_weights,
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}
    val_pack = data["eval"]["validation"]
    vx_num, vx_cat, vbt = val_pack[0]
    vy = val_pack[1]
    val_ds = DeepDataset(vx_num, vx_cat, vy)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    bce = None  # (BCEWithLogitsLoss is applied inline with pos_weight below)
    for epoch in range(epochs):
        model.train()
        running, nb = 0.0, 0
        for batch in train_loader:
            batch = [b.to(DEVICE) for b in batch]
            x_num_b, x_cat_b = batch[0], batch[1]
            targets = batch[2:]
            opt.zero_grad()
            heads = model.forward_heads(x_num_b, x_cat_b)
            loss = torch.nn.functional.cross_entropy(
                heads["bucket_logits"], targets[0])
            for i, (name, pw) in enumerate(event_pos_weights.items()):
                weight = torch.tensor(pw, device=DEVICE) if pw > 0 else None
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                    heads[f"event_{name}"], targets[1 + i].float(), pos_weight=weight)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(x_num_b)
            nb += len(x_num_b)
        history["train_loss"].append(running / max(1, nb))

        model.eval()
        running, nb = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(DEVICE) for b in batch]
                x_num_b, x_cat_b = batch[0], batch[1]
                targets = batch[2:]
                heads = model.forward_heads(x_num_b, x_cat_b)
                loss = torch.nn.functional.cross_entropy(
                    heads["bucket_logits"], targets[0])
                for i, (name, pw) in enumerate(event_pos_weights.items()):
                    weight = torch.tensor(pw, device=DEVICE) if pw > 0 else None
                    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                        heads[f"event_{name}"], targets[1 + i].float(),
                        pos_weight=weight)
                running += float(loss.item()) * len(x_num_b)
                nb += len(x_num_b)
        history["val_loss"].append(running / max(1, nb))
    return model, history


@torch.no_grad()
def deep_bucket_probs(model, frame: pd.DataFrame, artifacts: Dict,
                      num_cols: List[str], cat_cols: List[str],
                      buckets: List[str]) -> pd.DataFrame:
    """Bucket probabilities from a trained MLP for an eval frame."""
    X = _num_matrix(frame, num_cols, artifacts["num_means"], artifacts["num_stds"])
    C = _encode_cats(frame, artifacts["codes"])
    x_num = torch.tensor(X)
    x_cat = torch.tensor(C)
    heads = predict_heads_mlp(model, x_num, x_cat)
    logits = heads["bucket_logits"]
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    return pd.DataFrame(probs, columns=buckets, index=frame.index)


@torch.no_grad()
def predict_heads_mlp(model, x_num: torch.Tensor, x_cat: torch.Tensor,
                      batch_size: int = 8192) -> Dict[str, np.ndarray]:
    model.eval()
    out: Dict[str, List[np.ndarray]] = {}
    for i in range(0, len(x_num), batch_size):
        heads = model.forward_heads(x_num[i:i + batch_size].to(DEVICE),
                                    x_cat[i:i + batch_size].to(DEVICE))
        for k, v in heads.items():
            out.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def train_deep_gru(
    df_train: pd.DataFrame, df_val: pd.DataFrame, buckets: List[str],
    bucket_col: str, static_cols: List[str], seq_length: int = SEQ_LENGTH,
    epochs: int = 8, batch_size: int = 4096, lr: float = 1e-3,
) -> Tuple[OutcomeGRU, Dict[str, List[float]], Dict]:
    """Train the GRU sequence model on the previous-K delivery sequence."""
    set_seed()
    seq_tr = build_sequences(df_train, seq_length)
    seq_va = build_sequences(df_val, seq_length)
    means = df_train[static_cols].mean()
    stds = df_train[static_cols].std().fillna(1.0)
    st_tr = ((df_train[static_cols].fillna(means) - means) / stds.replace(0, 1)).values.astype(np.float32)
    st_va = ((df_val[static_cols].fillna(means) - means) / stds.replace(0, 1)).values.astype(np.float32)
    bidx = {b: i for i, b in enumerate(buckets)}
    y_tr = torch.tensor(df_train[bucket_col].map(bidx).fillna(0).astype(int).values)
    y_va = torch.tensor(df_val[bucket_col].map(bidx).fillna(0).astype(int).values)

    model = OutcomeGRU(seq_dim=len(SEQ_FEATURES),
                       n_static=len(static_cols), buckets=buckets).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    def run_epoch(frames, train: bool) -> float:
        total, nb = 0.0, 0
        for i in range(0, len(frames[0]), batch_size):
            xs = torch.tensor(frames[0][i:i + batch_size])
            xs_st = torch.tensor(frames[1][i:i + batch_size])
            yt = frames[2][i:i + batch_size].to(DEVICE)
            if train:
                opt.zero_grad()
            heads = model.forward_heads(xs.to(DEVICE), xs_st.to(DEVICE))
            loss = torch.nn.functional.cross_entropy(heads["bucket_logits"], yt)
            if train:
                loss.backward()
                opt.step()
            total += float(loss.item()) * len(xs)
            nb += len(xs)
        return total / max(1, nb)

    for epoch in range(epochs):
        model.train()
        history["train_loss"].append(run_epoch((seq_tr, st_tr, y_tr), True))
        model.eval()
        with torch.no_grad():
            history["val_loss"].append(run_epoch((seq_va, st_va, y_va), False))

    artifacts = {"static_means": means, "static_stds": stds, "bucket_index": bidx}
    return model, history, artifacts


@torch.no_grad()
def gru_bucket_probs(model, frame: pd.DataFrame, artifacts: Dict,
                     static_cols: List[str], buckets: List[str],
                     seq_length: int = SEQ_LENGTH) -> pd.DataFrame:
    seq = build_sequences(frame, seq_length)
    st = ((frame[static_cols].fillna(artifacts["static_means"])
           - artifacts["static_means"]) / artifacts["static_stds"].replace(0, 1)
          ).values.astype(np.float32)
    logits = []
    for i in range(0, len(frame), 8192):
        heads = model.forward_heads(torch.tensor(seq[i:i + 8192]).to(DEVICE),
                                    torch.tensor(st[i:i + 8192]).to(DEVICE))
        logits.append(heads["bucket_logits"].cpu().numpy())
    probs = torch.softmax(torch.tensor(np.concatenate(logits)), dim=1).numpy()
    return pd.DataFrame(probs, columns=buckets, index=frame.index)
