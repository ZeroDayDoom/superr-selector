"""GRU sequence model: last-K deliveries (strictly before the target ball).

Sequence features per step: runs, wicket, legal, extras flag, over number,
pressure score. Static context: state + batter/bowler class features.
The sequence ENDS before the target delivery — the target delivery's row is
never included.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from superr.models.deep_learning.trainer import DEVICE

SEQ_FEATURES = ["_seq_runs", "_seq_wicket", "_seq_legal", "_seq_extras",
                "_seq_over", "_seq_pressure"]
SEQ_LENGTH = 12


def _lag_prev(df: pd.DataFrame, col: str, fill: float = 0.0) -> pd.Series:
    """Previous-delivery value within (match, innings), as plain float64.

    Coerces to float64 BEFORE the groupby shift so nullable/boolean dtypes
    never reach fillna (masked-array fillna on boolean dtype raises for
    numeric fill values); the shift itself guarantees the current delivery
    never enters its own sequence.
    """
    num = pd.to_numeric(df[col], errors="coerce").astype(float)
    return (num.groupby([df["match_id"], df["innings"]])
            .shift(1).fillna(fill))


def build_sequences(df: pd.DataFrame, seq_length: int = SEQ_LENGTH) -> np.ndarray:
    """Per-delivery sequence of the previous K deliveries of the innings.

    Rows earlier than the sequence length are zero-padded. Uses groupby
    shift so the current delivery NEVER enters its own sequence.
    """
    seq_cols = {}
    seq_cols["_seq_runs"] = _lag_prev(df, "total_runs") / 6.0
    seq_cols["_seq_wicket"] = _lag_prev(df, "is_team_wicket")
    seq_cols["_seq_legal"] = _lag_prev(df, "is_legal_ball")
    seq_cols["_seq_extras"] = _lag_prev(
        df, "is_extras_ball") if "is_extras_ball" in df.columns else (
        df["extras_type"].notna().astype(float).groupby(
            [df["match_id"], df["innings"]]).shift(1).fillna(0.0))
    seq_cols["_seq_over"] = (df["legal_balls_before"] / 120.0).clip(0, 1)
    seq_cols["_seq_pressure"] = pd.to_numeric(
        df["PRESSURE_MODEL_C"], errors="coerce").fillna(0.0)
    S = np.stack([seq_cols[c].values for c in SEQ_FEATURES], axis=1)

    n = len(df)
    out = np.zeros((n, seq_length, len(SEQ_FEATURES)), dtype=np.float32)
    idx = np.arange(n)
    pos = df.groupby(["match_id", "innings"], sort=False).cumcount().values
    for k in range(1, seq_length + 1):
        # S is already shifted by one (S[i] = previous delivery); step k of
        # the sequence therefore reads S at (i - (k - 1))
        valid = pos - (k - 1) >= 1
        out[valid, k - 1] = S[idx[valid] - (k - 1)]
    return out


class OutcomeGRU(nn.Module):
    def __init__(self, seq_dim: int, n_static: int,
                 buckets: List[str], hidden_dim: int = 64, layers: int = 1):
        super().__init__()
        self.buckets = buckets
        self.gru = nn.GRU(seq_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.static_net = nn.Sequential(nn.Linear(n_static, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 64), nn.ReLU(),
            nn.Linear(64, len(buckets)),
        )

    def forward_heads(self, x_seq: torch.Tensor, x_static: torch.Tensor
                      ) -> Dict[str, torch.Tensor]:
        _, h = self.gru(x_seq)
        z = torch.cat([h[-1], self.static_net(x_static)], dim=1)
        return {"bucket_logits": self.head(z)}

    def total_loss(self, x: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        x_seq, x_static, bucket_t = x[0], x[1], x[2]
        heads = self.forward_heads(x_seq, x_static)
        return nn.functional.cross_entropy(heads["bucket_logits"], bucket_t)
