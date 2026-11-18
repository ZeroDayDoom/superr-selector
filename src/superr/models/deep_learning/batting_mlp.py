"""Multi-task MLP for delivery-outcome prediction (batting or bowling).

Architecture (spec): numeric normalisation + categorical embeddings ->
Dense 256 -> BatchNorm -> ReLU -> Dropout -> Dense 128 -> ReLU -> Dropout
-> Dense 64 -> ReLU -> heads.

Heads:
  - bucket logits (cross-entropy) over the mutually exclusive outcome buckets
  - wicket BCE head (batting: team wicket; bowling: bowler wicket)
  - bowling additionally: dot and boundary BCE heads
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from superr.models.deep_learning.trainer import DEVICE


class OutcomeMLP(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: List[int],
        buckets: List[str],
        bce_heads: Dict[str, float],   # name -> positive-class weight
        embedding_dim: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.buckets = buckets
        self.bce_heads = bce_heads
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, embedding_dim) for card in cat_cardinalities]
        )
        in_dim = n_numeric + embedding_dim * len(cat_cardinalities)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.bucket_head = nn.Linear(64, len(buckets))
        self.event_heads = nn.ModuleDict({
            name: nn.Linear(64, 1) for name in bce_heads
        })

    def trunk(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        embs = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
        z = torch.cat([x_num] + embs, dim=1)
        return self.net(z)

    def forward_heads(self, x_num: torch.Tensor, x_cat: torch.Tensor
                      ) -> Dict[str, torch.Tensor]:
        z = self.trunk(x_num, x_cat)
        out = {"bucket_logits": self.bucket_head(z)}
        for name, head in self.event_heads.items():
            out[f"event_{name}"] = head(z).squeeze(-1)
        return out

    def total_loss(self, x: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        # x = [x_num, x_cat, bucket_target, event targets...]
        x_num, x_cat, bucket_t = x[0], x[1], x[2]
        events = x[3:]
        heads = self.forward_heads(x_num, x_cat)
        loss = nn.functional.cross_entropy(heads["bucket_logits"], bucket_t)
        for i, (name, pos_weight) in enumerate(self.bce_heads.items()):
            pw = torch.tensor(pos_weight, device=DEVICE)
            loss = loss + nn.functional.binary_cross_entropy_with_logits(
                heads[f"event_{name}"], events[i].float(),
                pos_weight=pw if pos_weight > 0 else None,
            )
        return loss
