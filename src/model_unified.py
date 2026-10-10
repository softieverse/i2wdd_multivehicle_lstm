"""
model_unified.py

LSTM encoder-decoder for imputing missing x/y points, one model for all datasets.

Input per step (7 features):
    [x_masked, y_masked, mask, x_linear, y_linear, x_hermite, y_hermite]
plus agent-type and dataset embeddings concatenated to every step.
Encoder: bidirectional LSTM (sees points before AND after a gap).
Decoder: bidirectional LSTM over encoder outputs, then a linear head.
Output: HERMITE interpolation + learned correction (head initialised to zero,
so training starts exactly at the Hermite baseline and anything learned on top
is a real gain over it).
"""

import torch
import torch.nn as nn

from common_schema import N_AGENTS, N_DATASETS

N_FEAT = 7


class UnifiedImputer(nn.Module):
    def __init__(self, hidden=128, layers=2, dropout=0.2, emb=8):
        super().__init__()
        self.agent_emb = nn.Embedding(N_AGENTS, emb)
        self.ds_emb = nn.Embedding(N_DATASETS, emb)
        d_in = N_FEAT + 2 * emb
        drop = dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(d_in, hidden, layers, batch_first=True,
                               bidirectional=True, dropout=drop)
        self.decoder = nn.LSTM(2 * hidden, hidden, 1, batch_first=True,
                               bidirectional=True)
        self.head = nn.Linear(2 * hidden, 2)
        nn.init.zeros_(self.head.weight)       # start exactly at Hermite
        nn.init.zeros_(self.head.bias)

    def forward(self, x, agent, ds):
        B, T, _ = x.shape
        e = torch.cat([self.agent_emb(agent), self.ds_emb(ds)], dim=-1)   # (B, 2*emb)
        x_in = torch.cat([x, e.unsqueeze(1).expand(B, T, -1)], dim=-1)
        enc, _ = self.encoder(x_in)
        dec, _ = self.decoder(enc)
        return x[:, :, 5:7] + self.head(dec)          # correction on Hermite


def edge_mask(obs):
    """1.0 for positions before the first / after the last observed point."""
    T = obs.shape[1]
    t = torch.arange(T, device=obs.device)[None, :]
    first = obs.argmax(1, keepdim=True)
    last = T - 1 - obs.flip(1).argmax(1, keepdim=True)
    return ((t < first) | (t > last)).float()


def masked_loss(pred, target, obs, w_int=1.0, w_edge=0.25):
    """Weighted MSE over MISSING positions. Interior and edge points get separate
    weights so large edge errors don't swamp the interior."""
    miss = 1.0 - obs
    edge = edge_mask(obs) * miss
    interior = miss - edge
    w = w_int * interior + w_edge * edge                       # (B, T)
    se = ((pred - target) ** 2).sum(-1)                        # (B, T)
    return (se * w).sum() / (w.sum() * 2).clamp(min=1e-6)


def masked_mse(pred, target, obs):
    return masked_loss(pred, target, obs, 1.0, 1.0)

def relative_loss(pred, target, obs, base, w_int=1.0, w_edge=0.25):
    """Per-window MSE on missing points, divided by the Hermite baseline's MSE on
    the same window (floored), so easy windows count as much as hard ones."""
    miss = 1.0 - obs
    edge = edge_mask(obs) * miss
    interior = miss - edge
    w = w_int * interior + w_edge * edge
    se_m = ((pred - target) ** 2).sum(-1)
    se_h = ((base - target) ** 2).sum(-1)
    cnt = (w.sum(1) * 2).clamp(min=1e-6)
    mse_m = (se_m * w).sum(1) / cnt
    mse_h = ((se_h * w).sum(1) / cnt).detach()
    eps = 0.1 * mse_h.mean() + 1e-8
    return (mse_m / (mse_h + eps)).mean()
