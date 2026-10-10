"""
eval_unified.py

Fairer evaluation of the unified imputer.  For every dataset it compares

    model   : the trained LSTM encoder-decoder
    linear  : linear interpolation (flat at the window edges)
    hermite : cubic Hermite through the observed points (constant-velocity
              extrapolation at the edges)  <- the honest baseline

under controlled masks (same masks for all methods, fixed seed):
    random entries at missing rate 0.2 / 0.4 / 0.6 / 0.8   (Sir's protocol)
    one interior gap of 5 / 10 / 15 steps

and splits the error on missing points into
    interior : between two observed points
    edge     : before the first / after the last observed point
RMSE is Euclidean, in the dataset's own units (m or px).

Run from the project root:
    python src/eval_unified.py --ckpt checkpoints/unified_best.pt
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_schema import DATASET_NAMES, SCALE, UNIT, WINDOW, hermite_fill, linear_fill
from model_unified import UnifiedImputer
from unified_data import UnifiedWindows, load_all_tracks

RATES = (0.2, 0.4, 0.6, 0.8)
GAPS = (5, 10, 15)


# ---- masks ---------------------------------------------------------------------
def _ensure_two(obs, rng):
    for i in np.where(obs.sum(1) < 2)[0]:
        obs[i, rng.choice(WINDOW, 2, replace=False)] = True
    return obs


def random_obs(n, rate, rng):
    return _ensure_two(rng.random((n, WINDOW)) >= rate, rng)


def gap_obs(n, gap_len, rng):
    """One interior gap; first and last point stay observed."""
    obs = np.ones((n, WINDOW), bool)
    starts = rng.integers(1, WINDOW - gap_len, size=n)
    for i, s in enumerate(starts):
        obs[i, s:s + gap_len] = False
    return obs


# ---- run one scenario ----------------------------------------------------------
def predict_all(model, xy, ds, agent, obs, device):
    """Return per-point squared errors (native units^2) for model/linear/hermite."""
    N = len(xy)
    scale = np.array([SCALE[d] for d in ds], np.float32)[:, None, None]
    of = obs[:, :, None].astype(np.float32)
    center = (xy * of).sum(1, keepdims=True) / of.sum(1, keepdims=True)
    xyn = (xy - center) / scale
    lin = np.stack([linear_fill(xyn[i], obs[i]) for i in range(N)])
    her = np.stack([hermite_fill(xyn[i], obs[i]) for i in range(N)])

    x = np.zeros((N, WINDOW, 7), np.float32)
    x[:, :, 0:2] = xyn * of
    x[:, :, 2] = obs
    x[:, :, 3:5] = lin
    x[:, :, 5:7] = her
    preds = []
    with torch.no_grad():
        for s in range(0, N, 512):
            e = slice(s, s + 512)
            preds.append(model(torch.from_numpy(x[e]).to(device),
                               torch.from_numpy(agent[e]).long().to(device),
                               torch.from_numpy(ds[e]).long().to(device)).cpu().numpy())
    pred = np.concatenate(preds)

    sc2 = scale[:, :, 0] ** 2
    return {name: ((arr - xyn) ** 2).sum(-1) * sc2
            for name, arr in (("model", pred), ("linear", lin), ("hermite", her))}


def split_masks(obs):
    """interior / edge masks over the MISSING points."""
    t = np.arange(WINDOW)[None, :]
    first = obs.argmax(1)[:, None]
    last = (WINDOW - 1 - obs[:, ::-1].argmax(1))[:, None]
    miss = ~obs
    edge = miss & ((t < first) | (t > last))
    return miss, miss & ~edge, edge


def rmse(err, mask):
    n = mask.sum()
    return float(np.sqrt((err * mask).sum() / n)) if n else float("nan")


# ---- main ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--max-per-ds", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UnifiedImputer(hidden=args.hidden).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    data = UnifiedWindows(load_all_tracks(), args.split)
    rng = np.random.default_rng(args.seed)
    rows = []

    for d in sorted(np.unique(data.ds)):
        idx = np.where(data.ds == d)[0]
        if len(idx) > args.max_per_ds:
            idx = rng.choice(idx, args.max_per_ds, replace=False)
        xy, ds, agent = data.xy[idx], data.ds[idx], data.agent[idx]
        name, unit = DATASET_NAMES[int(d)], UNIT[int(d)]
        print(f"\n=== {name} ({len(idx)} windows, RMSE in {unit}) ===")
        print(f"{'scenario':12s} {'subset':9s} {'model':>9s} {'linear':>9s} {'hermite':>9s}")

        scenarios = [(f"rand {r}", random_obs(len(idx), r, rng)) for r in RATES]
        scenarios += [(f"gap {g}", gap_obs(len(idx), g, rng)) for g in GAPS]
        for sname, obs in scenarios:
            errs = predict_all(model, xy, ds, agent, obs, device)
            miss, interior, edge = split_masks(obs)
            for sub, mk in (("all", miss), ("interior", interior), ("edge", edge)):
                if mk.sum() == 0:
                    continue
                r = {k: rmse(v, mk) for k, v in errs.items()}
                print(f"{sname:12s} {sub:9s} {r['model']:9.3f} {r['linear']:9.3f} {r['hermite']:9.3f}")
                rows.append([name, unit, sname, sub, int(mk.sum()),
                             r["model"], r["linear"], r["hermite"]])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "unit", "scenario", "subset", "n_missing",
                    "model", "linear", "hermite"])
        w.writerows(rows)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
