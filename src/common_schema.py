"""
common_schema.py

The one schema every dataset is converted into before it reaches the model.

  - 5 Hz, fixed windows of WINDOW steps
  - positions x, y per step (metres for HetroD / IDD-MM / I2WDD, pixels for IDD-PeD)
  - each window is centred on the mean of its OBSERVED points and divided by a
    per-dataset scale, so the network sees roughly unit-sized coordinates
  - mask: 1 = observed, 0 = missing (to be imputed)
  - agent type + dataset id are passed to the model as embeddings
"""

import numpy as np

HZ = 5.0
WINDOW = 30            # 6 s at 5 Hz (IDD-PeD tracks are short, so not longer)

AGENT = {"vehicle": 0, "two_wheeler": 1, "pedestrian": 2}
DATASET = {"hetrod": 0, "idd_mm": 1, "idd_ped": 2, "i2wdd": 3}
DATASET_NAMES = {v: k for k, v in DATASET.items()}
N_AGENTS, N_DATASETS = 3, 4

# coordinate scale used to normalise each dataset (same units as the data)
SCALE = {0: 50.0, 1: 50.0, 2: 100.0, 3: 50.0}
UNIT = {0: "m", 1: "m", 2: "px", 3: "m"}

# window stride per dataset (smaller datasets get denser windows)
STRIDE_BY_DS = {0: 15, 1: 5, 2: 10, 3: 5}

# gap settings (steps). Max gap = WINDOW // 2 so a gap never eats the whole window.
MIN_GAP, MAX_GAP = 3, WINDOW // 2


def agent_from_class(name: str) -> int:
    n = name.lower()
    if "ped" in n:
        return AGENT["pedestrian"]
    if "motor" in n or "bicycle" in n or "bike" in n or "cycl" in n or "two" in n:
        return AGENT["two_wheeler"]
    return AGENT["vehicle"]


def sample_mask(window: int, rng: np.random.Generator) -> np.ndarray:
    """Return obs mask (True = observed). Half random entries (rate 0.2-0.8,
    as in Sir's paper), half contiguous gaps (total missing 20-60%)."""
    obs = np.ones(window, dtype=bool)
    if rng.random() < 0.5:
        rate = rng.uniform(0.2, 0.8)
        obs = rng.random(window) >= rate
    else:
        target = int(window * rng.uniform(0.2, 0.6))
        tries = 0
        while (~obs).sum() < target and tries < 20:
            gl = int(rng.integers(MIN_GAP, MAX_GAP + 1))
            s = int(rng.integers(0, window - gl + 1))
            obs[s:s + gl] = False
            tries += 1
    if obs.sum() < 2:                      # always keep >= 2 anchors
        obs[rng.choice(window, 2, replace=False)] = True
    return obs


def linear_fill(xy: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Linear interpolation of x and y through observed points (constant at the ends)."""
    t = np.arange(len(xy))
    idx = np.where(obs)[0]
    return np.stack([np.interp(t, idx, xy[idx, k]) for k in range(2)], axis=1)

def hermite_fill(xy: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Cubic Hermite through observed points (finite-difference tangents),
    constant-velocity extrapolation at the window edges."""
    n = len(xy)
    idx = np.where(obs)[0]
    P = xy[idx]
    k = len(idx)
    m = np.zeros_like(P)
    slope = np.diff(P, axis=0) / np.diff(idx)[:, None]
    m[0], m[-1] = slope[0], slope[-1]
    if k > 2:
        m[1:-1] = (slope[:-1] + slope[1:]) / 2.0
    out = np.empty((n, 2))
    for j in range(k - 1):
        a, b = idx[j], idx[j + 1]
        h = float(b - a)
        t = (np.arange(a, b + 1) - a) / h
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        out[a:b + 1] = (h00[:, None] * P[j] + h10[:, None] * h * m[j]
                        + h01[:, None] * P[j + 1] + h11[:, None] * h * m[j + 1])
    first, last = idx[0], idx[-1]
    if first > 0:
        out[:first] = P[0] + (np.arange(first) - first)[:, None] * m[0]
    if last < n - 1:
        out[last + 1:] = P[-1] + (np.arange(last + 1, n) - last)[:, None] * m[-1]
    return out

def hermite_fill(xy: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Cubic Hermite through observed points (finite-difference tangents),
    constant-velocity extrapolation at the window edges. Vectorised."""
    n = len(xy)
    idx = np.flatnonzero(obs)
    P = xy[idx]
    k = len(idx)
    d = np.diff(idx)
    slope = np.diff(P, axis=0) / d[:, None]
    m = np.empty(P.shape, dtype=slope.dtype)
    m[0], m[-1] = slope[0], slope[-1]
    if k > 2:
        m[1:-1] = (slope[:-1] + slope[1:]) / 2.0
    tt = np.arange(n)
    j = np.clip(np.searchsorted(idx, tt, side='right') - 1, 0, k - 2)
    h = d[j].astype(float)
    t = (tt - idx[j]) / h
    t2, t3 = t * t, t * t * t
    h00 = (2 * t3 - 3 * t2 + 1)[:, None]
    h10 = (t3 - 2 * t2 + t)[:, None]
    h01 = (-2 * t3 + 3 * t2)[:, None]
    h11 = (t3 - t2)[:, None]
    hc = h[:, None]
    out = h00 * P[j] + h10 * hc * m[j] + h01 * P[j + 1] + h11 * hc * m[j + 1]
    first, last = idx[0], idx[-1]
    if first > 0:
        out[:first] = P[0] + (tt[:first] - first)[:, None] * m[0]
    if last < n - 1:
        out[last + 1:] = P[-1] + (tt[last + 1:] - last)[:, None] * m[-1]
    return out
