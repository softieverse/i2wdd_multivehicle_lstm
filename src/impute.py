"""
impute.py -- fill missing x/y points in NEW trajectories with the trained UnifiedImputer.

Input  : DataFrame/CSV with columns  (x, y)  or  (lat, lon),  optional time column  t  (seconds).
         Missing points = NaN/blank cells, or timestamps that are simply absent.
Output : 5 Hz track, observed points untouched, gaps filled, `imputed` flag column.

Must mirror unified_data.py exactly: centre each 30-step window on its observed points,
divide by SCALE[ds], 7 features = [x_masked, y_masked, mask, lin_x, lin_y, herm_x, herm_y].
"""
import numpy as np
import pandas as pd

from common_schema import HZ, SCALE, UNIT, WINDOW, hermite_fill, linear_fill

# label -> (dataset id, agent id)   (ids must match common_schema.py / the checkpoint)
PROFILES = {
    "Car - HetroD style (metres)": (0, 0),
    "Car - IDD-Multimodal style (metres)": (1, 0),
    "Pedestrian - IDD-PeD style (pixels)": (2, 2),
    "Two-wheeler - I2WDD style (metres)": (3, 1),
}
STRIDE = WINDOW // 2          # windows overlap by 50 %
MAX_GAP = WINDOW // 2         # 15 steps = 3 s, longest contiguous gap seen in training


# ------------------------------------------------------------------ model
def load_model(ckpt, device="cpu"):
    import torch
    from model_unified import UnifiedImputer
    m = UnifiedImputer(hidden=128)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.to(device).eval()


def make_predictor(model, ds, agent, device="cpu", bs=256):
    """Returns f(X (n,30,7) float32) -> (n,30,2) normalised predictions."""
    import torch

    def predict(X):
        outs = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs], dtype=np.float32)).to(device)
            n = len(xb)
            a = torch.full((n,), agent, dtype=torch.long, device=device)
            d = torch.full((n,), ds, dtype=torch.long, device=device)
            with torch.no_grad():
                outs.append(model(xb, a, d).cpu().numpy())
        return np.concatenate(outs)
    return predict


# ------------------------------------------------------------------ input handling
def _lower(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def prepare(df, in_hz=5.0):
    """-> grid (N,2) on a 5 Hz grid (NaN = missing), t0, geo (None or (lat0, lon0))."""
    df = _lower(df)
    geo = None
    if {"x", "y"} <= set(df.columns):
        xy = df[["x", "y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    else:
        la = next((c for c in ("lat", "latitude") if c in df.columns), None)
        lo = next((c for c in ("lon", "lng", "longitude") if c in df.columns), None)
        if la is None or lo is None:
            raise ValueError("Need columns x,y  or  lat,lon (header row required).")
        lat = pd.to_numeric(df[la], errors="coerce").to_numpy(float)
        lon = pd.to_numeric(df[lo], errors="coerce").to_numpy(float)
        ok = np.isfinite(lat) & np.isfinite(lon)
        if ok.sum() < 2:
            raise ValueError("Fewer than 2 valid lat/lon rows.")
        lat0, lon0 = lat[ok][0], lon[ok][0]
        geo = (lat0, lon0)
        xy = np.stack([(lon - lon0) * 111_320.0 * np.cos(np.deg2rad(lat0)),
                       (lat - lat0) * 110_540.0], 1)       # same projection as training
    tcol = next((c for c in ("t", "time", "timestamp", "seconds") if c in df.columns), None)
    if tcol is not None:
        t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
        keep = np.isfinite(t)
        t, xy = t[keep], xy[keep]
        order = np.argsort(t, kind="stable")
        t, xy = t[order], xy[order]
    else:
        t = np.arange(len(xy)) / float(in_hz)
    if len(t) < 2:
        raise ValueError("Not enough rows.")
    t0 = float(t[0])
    idx = np.round((t - t0) * HZ).astype(int)               # nearest 5 Hz slot
    N = int(idx.max()) + 1
    grid = np.full((N, 2), np.nan)
    fin = np.isfinite(xy).all(1)
    idx_f, xy_f = idx[fin], xy[fin]
    if len(idx_f):
        _, first = np.unique(idx_f, return_index=True)       # first sample wins per slot
        grid[idx_f[first]] = xy_f[first]
    return grid, t0, geo


def _longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# ------------------------------------------------------------------ core
def impute_grid(grid, predict, ds):
    """grid (N,2), NaN = missing.  Returns filled, hermite, linear (all (N,2)) and info dict."""
    N = len(grid)
    obs = np.isfinite(grid).all(1)
    if N < WINDOW:
        raise ValueError(f"Track too short: need >= {WINDOW} slots at {HZ:g} Hz "
                         f"({WINDOW / HZ:g} s), got {N}.")
    if obs.sum() < 2:
        raise ValueError("Need at least 2 observed points.")
    scale = SCALE[ds]

    herm = hermite_fill(grid, obs)            # whole-track baselines (only observed rows are read)
    lin = linear_fill(grid, obs)

    starts = list(range(0, N - WINDOW + 1, STRIDE))
    if starts[-1] != N - WINDOW:
        starts.append(N - WINDOW)
    X, meta = [], []
    for s in starts:
        o = obs[s:s + WINDOW]
        if o.sum() < 2 or o.all():            # nothing to impute, or no anchors
            continue
        w = np.where(o[:, None], grid[s:s + WINDOW], 0.0)
        c = w[o].mean(0)
        xyn = (w - c) / scale
        x = np.zeros((WINDOW, 7), np.float32)
        x[:, 0:2] = xyn * o[:, None]
        x[:, 2] = o
        x[:, 3:5] = linear_fill(xyn, o)
        x[:, 5:7] = hermite_fill(xyn, o)
        X.append(x)
        meta.append((s, c))

    acc = np.zeros((N, 2))
    wsum = np.zeros(N)
    if X:
        P = predict(np.stack(X))
        wt = 1.0 + np.minimum(np.arange(WINDOW), WINDOW - 1 - np.arange(WINDOW))   # trust window centre
        for (s, c), p in zip(meta, P):
            acc[s:s + WINDOW] += (p * scale + c) * wt[:, None]
            wsum[s:s + WINDOW] += wt

    miss = ~obs
    filled = grid.copy()
    cov = miss & (wsum > 0)
    filled[cov] = acc[cov] / wsum[cov][:, None]
    unc = miss & (wsum == 0)
    filled[unc] = herm[unc]                   # fallback where no window could run

    first_obs, last_obs = int(np.argmax(obs)), int(N - 1 - np.argmax(obs[::-1]))
    warns = []
    lg = _longest_run(miss)
    if lg > MAX_GAP:
        warns.append(f"Longest gap is {lg / HZ:.1f} s; the model was trained on gaps up to "
                     f"{MAX_GAP / HZ:.0f} s, so those points are less reliable.")
    if miss.mean() > 0.6:
        warns.append(f"{miss.mean() * 100:.0f}% of points are missing (training used up to ~60-80%).")
    if first_obs > 0 or last_obs < N - 1:
        warns.append("Track starts/ends with missing points: those are extrapolated, least reliable.")
    if unc.any():
        warns.append(f"{int(unc.sum())} missing points had too few nearby observations; "
                     "filled with Hermite interpolation instead of the model.")
    info = dict(n=N, n_missing=int(miss.sum()), missing_pct=float(miss.mean() * 100),
                longest_gap_s=lg / HZ, warnings=warns)
    return filled, herm, lin, info


def impute_df(df, predict, ds, in_hz=5.0):
    """DataFrame in -> (DataFrame out, info). Observed points are never changed."""
    grid, t0, geo = prepare(df, in_hz)
    filled, herm, lin, info = impute_grid(grid, predict, ds)
    obs = np.isfinite(grid).all(1)
    out = pd.DataFrame({"t": t0 + np.arange(len(grid)) / HZ,
                        "x": filled[:, 0], "y": filled[:, 1],
                        "imputed": (~obs).astype(int),
                        "x_hermite": herm[:, 0], "y_hermite": herm[:, 1]})
    if geo is not None:
        lat0, lon0 = geo
        out["lat"] = lat0 + filled[:, 1] / 110_540.0
        out["lon"] = lon0 + filled[:, 0] / (111_320.0 * np.cos(np.deg2rad(lat0)))
    info["unit"] = UNIT[ds] if geo is None else "m"
    return out, info, grid


# ------------------------------------------------------------------ accuracy test
def simulate(df, predict, ds, gap_s=2.0, trials=20, in_hz=5.0, seed=0):
    """Delete a random gap from a (mostly) complete track, impute, score against the truth.
    Returns metrics dict + arrays for one example plot."""
    grid, _, _ = prepare(df, in_hz)
    N = len(grid)
    L = max(1, int(round(gap_s * HZ)))
    if N < max(WINDOW, L + 6):
        raise ValueError("Track too short for this gap length.")
    rng = np.random.default_rng(seed)
    se = dict(model=0.0, hermite=0.0, linear=0.0)
    cnt, example, done, tries = 0, None, 0, 0
    while done < trials and tries < trials * 20:
        tries += 1
        g0 = int(rng.integers(2, N - L - 1))
        if not np.isfinite(grid[g0:g0 + L]).all():
            continue
        lo, hi = max(0, g0 - WINDOW), min(N, g0 + L + WINDOW)      # local crop = fast
        piece = grid[lo:hi].copy()
        truth = piece[g0 - lo:g0 - lo + L].copy()
        piece[g0 - lo:g0 - lo + L] = np.nan
        if np.isfinite(piece).all(1).sum() < 2:
            continue
        filled, herm, lin, _ = impute_grid(piece, predict, ds)
        sl = slice(g0 - lo, g0 - lo + L)
        for k, arr in (("model", filled), ("hermite", herm), ("linear", lin)):
            se[k] += float(((arr[sl] - truth) ** 2).sum())
        cnt += L
        done += 1
        if example is None:
            example = dict(piece=piece, truth_full=grid[lo:hi], filled=filled, herm=herm, sl=sl)
    if done == 0:
        raise ValueError("Could not find a complete stretch to cut a gap from "
                         "(upload a track with few missing points).")
    rmse = {k: float(np.sqrt(v / cnt)) for k, v in se.items()}
    return dict(rmse=rmse, trials=done, points=cnt, unit=UNIT[ds], gap_s=L / HZ), example
