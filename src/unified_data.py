"""
unified_data.py

Loads HetroD, IDD-Multimodal (GPS), IDD-PeD and I2WDD (two-wheeler phone GPS)
into the common schema, and exposes one torch Dataset over all of them.

Each "track" is a dict: ds (dataset id), agent, split, xy (T, 2) float32 at 5 Hz.
"""

import glob
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common_schema import (AGENT, DATASET, HZ, SCALE, STRIDE_BY_DS, WINDOW,
                           agent_from_class, hermite_fill, linear_fill,
                           sample_mask)

# ---- paths (edit if yours differ) -------------------------------------------
HETROD_ROOT = os.path.expanduser(
    "~/Downloads/hetrod/hetroD-dataset-v1.0-without-Loc2/data/01_automated_processing")
IDD_MM_ROOT = os.path.expanduser(
    "~/datasets/multimodal/primary/idd_multimodal/primary")
PED_DIR = "data/processed/pedestrian_tracks"
PED_FPS = 30.0          # ASSUMED frame rate of IDD-PeD videos -- check and change
CACHE = "data/processed/unified_tracks.pkl"


# ---- HetroD ------------------------------------------------------------------
def _hetrod_split(rec_id: int) -> str:
    return "train" if rec_id <= 39 else "val" if rec_id <= 44 else "test"


def load_hetrod():
    from hetrod_loader import list_recordings, load_recording   # your existing file
    tracks = []
    for prefix in list_recordings(HETROD_ROOT):
        df, _ = load_recording(prefix, HETROD_ROOT, target_hz=HZ)
        split = _hetrod_split(int(prefix))
        for _, g in df.groupby("track"):
            g = g.sort_values("t")
            if len(g) < WINDOW:
                continue
            tracks.append(dict(ds=DATASET["hetrod"],
                               agent=agent_from_class(str(g["cls"].iloc[0])),
                               split=split,
                               xy=g[["x", "y"]].to_numpy(np.float32)))
    return tracks


# ---- IDD-Multimodal (GPS only, one car) -----------------------------------------
def _parse_ts(s: str) -> float:                    # "09-00-31-289685"
    h, m, sec, us = s.split("-")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(us) / 1e6


def load_idd_mm():
    """Only train.csv and val.csv have GPS (test.csv is unlabelled).
    train.csv -> train.  val.csv -> first half = val, second half = test,
    with a WINDOW-sized buffer dropped between them so windows never overlap."""
    tracks = []
    for d in ("d0", "d1", "d2"):
        for fname in ("train", "val"):
            df = pd.read_csv(f"{IDD_MM_ROOT}/{d}/{fname}.csv", dtype={"timestamp": str})
            t = df["timestamp"].map(_parse_ts).to_numpy()
            t = t - t[0]
            lat, lon = df["latitude"].to_numpy(), df["longitude"].to_numpy()
            grid = np.arange(0.0, t[-1], 1.0 / HZ)                 # resample to 5 Hz
            lat, lon = np.interp(grid, t, lat), np.interp(grid, t, lon)
            x = (lon - lon[0]) * 111_320.0 * np.cos(np.deg2rad(lat[0]))
            y = (lat - lat[0]) * 110_540.0
            xy = np.stack([x, y], 1).astype(np.float32)

            if fname == "train":
                parts = [("train", xy)]
            else:
                half = len(xy) // 2
                parts = [("val", xy[:half]), ("test", xy[half + WINDOW:])]
            for split, seg in parts:
                if len(seg) >= WINDOW:
                    tracks.append(dict(ds=DATASET["idd_mm"], agent=AGENT["vehicle"],
                                       split=split, xy=seg))
    return tracks


# ---- IDD-PeD (pixel tracks) ---------------------------------------------------
def _ped_split(video_key: str) -> str:               # split by video, never by track
    h = zlib.crc32(video_key.encode()) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def load_idd_ped():
    step = int(round(PED_FPS / HZ))
    tracks = []
    for path in sorted(glob.glob(f"{PED_DIR}/*.csv")):
        name = os.path.basename(path)
        split = _ped_split(name.split("_gp_")[0])        # gp_set_0001_vid_0001
        df = pd.read_csv(path).sort_values("frame")
        df = df[df["frame"] % step == 0]
        if len(df) < WINDOW:
            continue
        frames, xy = df["frame"].to_numpy(), df[["x", "y"]].to_numpy(np.float32)
        cuts = np.where(np.diff(frames) != step)[0] + 1   # split at frame gaps
        for fr, seg in zip(np.split(frames, cuts), np.split(xy, cuts)):
            if len(seg) >= WINDOW:
                tracks.append(dict(ds=DATASET["idd_ped"], agent=AGENT["pedestrian"],
                                   split=split, xy=seg))
    return tracks


# ---- I2WDD (two-wheeler phone GPS) --------------------------------------------
def load_i2wdd():
    """Local metres, 5 Hz, gap-free segments (see i2wdd_loader.py).
    Split by source file (never by window): train = 10_01_25a + 18_01_25,
    val = 28_12_24/GPS_2, test = 28_12_24/GPS_1."""
    from i2wdd_loader import load_i2wdd as load_raw     # reads ~/datasets/i2wdd
    out = []
    for tr in load_raw():
        src = tr["source"]
        if src == "28_12_24/GPS_1":
            split = "test"
        elif src == "28_12_24/GPS_2":
            split = "val"
        else:
            split = "train"
        out.append(dict(ds=DATASET["i2wdd"], agent=AGENT["two_wheeler"],
                        split=split, xy=tr["xy"]))
    return out


# ---- all tracks (cached) ------------------------------------------------------
def load_all_tracks(use_cache=True):
    if use_cache and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    tracks = load_hetrod() + load_idd_mm() + load_idd_ped() + load_i2wdd()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(tracks, f)
    return tracks


# ---- torch Dataset ------------------------------------------------------------
class UnifiedWindows(Dataset):
    """Windows of WINDOW steps with a fresh random mask each epoch (train) or a
    fixed mask per window (val/test, so results are comparable between runs).
    Val/test items are cached after the first epoch, so they are built only once.

    Input x per step (7 features):
        [x_masked, y_masked, mask, x_linear, y_linear, x_hermite, y_hermite]
    """

    def __init__(self, tracks, split, seed=0):
        wins, ds, agent = [], [], []
        for tr in tracks:
            if tr["split"] != split:
                continue
            for s in range(0, len(tr["xy"]) - WINDOW + 1, STRIDE_BY_DS[tr["ds"]]):
                wins.append(tr["xy"][s:s + WINDOW])
                ds.append(tr["ds"])
                agent.append(tr["agent"])
        if not wins:
            raise ValueError(f"no windows for split '{split}'")
        self.xy = np.stack(wins).astype(np.float32)
        self.ds, self.agent = np.array(ds), np.array(agent)
        self.fixed, self.seed = split != "train", seed
        self._cache = {}

    def __len__(self):
        return len(self.xy)

    def __getitem__(self, i):
        if self.fixed:                       # val/test masks are fixed, so cache them
            c = self._cache.get(i)
            if c is None:
                c = self._cache[i] = self._make(i)
            return c
        return self._make(i)

    def _make(self, i):
        rng = np.random.default_rng(self.seed + i) if self.fixed else np.random.default_rng()
        xy, ds = self.xy[i], int(self.ds[i])
        obs = sample_mask(WINDOW, rng)
        scale = SCALE[ds]
        xyn = (xy - xy[obs].mean(0)) / scale          # centre on observed points only
        x = np.zeros((WINDOW, 7), np.float32)
        x[:, 0:2] = xyn * obs[:, None]                # masked coordinates
        x[:, 2] = obs                                 # mask flag
        x[:, 3:5] = linear_fill(xyn, obs)             # linear-interp baseline
        x[:, 5:7] = hermite_fill(xyn, obs)            # Hermite baseline (model's start point)
        return dict(x=torch.from_numpy(x), y=torch.from_numpy(xyn.astype(np.float32)),
                    obs=torch.from_numpy(obs.astype(np.float32)),
                    agent=int(self.agent[i]), ds=ds, scale=float(scale))