"""
unified_loader.py -- one loader for HetroD, IDD Multimodal, IDD-PeD, I2WDD.

Pipeline:  adapter.load_tracks()  ->  resample to HZ  ->  cut 10 s windows  ->  punch a gap  ->  item dict

Item dict (what train_unified.py expects):
    xy        (T,2) float32  ground truth, metres, relative to the window's first point
    gap_mask  (T,)  bool     True = missing point (model must impute)
    imu       (T,6) float32  accel xyz + gyro xyz, zeros if the dataset has none
    imu_mask  (T,)  bool     True where imu is real
    agent     int            0 two-wheeler, 1 car, 2 pedestrian, 3 hetrod-mix
    dataset   int            0 hetrod, 1 idd_mm, 2 idd_ped, 3 i2wdd
    gap_len   int            gap length in samples (for per-gap-length eval)

TODO for you: fill in the 4 load_tracks() bodies (marked TODO). Everything else is done.
Expected layout:  root/hetrod, root/idd_mm, root/idd_ped, root/i2wdd
Self-test (no data needed):  python unified_loader.py
"""
import zlib, warnings
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

HZ = 5                      # common sampling rate (HetroD native; I2WDD 10 Hz is downsampled)
WIN_S = 10.0                # window length, seconds
STRIDE_S = 5.0              # window stride, seconds
GAP_S = (1.2, 3.0)          # gap length range, seconds (same as the HetroD eval)

DS_ID = dict(hetrod=0, idd_mm=1, idd_ped=2, i2wdd=3)
AGENT = dict(two_wheeler=0, car=1, pedestrian=2, mix=3)


# ------------------------------------------------------------------ helpers
@dataclass
class Track:
    xy: np.ndarray                  # (N,2) metres, any origin
    t: np.ndarray                   # (N,) seconds, increasing
    agent: int
    dataset: int
    group: str                      # recording / video / file id -- splits are done on this (no leakage)
    imu: Optional[np.ndarray] = None  # (N,6) or None


def latlon_to_xy(lat, lon, lat0=None, lon0=None):
    """Equirectangular lat/lon (deg) -> local metres. Fine for the few-hundred-metre extents here."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    lat0 = lat[0] if lat0 is None else lat0
    lon0 = lon[0] if lon0 is None else lon0
    R = 6371000.0
    x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R
    return np.stack([x, y], -1)


def hash_split(key: str, val=10, test=10) -> str:
    """Deterministic split on a group id: 80/10/10 by default."""
    b = zlib.crc32(str(key).encode()) % 100
    return "test" if b < test else "val" if b < test + val else "train"


def segment_on_jumps(tr: Track, max_dt: float) -> List[Track]:
    """Split a track wherever consecutive timestamps are more than max_dt apart."""
    cuts = np.where(np.diff(tr.t) > max_dt)[0] + 1
    out = []
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(tr.t)]):
        if e - s >= 2:
            out.append(Track(tr.xy[s:e], tr.t[s:e], tr.agent, tr.dataset, tr.group,
                             None if tr.imu is None else tr.imu[s:e]))
    return out


def resample(tr: Track, hz: int) -> Track:
    """Linear resample onto a uniform grid at hz."""
    t0, t1 = tr.t[0], tr.t[-1]
    n = int(np.floor((t1 - t0) * hz)) + 1
    g = t0 + np.arange(n) / hz
    xy = np.stack([np.interp(g, tr.t, tr.xy[:, k]) for k in range(2)], -1)
    imu = None if tr.imu is None else np.stack([np.interp(g, tr.t, tr.imu[:, k]) for k in range(6)], -1)
    return Track(xy.astype(np.float32), g, tr.agent, tr.dataset, tr.group,
                 None if imu is None else imu.astype(np.float32))


# ------------------------------------------------------------------ adapters
class Adapter:
    name = ""
    def load_tracks(self, root: str, split: str) -> List[Track]:
        raise NotImplementedError
    def split_of(self, group: str) -> str:
        return hash_split(group)


class HetroD(Adapter):
    name = "hetrod"
    def split_of(self, group):
        # your earlier setup: train recs 02-39, val 40-44, test 45-49  (group = recording number)
        r = int(group)
        return "train" if r <= 39 else "val" if r <= 44 else "test"
    def load_tracks(self, root, split):
        # TODO: for each recording file -> for each vehicle id: Track(
        #   xy=<x,y metres>, t=<seconds>, agent=AGENT['mix'], dataset=DS_ID['hetrod'], group=<rec number>)
        # keep only recordings where self.split_of(rec) == split
        raise NotImplementedError("HetroD adapter not filled in")


class IDDMultimodal(Adapter):
    name = "idd_mm"
    def load_tracks(self, root, split):
        # TODO: cars. If positions are lat/lon use latlon_to_xy(). GPS only -> imu=None.
        #   Track(xy, t, agent=AGENT['car'], dataset=DS_ID['idd_mm'], group=<drive/sequence id>)
        #   keep only tracks where self.split_of(group) == split
        raise NotImplementedError("IDD Multimodal adapter not filled in")


class IDDPeD(Adapter):
    name = "idd_ped"
    def load_tracks(self, root, split):
        # TODO: pedestrians. Your earlier table was in pixels -- convert to metres here (or
        #   calibrate with a homography) so the SCALE in train_unified.py makes sense. imu=None.
        #   Track(xy, t, agent=AGENT['pedestrian'], dataset=DS_ID['idd_ped'], group=<video id>)
        raise NotImplementedError("IDD-PeD adapter not filled in")


class I2WDD(Adapter):
    name = "i2wdd"
    def load_tracks(self, root, split):
        # TODO: two-wheeler. 10 Hz GPS + accel/gyro; mostly reuse src/data_merger.py.
        #   xy = latlon_to_xy(lat, lon); imu = (N,6) [ax,ay,az,gx,gy,gz]
        #   Track(xy, t, agent=AGENT['two_wheeler'], dataset=DS_ID['i2wdd'], group=<ride/file id>, imu=imu)
        raise NotImplementedError("I2WDD adapter not filled in")


ADAPTERS = dict(hetrod=HetroD, idd_mm=IDDMultimodal, idd_ped=IDDPeD, i2wdd=I2WDD)


# ------------------------------------------------------------------ dataset
class WindowDataset:
    """Windows of WIN_S seconds with a random 1.2-3 s gap cut from the middle.
    train: gap re-drawn every access.  val/test: gap fixed by index (reproducible numbers)."""
    def __init__(self, tracks: List[Track], split: str, hz=HZ, win_s=WIN_S, stride_s=STRIDE_S,
                 gap_s=GAP_S, min_disp=0.0, seed=0):
        self.split, self.hz, self.seed = split, hz, seed
        self.T = int(round(win_s * hz))
        self.gmin, self.gmax = int(round(gap_s[0] * hz)), int(round(gap_s[1] * hz))
        stride = max(1, int(round(stride_s * hz)))
        self.segs, self.index = [], []
        for tr in tracks:
            for seg in segment_on_jumps(tr, max_dt=2.5 / min(hz, 5)):    # break on >0.5 s timestamp holes
                seg = resample(seg, hz)
                si = len(self.segs)
                self.segs.append(seg)
                for s in range(0, len(seg.t) - self.T + 1, stride):
                    w = seg.xy[s:s + self.T]
                    if min_disp and np.linalg.norm(w[-1] - w[0]) < min_disp:
                        continue                                          # drop parked / standing windows
                    self.index.append((si, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, s = self.index[i]
        seg = self.segs[si]
        rng = np.random.default_rng(self.seed * 1_000_003 + i) if self.split != "train" else np.random
        xy = seg.xy[s:s + self.T]
        xy = (xy - xy[0]).astype(np.float32)
        L = int(rng.randint(self.gmin, self.gmax + 1)) if self.split == "train" else int(rng.integers(self.gmin, self.gmax + 1))
        lo, hi = self.T // 5, self.T - self.T // 5 - L           # keep gap away from both ends
        g0 = int(rng.randint(lo, hi + 1)) if self.split == "train" else int(rng.integers(lo, hi + 1))
        gap = np.zeros(self.T, bool); gap[g0:g0 + L] = True
        if seg.imu is None:
            imu = np.zeros((self.T, 6), np.float32); im = np.zeros(self.T, bool)
        else:
            imu = seg.imu[s:s + self.T].copy(); im = np.ones(self.T, bool)
        return dict(xy=xy, gap_mask=gap, imu=imu, imu_mask=im,
                    agent=int(seg.agent), dataset=int(seg.dataset), gap_len=L)


class Concat:
    def __init__(self, parts):
        self.parts = parts
        self.cum = np.cumsum([0] + [len(p) for p in parts])
    def __len__(self):
        return int(self.cum[-1])
    def __getitem__(self, i):
        k = int(np.searchsorted(self.cum, i, side="right") - 1)
        return self.parts[k][i - int(self.cum[k])]


def build(root: str, split: str, datasets=("hetrod", "idd_mm", "idd_ped", "i2wdd"), **kw):
    """Concatenate all four. Adapters that aren't filled in yet are skipped with a warning,
    so you can start training on HetroD alone."""
    import os
    parts = []
    for name in datasets:
        ad = ADAPTERS[name]()
        try:
            tracks = ad.load_tracks(os.path.join(root, name), split)
        except NotImplementedError as e:
            warnings.warn(f"skipping {name}: {e}")
            continue
        ds = WindowDataset(tracks, split, **kw)
        print(f"[loader] {split:5s} {name:8s} tracks={len(tracks):6d} windows={len(ds):8d}")
        if len(ds):
            parts.append(ds)
    if not parts:
        raise RuntimeError("no datasets loaded -- fill in at least one load_tracks()")
    return Concat(parts)


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    def fake(ds, agent, hz_native, n, with_imu, group):
        t = np.arange(n) / hz_native
        xy = np.cumsum(rng.normal(0, 0.3, (n, 2)), 0) + np.array([5.0, -3.0])
        return Track(xy, t, agent, ds, group, rng.normal(size=(n, 6)) if with_imu else None)
    tracks = [fake(3, 0, 10, 1200, True, "ride1"), fake(0, 3, 5, 800, False, "01")]
    tracks[1].t[400:] += 3.0                                     # a timestamp hole -> should segment
    for split in ("train", "val"):
        d = WindowDataset(tracks, split)
        it = d[0]
        assert it["xy"].shape == (50, 2) and it["gap_mask"].shape == (50,) and it["imu"].shape == (50, 6)
        assert abs(it["xy"][0]).max() == 0                       # window-relative
        assert not it["gap_mask"][0] and not it["gap_mask"][-1]  # ends observed
        assert 6 <= it["gap_len"] <= 15 and it["gap_mask"].sum() == it["gap_len"]
        if split != "train":
            assert (d[3]["gap_mask"] == d[3]["gap_mask"]).all() and d[3]["gap_len"] == d[3]["gap_len"]
        print(split, "windows:", len(d), "| first item gap_len:", it["gap_len"], "| imu real:", bool(it["imu_mask"][0]))
    d = Concat([WindowDataset(tracks[:1], "val"), WindowDataset(tracks[1:], "val")])
    print("concat len", len(d), "| item ds ids:", d[0]["dataset"], d[len(d) - 1]["dataset"])
    print("self-test ok")