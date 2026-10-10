import os
import numpy as np
import torch
from torch.utils.data import Dataset
from hetrod_loader import list_recordings, load_recording

CLASSES = ["bicycle", "car", "motorcycle", "pedestrian", "truck"]
TARGET_HZ, WINDOW, STRIDE = 5.0, 30, 15      # 6 s windows at 5 Hz
MAX_WIN_PER_TRACK, MIN_MOVE_M = 20, 1.0
POS_SCALE, VEL_SCALE = 20.0, 10.0            # meters, m/s (fixed, so RMSE converts back easily)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "hetrod_windows")
DYN_COLS = ["x", "y", "vx", "vy", "heading_sin", "heading_cos"]


def build_recording(prefix):
    df, _ = load_recording(prefix, target_hz=TARGET_HZ)
    dyn_all, stat_all, cls_all = [], [], []
    for _, g in df.groupby("track"):
        if len(g) < WINDOW:
            continue
        a = g[DYN_COLS].to_numpy(np.float32)
        starts = np.arange(0, len(a) - WINDOW + 1, STRIDE)
        moved = np.linalg.norm(a[starts + WINDOW - 1, :2] - a[starts, :2], axis=1)
        starts = starts[moved >= MIN_MOVE_M]
        if len(starts) == 0:
            continue
        if len(starts) > MAX_WIN_PER_TRACK:
            starts = starts[np.linspace(0, len(starts) - 1, MAX_WIN_PER_TRACK).astype(int)]
        c = g["cls"].iloc[0]
        onehot = np.zeros(len(CLASSES), np.float32)
        onehot[CLASSES.index(c)] = 1
        stat = np.concatenate([onehot, [g["width"].iloc[0] / 10, g["length"].iloc[0] / 10]]).astype(np.float32)
        for s in starts:
            w = a[s:s + WINDOW].copy()
            w[:, :2] = (w[:, :2] - w[0, :2]) / POS_SCALE   # relative to first step
            w[:, 2:4] /= VEL_SCALE
            dyn_all.append(w); stat_all.append(stat); cls_all.append(CLASSES.index(c))
    return (np.stack(dyn_all), np.stack(stat_all), np.array(cls_all, np.int8)) if dyn_all else None


def build_all():
    os.makedirs(OUT, exist_ok=True)
    total = np.zeros(len(CLASSES), int)
    for r in list_recordings():
        res = build_recording(r)
        if res is None:
            print(r, "no windows"); continue
        dyn, stat, cls = res
        np.savez_compressed(f"{OUT}/{r}.npz", dyn=dyn, stat=stat, cls=cls)
        total += np.bincount(cls, minlength=len(CLASSES))
        print(f"[{r}] windows={len(dyn)}")
    print("\nwindows per class:", dict(zip(CLASSES, total.tolist())), "total:", total.sum())


class HetroDDataset(Dataset):
    """Returns x (W,14), target (W,2), miss (W,) in normalized units. Multiply RMSE by POS_SCALE for meters."""
    def __init__(self, recs, gap_min=6, gap_max=15, seed=None):
        d = [np.load(f"{OUT}/{r}.npz") for r in recs]
        self.dyn = np.concatenate([z["dyn"] for z in d])
        self.stat = np.concatenate([z["stat"] for z in d])
        self.cls = np.concatenate([z["cls"] for z in d])
        self.gap_min, self.gap_max = gap_min, gap_max
        self.fixed = None
        if seed is not None:   # fixed gaps -> reproducible validation
            g = torch.Generator().manual_seed(seed)
            n = len(self.dyn)
            L = torch.randint(gap_min, gap_max + 1, (n,), generator=g)
            S = 1 + (torch.rand(n, generator=g) * (WINDOW - 1 - L)).long()
            self.fixed = (L, S)

    def __len__(self):
        return len(self.dyn)

    def __getitem__(self, i):
        if self.fixed is not None:
            L, S = int(self.fixed[0][i]), int(self.fixed[1][i])
        else:
            L = int(torch.randint(self.gap_min, self.gap_max + 1, ()))
            S = 1 + int(torch.rand(()) * (WINDOW - 1 - L))
        miss = np.zeros(WINDOW, np.float32); miss[S:S + L] = 1
        dyn = self.dyn[i].copy(); target = dyn[:, :2].copy()
        dyn[miss > 0] = 0                                  # hide position, velocity, heading in the gap
        stat = np.repeat(self.stat[i][None], WINDOW, axis=0)
        x = np.concatenate([dyn, miss[:, None], stat], axis=1)   # 6 + 1 + 7 = 14
        return torch.from_numpy(x), torch.from_numpy(target), torch.from_numpy(miss)


if __name__ == "__main__":
    build_all()