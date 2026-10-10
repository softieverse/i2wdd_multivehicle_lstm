"""
train_pedestrian.py

Trains TrajectoryLSTM (model.py) on pedestrian pixel-trajectory CSVs
built from IDD-PeD (build_pedestrian_tracks.py).

Mirrors train.py's design, adapted for pixel space instead of GPS:
  - Input per timestep: [x, y, bbox_w, bbox_h, mask_flag]. bbox_w/bbox_h
    are always-present context (like IMU in the GPS model) — NOT masked.
  - Masking strategy/windowing/route-level split: identical to train.py.
  - Output: [x, y] pixel centroid, evaluated via masked RMSE, then reported
    in pixel Euclidean distance (not haversine meters — no GPS here).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model import TrajectoryLSTM, masked_rmse_loss

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MERGED_DIR = Path("data/processed/pedestrian_tracks")
CHECKPOINT_PATH = Path("checkpoints/pedestrian_model.pt")

WINDOW_SIZE = 50
STRIDE = 25
MASK_STRATEGY = "contiguous"
MASK_RATIO = 0.2
MIN_GAP_LEN = 5
MAX_GAP_LEN = 15

VAL_ROUTES_FRACTION = 0.3
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEED = 42

FEATURE_COLS = ["x", "y", "bbox_w", "bbox_h"]  # fixed schema, no auto-discovery needed


def make_mask(n_points: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(n_points, dtype=bool)
    if MASK_STRATEGY == "random":
        n_mask = int(n_points * MASK_RATIO)
        idx = rng.choice(n_points, size=n_mask, replace=False)
        mask[idx] = True
    elif MASK_STRATEGY == "contiguous":
        target_masked = int(n_points * MASK_RATIO)
        masked_so_far = 0
        attempts = 0
        while masked_so_far < target_masked and attempts < 50:
            attempts += 1
            gap_len = rng.integers(MIN_GAP_LEN, MAX_GAP_LEN + 1)
            start = rng.integers(0, max(1, n_points - gap_len))
            end = min(start + gap_len, n_points)
            newly_masked = (~mask[start:end]).sum()
            mask[start:end] = True
            masked_so_far += newly_masked
    else:
        raise ValueError(f"Unknown MASK_STRATEGY: {MASK_STRATEGY}")
    return mask


def compute_norm_stats(track_dfs: dict[str, pd.DataFrame], feature_cols: list[str]) -> dict[str, tuple[float, float]]:
    all_data = pd.concat(
        [df[feature_cols] for df in track_dfs.values()],
        ignore_index=True,
    )
    stats = {}
    for col in feature_cols:
        mean, std = all_data[col].mean(), all_data[col].std()
        if std < 1e-8:
            std = 1.0
        stats[col] = (float(mean), float(std))
    return stats


class PedestrianWindowDataset(Dataset):
    def __init__(self, track_dfs: dict[str, pd.DataFrame], feature_cols: list[str],
                 seed: int, norm_stats: dict[str, tuple[float, float]]):
        self.samples = []
        rng = np.random.default_rng(seed)

        for track_name, df in track_dfs.items():
            n = len(df)
            if n < WINDOW_SIZE:
                continue
            for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
                window = df.iloc[start:start + WINDOW_SIZE]
                mask = make_mask(WINDOW_SIZE, rng)

                features = window[feature_cols].to_numpy(dtype=np.float32).copy()
                for i, col in enumerate(feature_cols):
                    mean, std = norm_stats[col]
                    features[:, i] = (features[:, i] - mean) / std

                x_mean, x_std = norm_stats["x"]
                y_mean, y_std = norm_stats["y"]
                target_xy = window[["x", "y"]].to_numpy(dtype=np.float32).copy()
                target_xy[:, 0] = (target_xy[:, 0] - x_mean) / x_std
                target_xy[:, 1] = (target_xy[:, 1] - y_mean) / y_std

                x_idx, y_idx = feature_cols.index("x"), feature_cols.index("y")
                features[mask, x_idx] = 0.0
                features[mask, y_idx] = 0.0
                mask_flag = mask.astype(np.float32).reshape(-1, 1)
                features = np.concatenate([features, mask_flag], axis=1)

                self.samples.append((features, target_xy, mask.astype(np.float32)))

        log.info(f"Built {len(self.samples)} windows (size={WINDOW_SIZE}, stride={STRIDE})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, target, mask = self.samples[idx]
        return (
            torch.from_numpy(features),
            torch.from_numpy(target),
            torch.from_numpy(mask),
        )


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0

    with torch.set_grad_enabled(train):
        for features, target, mask in loader:
            features, target, mask = features.to(device), target.to(device), mask.to(device)

            if train:
                optimizer.zero_grad()

            pred = model(features, target=target if train else None)
            loss = masked_rmse_loss(pred, target, mask)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    torch.manual_seed(SEED)

    track_files = sorted(MERGED_DIR.glob("*.csv"))
    if not track_files:
        raise FileNotFoundError(f"No pedestrian track CSVs found in {MERGED_DIR}. Run build_pedestrian_tracks.py first.")

    track_dfs = {f.stem: pd.read_csv(f) for f in track_files}
    log.info(f"Loaded {len(track_dfs)} pedestrian tracks")

    track_names = list(track_dfs.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(track_names)
    n_val = max(1, int(len(track_names) * VAL_ROUTES_FRACTION))
    val_tracks, train_tracks = track_names[:n_val], track_names[n_val:]
    log.info(f"Train tracks: {len(train_tracks)} | Val tracks: {len(val_tracks)}")

    train_ds_tracks = {t: track_dfs[t] for t in train_tracks}
    norm_stats = compute_norm_stats(train_ds_tracks, FEATURE_COLS)
    log.info(f"Computed normalization stats from {len(train_tracks)} train tracks")

    train_ds = PedestrianWindowDataset(train_ds_tracks, FEATURE_COLS, seed=SEED, norm_stats=norm_stats)
    val_ds = PedestrianWindowDataset({t: track_dfs[t] for t in val_tracks}, FEATURE_COLS, seed=SEED + 1, norm_stats=norm_stats)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    n_features = len(FEATURE_COLS) + 1  # +1 for mask_flag
    model = TrajectoryLSTM(n_features=n_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        log.info(f"Epoch {epoch:3d}/{EPOCHS} | train loss: {train_loss:.5f} | val loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_cols": FEATURE_COLS,
                "n_features": n_features,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "epoch": epoch,
                "val_loss": val_loss,
                "norm_stats": norm_stats,
            }, CHECKPOINT_PATH)
            log.info(f"  -> new best, saved checkpoint to {CHECKPOINT_PATH}")

    log.info(f"\nDone. Best val loss: {best_val_loss:.5f}")
    log.info("Note: val loss above is in normalized pixel units, not raw pixels.")
    log.info("Run evaluate_pedestrian.py for RMSE in pixels (Euclidean distance).")


if __name__ == "__main__":
    main()