"""
train.py

Trains TrajectoryLSTM (model.py) on the merged route CSVs from data_merger.py.

Design (review before trusting):
  - Masking strategy: CONTIGUOUS GAPS by default (mimics the real sensor
    dropout pattern found in the gyro data), not random scattered points.
    Set MASK_STRATEGY = "random" to switch, or run both and compare.
  - Windowing: each route is chopped into fixed-length overlapping windows
    (WINDOW_SIZE timesteps, STRIDE apart). Short trailing remainder is
    dropped.
  - Split: routes are split at the ROUTE level (not window level) into
    train/val, so no window from the same route leaks across the split —
    otherwise validation RMSE would look better than it really is.
  - Feature columns are auto-detected from the merged CSVs (whatever
    accel_*/gyro_* columns exist for that route) plus lat/lon/alt/speed,
    then a mask_flag column is appended. Routes with fewer sensor
    positions naturally have fewer feature columns — this script pads
    missing columns with 0 so all routes share the same feature schema
    for a single model. CHECK this is what you want; the alternative is
    training separate models per sensor configuration.
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

MERGED_DIR = Path("~/lstm-trajectory-imputation/data/processed/merged").expanduser()
CHECKPOINT_PATH = Path("~/lstm-trajectory-imputation/checkpoints/model.pt").expanduser()

WINDOW_SIZE = 50
STRIDE = 25
MASK_STRATEGY = "contiguous"  # "contiguous" or "random"
MASK_RATIO = 0.2  # fraction of each window's GPS points to mask
MIN_GAP_LEN = 5    # contiguous mode: min consecutive masked points per gap
MAX_GAP_LEN = 15   # contiguous mode: max consecutive masked points per gap

VAL_ROUTES_FRACTION = 0.3  # held out at the route level, not window level
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEED = 42

EARTH_RADIUS_M = 6_371_000


def discover_feature_columns(all_routes: dict[str, pd.DataFrame]) -> list[str]:
    """Union of all accel_*/gyro_* columns across every route, so every
    route can be projected onto the same feature schema (missing ones
    filled with 0)."""
    base = ["lat", "lon", "alt", "speed_2d", "speed_3d"]
    sensor_cols = set()
    for df in all_routes.values():
        sensor_cols.update(c for c in df.columns if c.startswith("accel_") or c.startswith("gyro_"))
    return base + sorted(sensor_cols)


def make_mask(n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Returns a boolean array, True = masked (imputation target)."""
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


def compute_norm_stats(route_dfs: dict[str, pd.DataFrame], feature_cols: list[str]) -> dict[str, tuple[float, float]]:
    """Mean/std per feature, computed ONLY from training routes to avoid
    leaking validation-route statistics into the normalization."""
    all_data = pd.concat(
        [df.reindex(columns=feature_cols, fill_value=0.0).fillna
        (0.0) for df in route_dfs.values()],
        ignore_index=True,
    )
    stats = {}
    for col in feature_cols:
        mean, std = all_data[col].mean(), all_data[col].std()
        if std < 1e-8:
            std = 1.0  # avoid divide-by-zero for constant columns
        stats[col] = (float(mean), float(std))
    return stats


class TrajectoryWindowDataset(Dataset):
    def __init__(self, route_dfs: dict[str, pd.DataFrame], feature_cols: list[str],
                 seed: int, norm_stats: dict[str, tuple[float, float]]):
        self.samples = []  # list of (features_df_slice, gps_target, mask)
        rng = np.random.default_rng(seed)

        for route_name, df in route_dfs.items():
            df = df.reindex(columns=feature_cols, fill_value=0.0).fillna(0.0)
            n = len(df)
            for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
                window = df.iloc[start:start + WINDOW_SIZE]
                mask = make_mask(WINDOW_SIZE, rng)

                features = window[feature_cols].to_numpy(dtype=np.float32).copy()
                # normalize every feature column (z-score, stats from train routes only)
                for i, col in enumerate(feature_cols):
                    mean, std = norm_stats[col]
                    features[:, i] = (features[:, i] - mean) / std

                # target stays normalized too, so the loss is on a sane scale;
                # evaluate.py inverts this back to real degrees for meter-RMSE
                lat_mean, lat_std = norm_stats["lat"]
                lon_mean, lon_std = norm_stats["lon"]
                target_latlon = window[["lat", "lon"]].to_numpy(dtype=np.float32).copy()
                target_latlon[:, 0] = (target_latlon[:, 0] - lat_mean) / lat_std
                target_latlon[:, 1] = (target_latlon[:, 1] - lon_mean) / lon_std

                lat_idx, lon_idx = feature_cols.index("lat"), feature_cols.index("lon")
                features[mask, lat_idx] = 0.0
                features[mask, lon_idx] = 0.0
                mask_flag = mask.astype(np.float32).reshape(-1, 1)
                features = np.concatenate([features, mask_flag], axis=1)

                self.samples.append((features, target_latlon, mask.astype(np.float32)))

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


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


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

    route_files = sorted(MERGED_DIR.glob("*.csv"))
    if not route_files:
        raise FileNotFoundError(f"No merged route CSVs found in {MERGED_DIR}. Run data_merger.py first.")

    route_dfs = {f.stem: pd.read_csv(f, parse_dates=["time"]) for f in route_files}
    log.info(f"Loaded {len(route_dfs)} merged routes: {list(route_dfs.keys())}")

    feature_cols = discover_feature_columns(route_dfs)
    log.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    route_names = list(route_dfs.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(route_names)
    n_val = max(1, int(len(route_names) * VAL_ROUTES_FRACTION))
    val_routes, train_routes = route_names[:n_val], route_names[n_val:]
    log.info(f"Train routes: {train_routes}")
    log.info(f"Val routes:   {val_routes}")

    train_ds_routes = {r: route_dfs[r] for r in train_routes}
    norm_stats = compute_norm_stats(train_ds_routes, feature_cols)
    log.info(f"Computed normalization stats from {len(train_routes)} train routes")

    train_ds = TrajectoryWindowDataset(train_ds_routes, feature_cols, seed=SEED, norm_stats=norm_stats)
    val_ds = TrajectoryWindowDataset({r: route_dfs[r] for r in val_routes}, feature_cols, seed=SEED + 1, norm_stats=norm_stats)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    n_features = len(feature_cols) + 1  # +1 for mask_flag
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
                "feature_cols": feature_cols,
                "n_features": n_features,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "epoch": epoch,
                "val_loss": val_loss,
                "norm_stats": norm_stats,
            }, CHECKPOINT_PATH)
            log.info(f"  -> new best, saved checkpoint to {CHECKPOINT_PATH}")

    log.info(f"\nDone. Best val loss: {best_val_loss:.5f}")
    log.info("Note: val loss above is in normalized lat/lon units, not meters.")
    log.info("Run evaluate.py for RMSE in meters via haversine distance.")


if __name__ == "__main__":
    main()
