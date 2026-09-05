"""
evaluate.py

Loads a trained TrajectoryLSTM checkpoint (from train.py) and reports
imputation RMSE in real-world meters (via haversine distance), computed
only at masked (imputed) GPS positions on the held-out validation routes.

This deliberately reuses train.py's own route-split, dataset, and haversine
logic (imported, not re-typed) so the val set here is guaranteed identical
to the one train.py held out during training -- as long as the merged CSVs
in MERGED_DIR haven't changed since that training run.

Also computes a per-route RMSE breakdown so a few outlier/degenerate routes
(e.g. SUMO public-transit "pt_subway" entries with teleporting positions)
don't silently dominate the aggregate metric.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import TrajectoryLSTM
from train import (
    CHECKPOINT_PATH as DEFAULT_CHECKPOINT_PATH,
    MERGED_DIR as DEFAULT_MERGED_DIR,
    SEED,
    VAL_ROUTES_FRACTION,
    TrajectoryWindowDataset,
    haversine_m,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained TrajectoryLSTM checkpoint: RMSE in meters on held-out val routes."
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH,
                    help=f"Path to checkpoint saved by train.py (default: {DEFAULT_CHECKPOINT_PATH})")
    p.add_argument("--merged-dir", type=Path, default=DEFAULT_MERGED_DIR,
                    help=f"Directory of merged route CSVs (default: {DEFAULT_MERGED_DIR})")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--baseline", type=float, default=85.0,
                    help="Baseline RMSE in meters to compare against (default: 85.0)")
    p.add_argument("--device", type=str, default=None,
                    help="cuda / cpu (default: auto-detect)")
    p.add_argument("--top-n", type=int, default=20,
                    help="Number of worst routes to print in the per-route breakdown (default: 20)")
    p.add_argument("--skip-per-route", action="store_true",
                    help="Skip the per-route breakdown (aggregate metrics only, faster)")
    return p.parse_args()


def load_val_routes(merged_dir: Path) -> tuple[list[str], list[str], dict[str, pd.DataFrame]]:
    """Reproduces train.py's exact route-level train/val split (same SEED,
    same shuffle, same fraction) so eval uses routes the model never trained on."""
    route_files = sorted(merged_dir.glob("*.csv"))
    if not route_files:
        raise FileNotFoundError(f"No merged route CSVs found in {merged_dir}. Run data_merger.py first.")

    route_dfs = {f.stem: pd.read_csv(f, parse_dates=["time"]) for f in route_files}

    route_names = list(route_dfs.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(route_names)
    n_val = max(1, int(len(route_names) * VAL_ROUTES_FRACTION))
    val_routes, train_routes = route_names[:n_val], route_names[n_val:]
    return val_routes, train_routes, route_dfs


def invert_norm(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def evaluate_routes(route_names, route_dfs, feature_cols, norm_stats, model, device,
                     batch_size, seed_offset=1) -> np.ndarray:
    """Runs the model over the given routes and returns the flat array of
    per-point haversine errors (meters) at masked positions. Returns an
    empty array if the routes produced no windows/masked points."""
    ds = TrajectoryWindowDataset(
        {r: route_dfs[r] for r in route_names},
        feature_cols,
        seed=SEED + seed_offset,
        norm_stats=norm_stats,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    lat_mean, lat_std = norm_stats["lat"]
    lon_mean, lon_std = norm_stats["lon"]

    dists = []
    with torch.no_grad():
        for features, target, mask in loader:
            features = features.to(device)
            pred = model(features, target=None)  # no teacher forcing at eval

            pred = pred.cpu().numpy()
            target = target.numpy()
            mask = mask.numpy().astype(bool)

            pred_lat = invert_norm(pred[..., 0], lat_mean, lat_std)
            pred_lon = invert_norm(pred[..., 1], lon_mean, lon_std)
            true_lat = invert_norm(target[..., 0], lat_mean, lat_std)
            true_lon = invert_norm(target[..., 1], lon_mean, lon_std)

            d = haversine_m(true_lat, true_lon, pred_lat, pred_lon)  # (batch, seq_len), meters
            dists.append(d[mask])

    return np.concatenate(dists) if dists else np.array([])
def discover_feature_columns(all_routes: dict[str, pd.DataFrame]) -> list[str]:
    """lat/lon/alt/speed only -- SUMO routes have no accel/gyro sensors,
    so including them meant those columns were zero-filled for the vast
    majority of training data, making the model unreliable on the small
    number of real I2WDD routes that do have genuine sensor readings."""
    return ["lat", "lon", "alt", "speed_2d", "speed_3d"]


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    log.info(f"Using device: {device}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint}. "
            f"Check where train.py actually saved it -- train.py's CHECKPOINT_PATH "
            f"points at ~/lstm-trajectory-imputation/checkpoints/model.pt, which may "
            f"not be the same repo you're running evaluate.py from."
        )

    ckpt = torch.load(args.checkpoint, map_location=device)
    feature_cols = ckpt["feature_cols"]
    norm_stats = ckpt["norm_stats"]
    n_features = ckpt["n_features"]
    hidden_size = ckpt["hidden_size"]
    num_layers = ckpt["num_layers"]

    log.info(f"Loaded checkpoint from epoch {ckpt.get('epoch')} "
              f"(normalized val loss at save time: {ckpt.get('val_loss'):.5f})")

    model = TrajectoryLSTM(n_features=n_features, hidden_size=hidden_size, num_layers=num_layers).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_routes, train_routes, route_dfs = load_val_routes(args.merged_dir)
    log.info(f"Val routes (held out during training): {val_routes}")

    # --- Aggregate metric ---
    all_dists_m = evaluate_routes(val_routes, route_dfs, feature_cols, norm_stats,
                                   model, device, args.batch_size, seed_offset=1)

    if len(all_dists_m) == 0:
        log.warning("No masked points evaluated -- check windowing/masking logic.")
        return

    rmse_m = float(np.sqrt(np.mean(all_dists_m ** 2)))
    mae_m = float(np.mean(all_dists_m))
    median_m = float(np.median(all_dists_m))

    print()
    print(f"Evaluated on {len(all_dists_m)} masked GPS points across {len(val_routes)} held-out routes")
    print(f"RMSE:   {rmse_m:.2f} m")
    print(f"MAE:    {mae_m:.2f} m")
    print(f"Median: {median_m:.2f} m")
    print()
    delta = rmse_m - args.baseline
    pct = (delta / args.baseline) * 100
    direction = "worse" if delta > 0 else "better"
    print(f"Baseline: {args.baseline:.2f} m  ->  New: {rmse_m:.2f} m  "
          f"({abs(pct):.1f}% {direction})")

    # --- Per-route breakdown, to catch outlier routes skewing RMSE ---
    if args.skip_per_route:
        return

    print("\nComputing per-route RMSE breakdown...")
    route_stats = []
    for r in val_routes:
        d = evaluate_routes([r], route_dfs, feature_cols, norm_stats,
                             model, device, args.batch_size, seed_offset=1)
        if len(d) == 0:
            continue
        r_rmse = float(np.sqrt(np.mean(d ** 2)))
        r_median = float(np.median(d))
        route_stats.append((r, r_rmse, r_median, len(d)))

    route_stats.sort(key=lambda x: -x[1])

    print(f"\nTop {args.top_n} worst routes by RMSE:")
    print(f"{'RMSE (m)':>12}  {'Median (m)':>12}  {'n pts':>7}  route")
    for name, r_rmse, r_median, n in route_stats[:args.top_n]:
        print(f"{r_rmse:12.2f}  {r_median:12.2f}  {n:7d}  {name}")


if __name__ == "__main__":
    main()