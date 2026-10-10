"""
evaluate_pedestrian.py

Loads the trained pedestrian checkpoint and reports RMSE in raw pixel
(Euclidean) distance on the val split, using the same track-level split
and windowing/masking logic as train_pedestrian.py.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import TrajectoryLSTM
from train_pedestrian import (
    MERGED_DIR,
    CHECKPOINT_PATH,
    VAL_ROUTES_FRACTION,
    BATCH_SIZE,
    SEED,
    FEATURE_COLS,
    PedestrianWindowDataset,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def pixel_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               x_stats: tuple[float, float], y_stats: tuple[float, float]) -> float:
    """
    Denormalizes pred/target back to raw pixel coords, then computes
    Euclidean RMSE at masked positions only.
    """
    x_mean, x_std = x_stats
    y_mean, y_std = y_stats

    pred_px = pred.clone()
    target_px = target.clone()
    pred_px[..., 0] = pred_px[..., 0] * x_std + x_mean
    pred_px[..., 1] = pred_px[..., 1] * y_std + y_mean
    target_px[..., 0] = target_px[..., 0] * x_std + x_mean
    target_px[..., 1] = target_px[..., 1] * y_std + y_mean

    mask = mask.bool()
    sq_err = ((pred_px - target_px) ** 2).sum(dim=-1)  # (batch, seq_len)
    sq_err = sq_err[mask]  # only masked/imputed positions

    if sq_err.numel() == 0:
        return float("nan")
    return torch.sqrt(sq_err.mean()).item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    log.info(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f}")

    norm_stats = ckpt["norm_stats"]
    feature_cols = ckpt["feature_cols"]

    model = TrajectoryLSTM(
        n_features=ckpt["n_features"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Reproduce the exact same train/val track split as training (same SEED)
    track_files = sorted(MERGED_DIR.glob("*.csv"))
    if not track_files:
        raise FileNotFoundError(f"No pedestrian track CSVs found in {MERGED_DIR}.")

    track_dfs = {f.stem: pd.read_csv(f) for f in track_files}
    track_names = list(track_dfs.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(track_names)
    n_val = max(1, int(len(track_names) * VAL_ROUTES_FRACTION))
    val_tracks = track_names[:n_val]
    log.info(f"Evaluating on {len(val_tracks)} val tracks (same split as training)")

    val_ds_tracks = {t: track_dfs[t] for t in val_tracks}
    val_ds = PedestrianWindowDataset(val_ds_tracks, feature_cols, seed=SEED + 1, norm_stats=norm_stats)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    x_stats = norm_stats["x"]
    y_stats = norm_stats["y"]

    all_rmse = []
    with torch.no_grad():
        for features, target, mask in val_loader:
            features, target, mask = features.to(device), target.to(device), mask.to(device)
            pred = model(features, target=None)  # no teacher forcing at eval
            batch_rmse = pixel_rmse(pred, target, mask, x_stats, y_stats)
            if not np.isnan(batch_rmse):
                all_rmse.append(batch_rmse)

    final_rmse = float(np.mean(all_rmse)) if all_rmse else float("nan")
    log.info(f"\nFinal RMSE (pixels, Euclidean, masked positions only): {final_rmse:.3f}")

    # Save so you don't lose it again if the terminal closes
    results_path = Path("checkpoints/pedestrian_eval_results.txt")
    with open(results_path, "w") as f:
        f.write(f"checkpoint_epoch: {ckpt['epoch']}\n")
        f.write(f"val_loss_normalized: {ckpt['val_loss']:.5f}\n")
        f.write(f"rmse_pixels: {final_rmse:.5f}\n")
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
