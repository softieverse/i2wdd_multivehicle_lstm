"""
Quick sanity check: plot the GPS trajectory of a merged route to eyeball
whether it looks like a real path (continuous, no teleporting) rather than
scrambled/misaligned points.

Usage: python plot_check.py <route_name>
e.g.   python plot_check.py 10_01_25a
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MERGED_DIR = Path("~/lstm-trajectory-imputation/data/processed/merged").expanduser()


def main():
    route = sys.argv[1] if len(sys.argv) > 1 else "10_01_25a"
    path = MERGED_DIR / f"{route}.csv"
    if not path.exists():
        print(f"No merged file found at {path}")
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["time"])
    print(f"Loaded {len(df)} rows for {route}")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    print(f"Lat range: {df['lat'].min():.6f} to {df['lat'].max():.6f}")
    print(f"Lon range: {df['lon'].min():.6f} to {df['lon'].max():.6f}")
    print(f"NaN counts:\n{df.isna().sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(df["lon"], df["lat"], "-", linewidth=0.5, alpha=0.7)
    axes[0].scatter(df["lon"].iloc[0], df["lat"].iloc[0], c="green", label="start", zorder=5)
    axes[0].scatter(df["lon"].iloc[-1], df["lat"].iloc[-1], c="red", label="end", zorder=5)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title(f"{route}: GPS trajectory")
    axes[0].legend()
    axes[0].axis("equal")

    if "accel_x_back" in df.columns:
        axes[1].plot(df["time"], df["accel_x_back"], linewidth=0.5, label="accel_x_back")
    elif "accel_x_default" in df.columns:
        axes[1].plot(df["time"], df["accel_x_default"], linewidth=0.5, label="accel_x_default")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Accel (m/s^2)")
    axes[1].set_title(f"{route}: accel_x over time")
    axes[1].legend()

    plt.tight_layout()
    out_path = MERGED_DIR / f"{route}_check.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()