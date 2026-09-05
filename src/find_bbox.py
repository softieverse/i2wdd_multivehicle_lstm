import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DATA_ROOT = Path("~/lstm-trajectory-imputation/data/raw/i2wdd").expanduser()
LAT_COL = "GPS (Lat.) [deg]"
LON_COL = "GPS (Long.) [deg]"
OUT_SUMMARY = Path("~/lstm-trajectory-imputation/data/processed/route_bounds.csv").expanduser()

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def load_gps_files(root: Path) -> tuple[list[pd.DataFrame], list[Path]]:
    gps_files = sorted(root.glob("*/IMU/GPS_*.csv"))
    log.info(f"Found {len(gps_files)} GPS files")
    if not gps_files:
        raise FileNotFoundError(f"No GPS files found under {root} — check path/glob pattern")

    frames, bad_files = [], []
    for f in gps_files:
        try:
            df = pd.read_csv(f, usecols=[LAT_COL, LON_COL])
        except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            bad_files.append((f, str(e)))
            continue

        df = df.dropna(subset=[LAT_COL, LON_COL])
        # sanity bounds — drop obviously corrupt GPS rows (e.g. 0,0 or out-of-range)
        df = df[df[LAT_COL].between(-90, 90) & df[LON_COL].between(-180, 180)]
        df = df[(df[LAT_COL] != 0) | (df[LON_COL] != 0)]
        if df.empty:
            continue

        df["route"] = f.parent.parent.name
        df["source_file"] = f.name
        frames.append(df)

    if bad_files:
        log.warning(f"Skipped {len(bad_files)} files with bad/missing columns:")
        for f, err in bad_files:
            log.warning(f"  - {f}: {err}")

    if not frames:
        raise RuntimeError("No valid GPS data collected from any file")

    return frames, bad_files

def per_route_stats(all_data: pd.DataFrame) -> pd.DataFrame:
    def route_metrics(g):
        dist = haversine_km(
            g[LAT_COL].values[:-1], g[LON_COL].values[:-1],
            g[LAT_COL].values[1:], g[LON_COL].values[1:],
        ).sum() if len(g) > 1 else 0.0
        return pd.Series({
            "n_points": len(g),
            "min_lat": g[LAT_COL].min(), "max_lat": g[LAT_COL].max(),
            "min_lon": g[LON_COL].min(), "max_lon": g[LON_COL].max(),
            "approx_distance_km": dist,
        })
    return all_data.groupby("route").apply(route_metrics, include_groups=False)

def main():
    frames, _ = load_gps_files(DATA_ROOT)
    all_data = pd.concat(frames, ignore_index=True)

    min_lat, max_lat = all_data[LAT_COL].min(), all_data[LAT_COL].max()
    min_lon, max_lon = all_data[LON_COL].min(), all_data[LON_COL].max()

    log.info(f"\nTotal valid GPS points: {len(all_data)}")
    log.info(f"Routes covered: {all_data['route'].nunique()}")
    log.info(f"\nBounding box:")
    log.info(f"  Latitude:  {min_lat:.6f} to {max_lat:.6f}")
    log.info(f"  Longitude: {min_lon:.6f} to {max_lon:.6f}")

    route_bounds = per_route_stats(all_data)
    log.info("\nPer-route bounding boxes & distances:")
    log.info(route_bounds.to_string())

    # flag routes with suspiciously few points or near-zero distance (likely bad recordings)
    suspects = route_bounds[(route_bounds["n_points"] < 50) | (route_bounds["approx_distance_km"] < 0.05)]
    if not suspects.empty:
        log.warning(f"\nSuspect routes (too few points or ~stationary):\n{suspects.to_string()}")

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    route_bounds.to_csv(OUT_SUMMARY)
    log.info(f"\nSaved per-route summary to {OUT_SUMMARY}")

if __name__ == "__main__":
    main()