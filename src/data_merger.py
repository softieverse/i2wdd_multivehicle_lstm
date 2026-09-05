"""
data_merger.py

Merges per-route I2WDD sensor files (GPS + accelerometer + gyroscope) into a
single time-aligned CSV per route, ready for the LSTM pipeline.

ASSUMPTIONS (review these against what Sir actually wants before trusting output):
  1. GPS timestamps ('date' column, e.g. "2025-01-10T09:09:22.574Z") are UTC.
     Accel/gyro timestamps ('timestamp' column, e.g. "2025-01-10 14:41:38+05:30")
     carry an explicit UTC offset. Both are converted to UTC before alignment.
  2. Sensor "position" (back / front / handlebar) is inconsistent across routes.
     This script merges ALL available positions as separately-prefixed columns
     (e.g. accel_back_x, accel_handlebar_x). Routes missing a position simply
     get NaN columns for it — nothing is dropped, nothing is invented.
  3. Accel/gyro are fixed-rate (10Hz downsampled). GPS is irregular. We resample
     everything onto the accel/gyro 10Hz time grid via merge_asof (nearest,
     backward), since accel/gyro are the denser signal.
  4. Multiple GPS segments per route (GPS_1.csv, GPS_2.csv, ...) are concatenated
     and sorted by time before merging, treated as one continuous trajectory.
  5. Routes with zero usable GPS CSVs (e.g. only .gpx present) are skipped with
     a warning, not silently dropped — check the log output.

Output: one CSV per route under data/processed/merged/<route>.csv
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DATA_ROOT = Path("~/lstm-trajectory-imputation/data/raw/i2wdd").expanduser()
OUT_DIR = Path("~/lstm-trajectory-imputation/data/processed/merged").expanduser()

GPS_LAT_COL = "GPS (Lat.) [deg]"
GPS_LON_COL = "GPS (Long.) [deg]"
GPS_ALT_COL = "GPS (Alt.) [m]"
GPS_SPEED2D_COL = "GPS (2D speed) [m/s]"
GPS_SPEED3D_COL = "GPS (3D speed) [m/s]"
GPS_DATE_COL = "date"

ACCEL_COLS = {
    "Accelerometer (x) [m/s²]": "accel_x",
    "Accelerometer (y) [m/s²]": "accel_y",
    "Accelerometer (z) [m/s²]": "accel_z",
}
GYRO_COLS = {
    "Gyroscope (x) [rad/s]": "gyro_x",
    "Gyroscope (y) [rad/s]": "gyro_y",
    "Gyroscope (z) [rad/s]": "gyro_z",
}

# matches e.g. "GPS_back_1.csv", "GPS_1.csv", "accl_downsampled_front_10hz.csv",
# "gyro_downsampled.csv"
POSITION_RE = re.compile(r"(back|front|handlebar)")


def detect_position(filename: str) -> str:
    """Return sensor mount position, or 'default' if the file doesn't tag one."""
    m = POSITION_RE.search(filename)
    return m.group(1) if m else "default"


def load_gps_gpx(route_dir: Path) -> pd.DataFrame | None:
    """Fallback GPS loader for routes with only .gpx files (no CSV).
    GPX has no speed columns, so speed_2d/speed_3d come back as NaN —
    keeps the merged schema consistent with the CSV path."""
    import xml.etree.ElementTree as ET

    gpx_files = sorted(route_dir.glob("IMU/GPS_*.gpx"))
    if not gpx_files:
        return None

    NS = {"gpx": "http://www.topografix.com/GPX/1/1"}
    frames = []
    for f in gpx_files:
        try:
            tree = ET.parse(f)
        except ET.ParseError as e:
            log.warning(f"  Skipping unparseable GPX file {f.name}: {e}")
            continue

        root = tree.getroot()
        # namespace can vary/be absent depending on how the file was exported;
        # try namespaced search first, fall back to no-namespace
        trkpts = root.findall(".//gpx:trkpt", NS) or root.findall(".//trkpt")
        rows = []
        for pt in trkpts:
            lat = pt.get("lat")
            lon = pt.get("lon")
            ele_el = pt.find("gpx:ele", NS) if pt.find("gpx:ele", NS) is not None else pt.find("ele")
            time_el = pt.find("gpx:time", NS) if pt.find("gpx:time", NS) is not None else pt.find("time")
            if lat is None or lon is None or time_el is None or time_el.text is None:
                continue
            rows.append({
                GPS_LAT_COL: float(lat),
                GPS_LON_COL: float(lon),
                GPS_ALT_COL: float(ele_el.text) if ele_el is not None and ele_el.text else np.nan,
                GPS_SPEED2D_COL: np.nan,
                GPS_SPEED3D_COL: np.nan,
                "date": time_el.text,
            })
        if not rows:
            log.warning(f"  No trackpoints extracted from {f.name}")
            continue
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["date"], utc=True)
        df = df.drop(columns=["date"])
        df = df.dropna(subset=["time"])
        frames.append(df)

    if not frames:
        return None

    gps = pd.concat(frames, ignore_index=True)
    log.info(f"  Loaded {len(gps)} GPS points from .gpx fallback ({len(gpx_files)} files)")
    return gps


def load_gps(route_dir: Path) -> pd.DataFrame | None:
    gps_files = sorted(route_dir.glob("IMU/GPS_*.csv"))
    if not gps_files:
        log.info(f"  No GPS CSVs found in {route_dir.name}, trying .gpx fallback")
        gpx_gps = load_gps_gpx(route_dir)
        if gpx_gps is None:
            log.warning(f"  No usable GPS data (CSV or GPX) in {route_dir.name}, skipping route")
            return None
        gpx_gps = gpx_gps.dropna(subset=[GPS_LAT_COL, GPS_LON_COL])
        gpx_gps = gpx_gps[gpx_gps[GPS_LAT_COL].between(-90, 90) & gpx_gps[GPS_LON_COL].between(-180, 180)]
        gpx_gps = gpx_gps[(gpx_gps[GPS_LAT_COL] != 0) | (gpx_gps[GPS_LON_COL] != 0)]
        gpx_gps = gpx_gps.sort_values("time").reset_index(drop=True)
        gpx_gps = gpx_gps.rename(columns={
            GPS_LAT_COL: "lat", GPS_LON_COL: "lon", GPS_ALT_COL: "alt",
            GPS_SPEED2D_COL: "speed_2d", GPS_SPEED3D_COL: "speed_3d",
        })
        return gpx_gps

    frames = []
    for f in gps_files:
        try:
            df = pd.read_csv(
                f,
                usecols=[GPS_DATE_COL, GPS_LAT_COL, GPS_LON_COL, GPS_ALT_COL,
                         GPS_SPEED2D_COL, GPS_SPEED3D_COL],
            )
        except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            log.warning(f"  Skipping unreadable GPS file {f.name}: {e}")
            continue
        df["time"] = pd.to_datetime(df[GPS_DATE_COL], utc=True)
        df = df.drop(columns=[GPS_DATE_COL])
        df = df.dropna(subset=["time"])
        frames.append(df)

    if not frames:
        return None

    gps = pd.concat(frames, ignore_index=True)
    gps = gps.dropna(subset=[GPS_LAT_COL, GPS_LON_COL])
    gps = gps[gps[GPS_LAT_COL].between(-90, 90) & gps[GPS_LON_COL].between(-180, 180)]
    gps = gps[(gps[GPS_LAT_COL] != 0) | (gps[GPS_LON_COL] != 0)]
    gps = gps.sort_values("time").reset_index(drop=True)
    gps = gps.rename(columns={
        GPS_LAT_COL: "lat", GPS_LON_COL: "lon", GPS_ALT_COL: "alt",
        GPS_SPEED2D_COL: "speed_2d", GPS_SPEED3D_COL: "speed_3d",
    })
    return gps


def load_imu_group(route_dir: Path, pattern: str, col_map: dict) -> dict[str, pd.DataFrame]:
    """Load accel or gyro files, grouped by mount position. Returns {position: df}."""
    files = sorted(route_dir.glob(f"IMU/{pattern}"))
    by_position = {}
    for f in files:
        pos = detect_position(f.name)
        try:
            df = pd.read_csv(f, usecols=["timestamp"] + list(col_map.keys()))
        except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            log.warning(f"  Skipping unreadable file {f.name}: {e}")
            continue
        df["time"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        df = df.drop(columns=["timestamp"]).rename(columns=col_map)
        n_before = len(df)
        df = df.dropna(subset=["time"])
        if len(df) < n_before:
            log.warning(f"  Dropped {n_before - len(df)} rows with unparseable timestamp in {f.name}")
        df = df.sort_values("time").reset_index(drop=True)
        by_position[pos] = df
    return by_position


def merge_route(route_dir: Path) -> pd.DataFrame | None:
    gps = load_gps(route_dir)
    if gps is None:
        return None

    accel_by_pos = load_imu_group(route_dir, "accl_downsampled*.csv", ACCEL_COLS)
    gyro_by_pos = load_imu_group(route_dir, "gyro_downsampled*.csv", GYRO_COLS)

    if not accel_by_pos and not gyro_by_pos:
        log.warning(f"  No accel/gyro files found in {route_dir.name}, GPS-only output")

    # Base timeline = accel if present, else gyro, else GPS itself
    base = None
    for pos, df in accel_by_pos.items():
        base = df[["time"]] if base is None else base
    if base is None:
        for pos, df in gyro_by_pos.items():
            base = df[["time"]] if base is None else base
    if base is None:
        base = gps[["time"]].copy()

    merged = base.sort_values("time").reset_index(drop=True)

    # Align GPS onto the base timeline (nearest, backward — GPS is sparser)
    merged = pd.merge_asof(
        merged, gps.sort_values("time"), on="time", direction="nearest",
        tolerance=pd.Timedelta("2s"),
    )

    # Fold in each accel/gyro position as its own prefixed columns
    for pos, df in accel_by_pos.items():
        renamed = df.rename(columns={c: f"{c}_{pos}" for c in ACCEL_COLS.values()})
        merged = pd.merge_asof(
            merged.sort_values("time"), renamed.sort_values("time"), on="time",
            direction="nearest", tolerance=pd.Timedelta("200ms"),
        )
    for pos, df in gyro_by_pos.items():
        renamed = df.rename(columns={c: f"{c}_{pos}" for c in GYRO_COLS.values()})
        merged = pd.merge_asof(
            merged.sort_values("time"), renamed.sort_values("time"), on="time",
            direction="nearest", tolerance=pd.Timedelta("200ms"),
        )

    merged["route"] = route_dir.name
    return merged


def main():
    route_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
    log.info(f"Found {len(route_dirs)} route directories")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, skipped = 0, 0

    for route_dir in route_dirs:
        log.info(f"Processing {route_dir.name}...")
        merged = merge_route(route_dir)
        if merged is None or merged.empty:
            log.warning(f"  -> skipped (no usable data)")
            skipped += 1
            continue
        out_path = OUT_DIR / f"{route_dir.name}.csv"
        merged.to_csv(out_path, index=False)
        log.info(f"  -> {len(merged)} rows, saved to {out_path}")
        ok += 1

    log.info(f"\nDone. {ok} routes merged, {skipped} skipped.")


if __name__ == "__main__":
    main()