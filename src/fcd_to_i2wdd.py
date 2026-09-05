"""
fcd_to_i2wdd.py
Converts a SUMO --fcd-output.geo=true XML file into I2WDD-style route
directories consumable by data_merger.py. Each vehicle trajectory becomes
one route: <out_root>/<route_name>/IMU/GPS_1.csv

Only GPS is produced (SUMO has no IMU data) -- data_merger.py already
forward/backward-fills missing accel/gyro columns, so these routes are
treated the same as I2WDD gaps.

Usage:
    python src/fcd_to_i2wdd.py --fcd path/to/fcd_output.xml \
        --location gachibowli --run 1 \
        --out-root ~/lstm-trajectory-imputation/data/raw/i2wdd
"""
import argparse
import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

GPS_DATE_COL = "date"
GPS_LAT_COL = "GPS (Lat.) [deg]"
GPS_LON_COL = "GPS (Long.) [deg]"
GPS_ALT_COL = "GPS (Alt.) [m]"
GPS_SPEED2D_COL = "GPS (2D speed) [m/s]"
GPS_SPEED3D_COL = "GPS (3D speed) [m/s]"

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def convert(fcd_path: Path, out_root: Path, location: str, run: str, min_points: int):
    vehicles = {}  # veh_id -> list of (t, lat, lon, speed)

    # iterparse to handle large (100MB+) files without loading fully into memory
    context = ET.iterparse(fcd_path, events=("start",))
    for _, elem in context:
        if elem.tag == "timestep":
            t = float(elem.get("time"))
        elif elem.tag == "vehicle":
            veh_id = elem.get("id")
            lon = float(elem.get("x"))
            lat = float(elem.get("y"))
            speed = float(elem.get("speed"))
            vehicles.setdefault(veh_id, []).append((t, lat, lon, speed))
            elem.clear()

    written = 0
    for veh_id, rows in vehicles.items():
        if len(rows) < min_points:
            continue
        route_name = f"sumo_{location}_run{run}_{veh_id}"
        route_dir = out_root / route_name / "IMU"
        route_dir.mkdir(parents=True, exist_ok=True)
        out_path = route_dir / "GPS_1.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                GPS_DATE_COL, GPS_LAT_COL, GPS_LON_COL,
                GPS_ALT_COL, GPS_SPEED2D_COL, GPS_SPEED3D_COL,
            ])
            for t, lat, lon, speed in rows:
                ts = BASE_TIME + timedelta(seconds=t)
                date_str = ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"
                writer.writerow([date_str, lat, lon, "", speed, ""])
        written += 1

    print(f"[fcd_to_i2wdd] Wrote {written} routes to {out_root} "
          f"(skipped {len(vehicles) - written} with < {min_points} points)")


def main():
    p = argparse.ArgumentParser(description="Convert SUMO FCD geo output to I2WDD route directories")
    p.add_argument("--fcd", required=True, type=Path, help="Path to fcd_output.xml (with --fcd-output.geo true)")
    p.add_argument("--location", required=True, help="Location label, e.g. gachibowli")
    p.add_argument("--run", required=True, help="Run label/number, e.g. 1")
    p.add_argument("--out-root", required=True, type=Path, help="I2WDD raw data root (DATA_ROOT in data_merger.py)")
    p.add_argument("--min-points", type=int, default=10, help="Skip vehicles with fewer than this many FCD points")
    args = p.parse_args()
    convert(args.fcd, args.out_root, args.location, args.run, args.min_points)


if __name__ == "__main__":
    main()
