"""
sumo_runner.py

Runs a SUMO simulation on an existing .net.xml road network to generate
synthetic multi-vehicle trajectory data. This addresses the core limitation
of the I2WDD dataset (single ego-vehicle only) by producing multi-agent
interaction data that can be merged into the same training pipeline used
for I2WDD (data_merger.py -> train.py).

Requirements:
    pip install traci sumolib
    SUMO must be installed and SUMO_HOME set as an environment variable.

Usage:
    python sumo_runner.py --net path/to/network.net.xml --out sumo_sim/output --steps 3600

Output:
    One CSV per simulated route, written to --out, with columns:
        timestamp, vehicle_id, lat, lon, speed
    IMU columns (accl_x/y/z, gyro_x/y/z) are NOT produced by SUMO and are
    intentionally omitted here -- data_merger.py / train.py already handle
    missing sensor columns via forward/backward-fill, so these rows will
    be treated the same way as I2WDD gaps.
"""

import os
import sys
import csv
import argparse
import subprocess
from pathlib import Path

# --- SUMO_HOME setup -------------------------------------------------------
if "SUMO_HOME" in os.environ:
    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    sys.exit(
        "ERROR: please set SUMO_HOME (e.g. export SUMO_HOME=/usr/share/sumo) "
        "before running this script."
    )

import sumolib  # noqa: E402
import traci  # noqa: E402


def generate_random_routes(net_file: str, route_file: str, num_vehicles: int, seed: int = 42):
    """
    If no .rou.xml exists yet, generate one using SUMO's randomTrips.py tool.
    This creates plausible multi-vehicle traffic demand on the given network.
    """
    random_trips_script = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")
    trips_file = route_file.replace(".rou.xml", ".trips.xml")

    cmd = [
        sys.executable,
        random_trips_script,
        "-n", net_file,
        "-o", trips_file,
        "-r", route_file,
        "-e", str(num_vehicles),
        "--seed", str(seed),
        "--validate",
    ]
    print(f"[sumo_runner] Generating random routes: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_simulation(net_file: str, route_file: str, out_dir: Path, steps: int, gui: bool = False):
    """
    Runs the SUMO simulation via TraCI, recording per-vehicle trajectory
    data (lat, lon, speed) at every simulation step, then writes one CSV
    per vehicle to out_dir in the schema expected by data_merger.py.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    net = sumolib.net.readNet(net_file)

    sumo_binary = "sumo-gui" if gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-n", net_file,
        "-r", route_file,
        "--step-length", "1.0",
        "--no-warnings", "true",
    ]

    traci.start(sumo_cmd)

    # vehicle_id -> list of (timestamp, lat, lon, speed)
    trajectories = {}

    step = 0
    try:
        while step < steps:
            traci.simulationStep()
            timestamp = traci.simulation.getTime()

            for veh_id in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(veh_id)
                lon, lat = net.convertXY2LonLat(x, y)
                speed = traci.vehicle.getSpeed(veh_id)

                trajectories.setdefault(veh_id, []).append(
                    (timestamp, lat, lon, speed)
                )

            step += 1
    finally:
        traci.close()

    # Write one CSV per vehicle -- treat each as a synthetic "route"
    written = 0
    for veh_id, rows in trajectories.items():
        if len(rows) < 2:
            continue  # skip vehicles that barely appear (spawned/despawned instantly)

        out_path = out_dir / f"sumo_route_{veh_id}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "vehicle_id", "lat", "lon", "speed"])
            writer.writerows(rows)
        written += 1

    print(f"[sumo_runner] Wrote {written} trajectory files to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-vehicle trajectory data via SUMO")
    parser.add_argument("--net", required=True, help="Path to existing .net.xml network file")
    parser.add_argument("--route", default="sumo_sim/generated.rou.xml", help="Path to .rou.xml (generated if missing)")
    parser.add_argument("--out", default="sumo_sim/output", help="Output directory for trajectory CSVs")
    parser.add_argument("--steps", type=int, default=3600, help="Number of simulation steps (seconds) to run")
    parser.add_argument("--num-vehicles", type=int, default=50, help="Number of vehicles to generate if routes don't exist")
    parser.add_argument("--gui", action="store_true", help="Run with sumo-gui instead of headless sumo")
    args = parser.parse_args()

    route_path = Path(args.route)
    route_path.parent.mkdir(parents=True, exist_ok=True)

    if not route_path.exists():
        generate_random_routes(args.net, str(route_path), args.num_vehicles)
    else:
        print(f"[sumo_runner] Using existing route file: {route_path}")

    run_simulation(args.net, str(route_path), Path(args.out), args.steps, gui=args.gui)


if __name__ == "__main__":
    main()