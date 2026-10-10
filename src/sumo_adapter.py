import glob, os
import numpy as np, pandas as pd
from unified_loader import Track, latlon_to_xy, segment_on_jumps, resample, HZ, WIN_S, AGENT

SUMO_DS = 4  # new dataset id (hetrod 0, idd_mm 1, idd_ped 2, i2wdd 3)


def load_sumo_tracks(root="data/sumo/routes", hz=HZ):
    out = []
    for folder in sorted(glob.glob(os.path.join(os.path.expanduser(root), "sumo_*"))):
        name = os.path.basename(folder)
        csv = os.path.join(folder, "IMU", "GPS_1.csv")
        if not os.path.exists(csv):
            continue
        d = pd.read_csv(csv)
        t = pd.to_datetime(d["date"], utc=True)
        t = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
        xy = latlon_to_xy(d["GPS (Lat.) [deg]"].to_numpy(), d["GPS (Long.) [deg]"].to_numpy())
        kind = name.split("_")[-1].rstrip("0123456789")      # moto / car / bus
        agent = 1 if kind == "moto" else 0   # common_schema: vehicle 0, two_wheeler 1
        tr = Track(xy.astype(np.float32), t, agent, SUMO_DS, name)
        for seg in segment_on_jumps(tr, max_dt=0.5):
            seg = resample(seg, hz)
            if len(seg.t) >= 30:                # keep only tracks that fit a 10 s window
                out.append(seg)
    return out