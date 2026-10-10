import glob, os
import numpy as np
import pandas as pd

ROOT = os.path.expanduser(
    "~/Downloads/hetrod/hetroD-dataset-v1.0-without-Loc2/data/01_automated_processing"
)

def list_recordings(root=ROOT):
    return sorted(os.path.basename(p).split("_")[0]
                  for p in glob.glob(f"{root}/*_tracks.csv"))

def load_recording(prefix, root=ROOT, target_hz=5.0):
    """Common schema, resampled to target_hz (30Hz -> step 6, 25Hz -> step 5)."""
    tr = pd.read_csv(f"{root}/{prefix}_tracks.csv")
    meta = pd.read_csv(f"{root}/{prefix}_tracksMeta.csv")[["recordingId", "trackId", "class"]]
    rec = pd.read_csv(f"{root}/{prefix}_recordingMeta.csv").iloc[0]
    fps = float(rec["frameRate"])
    step = int(round(fps / target_hz))

    df = tr.merge(meta, on=["recordingId", "trackId"])
    df = df[df["frame"] % step == 0].copy()

    out = pd.DataFrame({
        "rec": df["recordingId"], "track": df["trackId"],
        "t": df["frame"] / fps,
        "x": df["xCenter"], "y": df["yCenter"],
        "vx": df["xVelocity"], "vy": df["yVelocity"],
        "heading_sin": np.sin(np.deg2rad(df["heading"])),
        "heading_cos": np.cos(np.deg2rad(df["heading"])),
        "width": df["width"], "length": df["length"],
        "cls": df["class"].replace({"trailer": "truck"}),
    })
    return out.sort_values(["track", "t"]).reset_index(drop=True), fps / step