import glob, os
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET

ROOT = os.path.expanduser("~/datasets/i2wdd/i2wdd")
NS = "{http://www.topografix.com/GPX/1/1}"
HZ, WIN, STRIDE, MAX_GAP = 5, 30, 15, 1.0


def _read_csv(p):
    d = pd.read_csv(p)
    ts = pd.to_datetime(d["date"], utc=True)
    t = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    return pd.DataFrame({"t": t, "lat": d["GPS (Lat.) [deg]"],
                         "lon": d["GPS (Long.) [deg]"], "fix": d["fix"]})


def _read_gpx(p):
    rows = []
    for pt in ET.parse(p).getroot().iter(NS + "trkpt"):
        fx = pt.findtext(NS + "fix", "3d")
        rows.append((pd.Timestamp(pt.findtext(NS + "time")).timestamp(),
                     float(pt.get("lat")), float(pt.get("lon")),
                     3 if fx == "3d" else 2))
    return pd.DataFrame(rows, columns=["t", "lat", "lon", "fix"])


def _segments(df):
    df = df[df.fix >= 3].sort_values("t").drop_duplicates("t")
    t, lat, lon = df.t.values, df.lat.values, df.lon.values
    lat0, lon0 = lat[0], lon[0]
    x = (lon - lon0) * np.cos(np.radians(lat0)) * 111320.0
    y = (lat - lat0) * 110540.0
    cuts = np.where(np.diff(t) > MAX_GAP)[0] + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        if t[b - 1] - t[a] < WIN / HZ:
            continue
        g = np.arange(t[a], t[b - 1], 1.0 / HZ)
        yield np.stack([np.interp(g, t[a:b], x[a:b]),
                        np.interp(g, t[a:b], y[a:b])], 1).astype(np.float32)


def load_i2wdd():
    """Returns list of dicts: track_id, source, xy (T,2) metres at 5 Hz, one per gap-free segment."""
    out = []
    for folder in sorted(glob.glob(f"{ROOT}/*/IMU")):
        date = os.path.basename(os.path.dirname(folder))
        csvs = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{folder}/GPS_*.csv")}
        gpxs = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{folder}/GPS_*.gpx")}
        for stem in sorted(set(csvs) | set(gpxs)):
            df = _read_csv(csvs[stem]) if stem in csvs else _read_gpx(gpxs[stem])
            for i, xy in enumerate(_segments(df)):
                out.append(dict(track_id=f"{date}/{stem}/seg{i}",
                                source=f"{date}/{stem}", xy=xy))
    return out


def windows(tracks):
    for tr in tracks:
        xy = tr["xy"]
        for s in range(0, len(xy) - WIN + 1, STRIDE):
            yield tr["source"], xy[s:s + WIN]


if __name__ == "__main__":
    tr = load_i2wdd()
    print("segments:", len(tr))
    for src in sorted({t["source"] for t in tr}):
        L = [len(t["xy"]) / HZ for t in tr if t["source"] == src]
        print(f"{src:28s} segs={len(L):3d} total={sum(L):7.1f}s")
    print("windows:", sum(1 for _ in windows(tr)))