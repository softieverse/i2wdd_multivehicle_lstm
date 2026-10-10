import pickle, sys, zlib, collections
import numpy as np
sys.path.insert(0, "src")
from sumo_adapter import load_sumo_tracks, SUMO_DS

sp = lambda k: "test" if zlib.crc32(k.encode()) % 100 < 10 else "val" if zlib.crc32(k.encode()) % 100 < 20 else "train"
out = [dict(ds=SUMO_DS, agent=int(s.agent), split=sp(s.group), xy=s.xy.astype(np.float32), id=s.group, source=s.group) for s in load_sumo_tracks()]
pickle.dump(out, open("data/processed/sumo_tracks.pkl", "wb"))
print(len(out), collections.Counter((t["agent"], t["split"]) for t in out))