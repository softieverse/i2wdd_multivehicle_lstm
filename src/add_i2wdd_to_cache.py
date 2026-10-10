import pickle
from common_schema import DATASET
from unified_data import load_i2wdd

P = "data/processed/unified_tracks.pkl"
tracks = pickle.load(open(P, "rb"))
tracks = [t for t in tracks if t["ds"] != DATASET["i2wdd"]]
new = load_i2wdd()
tracks += new
pickle.dump(tracks, open(P, "wb"))
from collections import Counter
print(len(tracks), Counter((t["ds"], t["split"]) for t in tracks))