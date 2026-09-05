import sys
sys.path.insert(0, "src")
import pandas as pd
from pathlib import Path
from train import discover_feature_columns

MERGED_DIR = Path.home() / "lstm-trajectory-imputation" / "data" / "processed" / "merged"
df = pd.read_csv(MERGED_DIR / "28_12_24.csv", parse_dates=["time"])
good = pd.read_csv(MERGED_DIR / "sumo_gachibowli_run1_veh65.csv", parse_dates=["time"])

feature_cols = discover_feature_columns({"28_12_24": df})
print("Feature columns:", feature_cols)

print("\n--- 28_12_24 features ---")
print(df[feature_cols].describe())

print("\n--- sumo_gachibowli_run1_veh65 features ---")
print(good[feature_cols].describe())

print("\n28_12_24 time deltas (s):")
print(df["time"].diff().dt.total_seconds().describe())

print("\nSUMO route time deltas (s):")
print(good["time"].diff().dt.total_seconds().describe())
