import matplotlib.pyplot as plt
from hetrod_loader import list_recordings, load_recording

recs = list_recordings()
print("recordings:", recs)

for p in recs:
    df, hz = load_recording(p)
    n = df.groupby("track").size()
    print(f"\n[{p}] rows={len(df)} tracks={df.track.nunique()} hz={hz}")
    print("tracks per class:\n", df.groupby("cls").track.nunique().to_string())
    print("track length (steps): min/median/max =", n.min(), n.median(), n.max())
    print("x range:", df.x.min(), df.x.max(), " y range:", df.y.min(), df.y.max())

df, _ = load_recording(recs[0])
plt.figure(figsize=(8, 8))
for tid, g in list(df.groupby("track"))[:60]:
    plt.plot(g.x, g.y, lw=0.8)
plt.axis("equal"); plt.title(f"recording {recs[0]}: 60 tracks")
plt.savefig("hetrod_sanity.png", dpi=120)