"""finetune_video.py -- fine-tune v4 on tracks extracted from Sir's traffic videos (+ replay of 4 real datasets)."""
import argparse, copy, glob, os, pickle, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_schema as cs

VIDEO = 5                         # new dataset id (4 is left free for SUMO)
cs.DATASET["video"] = VIDEO
cs.DATASET_NAMES[VIDEO] = "video"
cs.SCALE[VIDEO] = 100.0           # pixels, same scale as IDD-PeD
cs.UNIT[VIDEO] = "px"
cs.STRIDE_BY_DS[VIDEO] = 5
cs.N_DATASETS = 6                 # must be set BEFORE model_unified is imported

from model_unified import UnifiedImputer, masked_loss
import unified_data as ud
from unified_data import UnifiedWindows

REAL = [0, 1, 2, 3]
KEYS = ("ds", "agent", "split", "xy")
CLS2AGENT = {0: 2, 1: 1, 3: 1, 2: 0, 5: 0, 7: 0}   # person->ped(2), bicycle/motorcycle->two_wheeler(1), car/bus/truck->vehicle(0)
SPLIT_OF = {"sb1": "train", "sb2": "train", "sb3": "train", "sb4": "val", "sb5": "test"}


def load_video_tracks(vdir, max_dt=0.85):
    out, frame_files = [], 0
    for f in sorted(glob.glob(os.path.join(vdir, "*.csv"))):
        vid, _, cls = os.path.basename(f)[:-4].split("_")
        agent = CLS2AGENT.get(int(cls.replace("cls", "")), 0)
        d = pd.read_csv(f).dropna(subset=["t", "x", "y"]).sort_values("t").drop_duplicates("t")
        if len(d) < 2:
            continue
        t, xy = d["t"].to_numpy(float), d[["x", "y"]].to_numpy(float)
        if np.median(np.diff(t)) > 0.5:            # t is in frames, not seconds -> 5 fps
            t = t / 5.0
            frame_files += 1
        cuts = np.where(np.diff(t) > max_dt)[0] + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
            if b - a < 2:
                continue
            tt, p = t[a:b], xy[a:b]
            n = int(np.floor((tt[-1] - tt[0]) * 5 + 1e-6)) + 1
            if n < cs.WINDOW:
                continue
            g = tt[0] + np.arange(n) / 5.0
            grid = np.stack([np.interp(g, tt, p[:, k]) for k in (0, 1)], 1).astype(np.float32)
            out.append(dict(ds=VIDEO, agent=agent, split=SPLIT_OF.get(vid, "train"), xy=grid))
    print(f"video segments: {len(out)} ({frame_files} files had t in frames)")
    return out


@torch.no_grad()
def evaluate(models, loader, dev):
    for m in models.values():
        m.eval()
    acc = {}
    for b in loader:
        x, y, obs = b["x"].to(dev), b["y"].to(dev), b["obs"].to(dev)
        ag, dsb = b["agent"].to(dev), b["ds"].to(dev)
        miss = 1.0 - obs
        sc2 = b["scale"].to(dev).view(-1, 1) ** 2
        se = lambda p: (((p - y) ** 2).sum(-1) * sc2 * miss).sum(1)
        e = {"linear": se(x[:, :, 3:5]), "hermite": se(x[:, :, 5:7]), "n": miss.sum(1)}
        for k, m in models.items():
            e[k] = se(m(x, ag, dsb))
        for d in dsb.unique().tolist():
            s = dsb == d
            a = acc.setdefault(d, {k: 0.0 for k in e})
            for k in e:
                a[k] += e[k][s].sum().item()
    return {d: {k: (float(np.sqrt(v / max(a["n"], 1))) if k != "n" else v) for k, v in a.items()}
            for d, a in acc.items()}


def show(title, res):
    print(title)
    for d in sorted(res):
        r = res[d]
        print(f"  {cs.DATASET_NAMES[d]:8s} v4 {r['v4']:8.3f} | v7 {r['v7']:8.3f} | hermite {r['hermite']:8.3f} | "
              f"linear {r['linear']:8.3f} {cs.UNIT[d]} | v7/hermite {r['v7'] / r['hermite']:.3f}")


def real_ratio(res, key):
    return float(np.mean([res[d][key] / res[d]["hermite"] for d in REAL if d in res]))


def vid_ratio(res, key):
    return res[VIDEO][key] / res[VIDEO]["hermite"]


def eval_fixed_gap(models, tracks, split, L, dev):
    """Video windows with ONE contiguous gap of L steps (10 = 2 s, 15 = 3 s)."""
    orig = ud.sample_mask
    def fixed(window, rng):
        obs = np.ones(window, bool)
        lo, hi = window // 5, window - window // 5 - L
        s = int(rng.integers(lo, hi + 1))
        obs[s:s + L] = False
        return obs
    ud.sample_mask = fixed
    try:
        ds = UnifiedWindows([t for t in tracks if t["ds"] == VIDEO], split)
        loader = DataLoader(ds, batch_size=256)
        b0 = next(iter(loader))
        print(f"  [{split} | gap {L} steps] windows {len(ds)} | missing per window (should be [{L}]): "
              f"{sorted(set((1 - b0['obs']).sum(1).int().tolist()))}")
        return evaluate(models, loader, dev)
    finally:
        ud.sample_mask = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="unified_tracks.pkl")
    ap.add_argument("--video_dir", default="../video_tracks/csv")
    ap.add_argument("--v4", default="unified_v4.pt")
    ap.add_argument("--out", default="unified_v7_video.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--per_epoch", type=int, default=30000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--video_boost", type=float, default=2.0)
    ap.add_argument("--emb_only", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tracks = [{k: t[k] for k in KEYS} for t in pickle.load(open(a.cache, "rb"))] + load_video_tracks(a.video_dir)
    train, val, test = (UnifiedWindows(tracks, s) for s in ("train", "val", "test"))
    for n, d_ in (("train", train), ("val", val), ("test", test)):
        print(n, len(d_), {cs.DATASET_NAMES[d]: int((np.asarray(d_.ds) == d).sum()) for d in np.unique(d_.ds)})

    model = UnifiedImputer(hidden=a.hidden)
    sd = model.state_dict()
    for k, v in torch.load(a.v4, map_location="cpu").items():
        if k == "ds_emb.weight":
            sd[k][:v.shape[0]] = v
            sd[k][v.shape[0]:] = v.mean(0)       # new rows (4, 5) start as the mean of the 4 old ones
        else:
            sd[k] = v
    model.load_state_dict(sd)
    model.to(dev)
    base = copy.deepcopy(model).eval()           # v4 (on the video row it uses the neutral new embedding)

    if a.emb_only:
        for n_, p_ in model.named_parameters():
            p_.requires_grad = (n_ == "ds_emb.weight")
        model.ds_emb.weight.register_hook(lambda g: torch.cat([torch.zeros_like(g[:VIDEO]), g[VIDEO:]]))
    dsarr = np.asarray(train.ds)
    cnt = np.bincount(dsarr, minlength=6)
    w = 1.0 / cnt[dsarr]
    w[dsarr == VIDEO] *= a.video_boost
    sampler = WeightedRandomSampler(w, num_samples=a.per_epoch, replacement=True)
    tl = DataLoader(train, batch_size=a.bs, sampler=sampler, drop_last=True, num_workers=2, persistent_workers=True)
    vl, sl = DataLoader(val, batch_size=256), DataLoader(test, batch_size=256)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    models = {"v4": base, "v7": model}

    show("VAL before fine-tuning", evaluate(models, vl, dev))
    best = float("inf")
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0; nb = 0
        for b in tl:
            x, y, obs = b["x"].to(dev), b["y"].to(dev), b["obs"].to(dev)
            loss = masked_loss(model(x, b["agent"].to(dev), b["ds"].to(dev)), y, obs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        res = evaluate(models, vl, dev)
        score = vid_ratio(res, "v7") + 2 * max(0.0, real_ratio(res, "v7") - real_ratio(res, "v4"))
        print(f"epoch {ep} loss {tot / nb:.5f} | video val v7/hermite {vid_ratio(res, 'v7'):.3f} "
              f"| real val v7 {real_ratio(res, 'v7'):.3f} (v4 {real_ratio(res, 'v4'):.3f}) | score {score:.3f}")
        if score < best:
            best = score
            torch.save(model.state_dict(), a.out)

    model.load_state_dict(torch.load(a.out, map_location=dev))
    show("\nVAL (best v7, mixed masks)", evaluate(models, vl, dev))
    rt = evaluate(models, sl, dev)
    show("TEST (best v7, mixed masks)", rt)
    print("\nFIXED-GAP EVAL on video (px)")
    for split in ("val", "test"):
        for L, name in ((10, "2 s"), (15, "3 s")):
            r = eval_fixed_gap(models, tracks, split, L, dev)[VIDEO]
            print(f"    {split} gap {name}: v4 {r['v4']:.2f} | v7 {r['v7']:.2f} | hermite {r['hermite']:.2f} | "
                  f"linear {r['linear']:.2f} | v7/hermite {r['v7'] / r['hermite']:.3f}")
    print(f"\nTEST mixed masks | real datasets ratio vs hermite: v4 {real_ratio(rt, 'v4'):.3f} | v7 {real_ratio(rt, 'v7'):.3f}"
          f" | video: v4 {vid_ratio(rt, 'v4'):.3f} | v7 {vid_ratio(rt, 'v7'):.3f}")


if __name__ == "__main__":
    main()
