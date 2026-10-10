"""finetune_sumo.py -- fine-tune v4 on SUMO + replay of the 4 real datasets; compares v4 vs v6 vs Hermite."""
import argparse, copy, os, pickle, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_schema as cs

# register SUMO as dataset 4 IN MEMORY ONLY (common_schema.py on disk is untouched)
SUMO = 4
cs.DATASET["sumo"] = SUMO
cs.DATASET_NAMES[SUMO] = "sumo"
cs.SCALE[SUMO] = 50.0
cs.UNIT[SUMO] = "m"
cs.STRIDE_BY_DS[SUMO] = 5
cs.N_DATASETS = 5

from model_unified import UnifiedImputer, masked_loss   # must be imported AFTER the patch
from unified_data import UnifiedWindows

REAL = [0, 1, 2, 3]
KEYS = ("ds", "agent", "split", "xy")


@torch.no_grad()
def evaluate(models, loader, dev):
    for m in models.values():
        m.eval()
    acc = {}
    for b in loader:
        x, y, obs = b["x"].to(dev), b["y"].to(dev), b["obs"].to(dev)
        ag, ds = b["agent"].to(dev), b["ds"].to(dev)
        miss = 1.0 - obs
        sc2 = b["scale"].to(dev).view(-1, 1) ** 2
        se = lambda p: (((p - y) ** 2).sum(-1) * sc2 * miss).sum(1)
        e = {"linear": se(x[:, :, 3:5]), "hermite": se(x[:, :, 5:7]), "n": miss.sum(1)}
        for k, m in models.items():
            e[k] = se(m(x, ag, ds))
        for d in ds.unique().tolist():
            s = ds == d
            a = acc.setdefault(d, {k: 0.0 for k in e})
            for k in e:
                a[k] += e[k][s].sum().item()
    return {d: {k: (float(np.sqrt(v / max(a["n"], 1))) if k != "n" else v) for k, v in a.items()}
            for d, a in acc.items()}


def show(title, res):
    print(title)
    for d in sorted(res):
        r = res[d]
        print(f"  {cs.DATASET_NAMES[d]:8s} v4 {r['v4']:8.3f} | v6 {r['v6']:8.3f} | hermite {r['hermite']:8.3f} "
              f"| linear {r['linear']:8.3f} {cs.UNIT[d]} | v6/hermite {r['v6'] / r['hermite']:.3f}")


def score(res, key="v6"):
    return float(np.mean([res[d][key] / res[d]["hermite"] for d in REAL if d in res]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="unified_tracks.pkl")
    ap.add_argument("--sumo", default="sumo_tracks.pkl")
    ap.add_argument("--v4", default="unified_v4.pt")
    ap.add_argument("--out", default="unified_v6_sumo.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--per_epoch", type=int, default=30000)
    ap.add_argument("--hidden", type=int, default=128)
    a = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tracks = [{k: t[k] for k in KEYS}
              for t in pickle.load(open(a.cache, "rb")) + pickle.load(open(a.sumo, "rb"))]
    train, val, test = (UnifiedWindows(tracks, s) for s in ("train", "val", "test"))
    for n, d_ in (("train", train), ("val", val), ("test", test)):
        print(n, len(d_), {cs.DATASET_NAMES[d]: int((np.asarray(d_.ds) == d).sum()) for d in np.unique(d_.ds)})

    # v4 weights -> 5-row dataset embedding (new SUMO row starts as the mean of the 4 old rows)
    model = UnifiedImputer(hidden=a.hidden)
    sd = model.state_dict()
    for k, v in torch.load(a.v4, map_location="cpu").items():
        if k == "ds_emb.weight":
            sd[k][:v.shape[0]] = v
            sd[k][v.shape[0]:] = v.mean(0)
        else:
            sd[k] = v
    model.load_state_dict(sd)
    model.to(dev)
    base = copy.deepcopy(model).eval()

    cnt = np.bincount(np.asarray(train.ds), minlength=5)
    w = 1.0 / cnt[np.asarray(train.ds)]
    sampler = WeightedRandomSampler(w, num_samples=a.per_epoch, replacement=True)   # equal share per dataset
    tl = DataLoader(train, batch_size=a.bs, sampler=sampler, drop_last=True, num_workers=2, persistent_workers=True)
    vl, sl = DataLoader(val, batch_size=256), DataLoader(test, batch_size=256)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    show("VAL before fine-tuning (v6 = v4 + new SUMO embedding row)", evaluate({"v4": base, "v6": model}, vl, dev))
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
        res = evaluate({"v4": base, "v6": model}, vl, dev)
        sc = score(res)
        print(f"epoch {ep} loss {tot / nb:.5f} | real-ds val ratio vs hermite: v6 {sc:.3f} (v4 {score(res, 'v4'):.3f})"
              f" | sumo v6/hermite {res[SUMO]['v6'] / res[SUMO]['hermite']:.3f}")
        if sc < best:
            best = sc
            torch.save(model.state_dict(), a.out)

    model.load_state_dict(torch.load(a.out, map_location=dev))
    show("\nVAL (best v6)", evaluate({"v4": base, "v6": model}, vl, dev))
    rt = evaluate({"v4": base, "v6": model}, sl, dev)
    show("TEST (best v6)", rt)
    print(f"\nTEST real-datasets ratio vs hermite: v4 {score(rt, 'v4'):.3f} | v6 {score(rt):.3f}")


if __name__ == "__main__":
    main()