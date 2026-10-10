"""
train_unified.py -- one LSTM encoder-decoder, 4 datasets, per-dataset val metrics.

Expects (from unified_loader.py):  build(root, split) -> torch Dataset whose items are dicts:
    xy        (T,2) float  ground-truth x/y in metres, window-relative
    gap_mask  (T,)  bool   True where the point is MISSING   (or obs_mask = True where observed)
    imu       (T,6) float  zeros if absent
    imu_mask  (T,)  bool or scalar  True where imu is real
    agent     int          0 two-wheeler, 1 car, 2 pedestrian, 3 hetrod-mix
    dataset   int          0 hetrod, 1 idd_mm, 2 idd_ped, 3 i2wdd
If your loader uses other key names, edit ONLY to_batch() below.

Usage (Colab):
    python src/train_unified.py --root /content/data --epochs 30 --out /content/drive/MyDrive/proj/ckpt.pt
    python src/train_unified.py --synthetic --epochs 2 --out /tmp/t.pt     # smoke test, no data needed
"""
import argparse, os, time, json
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DS_NAMES = ["hetrod", "idd_mm", "idd_ped", "i2wdd"]
N_AGENT, N_DS, SCALE = 4, 4, 10.0   # SCALE: metres -> network units


# ---------------------------------------------------------------- batch plumbing
def to_batch(b, dev):
    xy = b["xy"].float()
    if "gap_mask" in b:
        gap = b["gap_mask"].bool()
    else:
        gap = ~b["obs_mask"].bool()
    B, T, _ = xy.shape
    imu = b["imu"].float()
    im = b["imu_mask"].bool()
    if im.dim() == 1:                       # per-sample flag -> per-step
        im = im[:, None].expand(B, T)
    out = dict(xy=xy, gap=gap, imu=imu * im[..., None], imu_mask=im,
               agent=b["agent"].long(), ds=b["dataset"].long())
    return {k: v.to(dev) for k, v in out.items()}


def linear_fill(xy, gap):
    """Batched linear interpolation across gap points. xy (B,T,2), gap (B,T) bool."""
    B, T, _ = xy.shape
    obs = ~gap
    idx = torch.arange(T, device=xy.device).expand(B, T)
    prev = torch.cummax(torch.where(obs, idx, torch.full_like(idx, -1)), 1).values
    nxt = torch.flip(torch.cummin(torch.flip(
        torch.where(obs, idx, torch.full_like(idx, T)), [1]), 1).values, [1])
    prev_c, nxt_c = prev.clamp(0, T - 1), nxt.clamp(0, T - 1)
    prev_c = torch.where(prev < 0, nxt_c, prev_c)      # edge: hold nearest observed
    nxt_c = torch.where(nxt >= T, prev_c, nxt_c)
    xp = torch.gather(xy, 1, prev_c[..., None].expand(-1, -1, 2))
    xn = torch.gather(xy, 1, nxt_c[..., None].expand(-1, -1, 2))
    den = (nxt_c - prev_c).clamp(min=1).float()
    w = ((idx - prev_c).float() / den)[..., None]
    w = torch.where((nxt_c == prev_c)[..., None], torch.zeros_like(w), w)
    return xp + w * (xn - xp)


# ---------------------------------------------------------------- model
class Imputer(nn.Module):
    """BiLSTM encoder -> LSTM decoder -> residual added on top of the linear fill."""
    def __init__(self, hid=128, layers=2, emb=16, drop=0.1):
        super().__init__()
        self.agent_emb = nn.Embedding(N_AGENT, emb)
        self.ds_emb = nn.Embedding(N_DS, emb)
        in_dim = 2 + 1 + 6 + 1 + 2 * emb        # lin xy, gap flag, imu, imu_mask, embeddings
        self.enc = nn.LSTM(in_dim, hid, layers, batch_first=True, bidirectional=True,
                           dropout=drop if layers > 1 else 0)
        self.dec = nn.LSTM(2 * hid + in_dim, hid, layers, batch_first=True,
                           dropout=drop if layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 2))

    def forward(self, xy, gap, imu, imu_mask, agent, ds):
        lin = linear_fill(xy, gap) / SCALE
        e = torch.cat([self.agent_emb(agent), self.ds_emb(ds)], -1)
        e = e[:, None, :].expand(-1, xy.shape[1], -1)
        x = torch.cat([lin, gap.float()[..., None], imu, imu_mask.float()[..., None], e], -1)
        h, _ = self.enc(x)
        d, _ = self.dec(torch.cat([h, x], -1))
        return (lin + self.head(d)) * SCALE, lin * SCALE      # (pred, linear baseline) in metres


# ---------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    se_m = torch.zeros(N_DS); se_l = torch.zeros(N_DS); n = torch.zeros(N_DS)
    for raw in loader:
        b = to_batch(raw, dev)
        pred, lin = model(b["xy"], b["gap"], b["imu"], b["imu_mask"], b["agent"], b["ds"])
        g = b["gap"]
        em = ((pred - b["xy"]) ** 2).sum(-1) * g       # squared euclid error on gap pts only
        el = ((lin - b["xy"]) ** 2).sum(-1) * g
        for d in range(N_DS):
            s = b["ds"] == d
            if s.any():
                se_m[d] += em[s].sum().cpu(); se_l[d] += el[s].sum().cpu(); n[d] += g[s].sum().cpu()
    res = {}
    for d in range(N_DS):
        if n[d] > 0:
            res[DS_NAMES[d]] = dict(model=(se_m[d] / n[d]).sqrt().item(),
                                    linear=(se_l[d] / n[d]).sqrt().item(), n=int(n[d]))
    return res


def print_table(res, title):
    print(f"--- {title} (RMSE m, gap points) ---")
    print(f"{'dataset':10s} {'model':>8s} {'linear':>8s} {'gain%':>7s} {'n':>8s}")
    for k, v in res.items():
        gain = 100 * (1 - v["model"] / max(v["linear"], 1e-9))
        print(f"{k:10s} {v['model']:8.3f} {v['linear']:8.3f} {gain:7.1f} {v['n']:8d}")


# ---------------------------------------------------------------- synthetic (smoke test only)
class SyntheticDS(Dataset):
    def __init__(self, n=256, T=50, seed=0):
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(n, 1, 2, generator=g) * 2
        acc = torch.randn(n, T, 2, generator=g) * 0.3
        self.xy = torch.cumsum(v + torch.cumsum(acc, 1) * 0.1, 1) * 0.2
        self.xy = self.xy - self.xy[:, :1]
        self.ds = torch.randint(0, N_DS, (n,), generator=g)
        self.gap = torch.zeros(n, T, dtype=torch.bool)
        for i in range(n):
            L = int(torch.randint(6, 15, (1,), generator=g)); s = int(torch.randint(10, T - 10 - L, (1,), generator=g))
            self.gap[i, s:s + L] = True
        self.imu = torch.randn(n, T, 6)
    def __len__(self): return len(self.xy)
    def __getitem__(self, i):
        d = int(self.ds[i])
        return dict(xy=self.xy[i], gap_mask=self.gap[i], imu=self.imu[i],
                    imu_mask=torch.full((self.xy.shape[1],), d == 3), agent=torch.tensor(d), dataset=torch.tensor(d))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/data")
    ap.add_argument("--out", default="/content/drive/MyDrive/proj/ckpt.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_batches", type=int, default=0, help="limit batches/epoch (profiling)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.synthetic:
        tr, va = SyntheticDS(512, seed=0), SyntheticDS(128, seed=1)
    else:
        from unified_loader import build
        tr, va = build(a.root, "train"), build(a.root, "val")
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=dev == "cuda",
                    persistent_workers=a.workers > 0)
    vl = DataLoader(va, a.bs, num_workers=a.workers)

    model = Imputer(a.hid).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    start, best = 0, float("inf")
    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {start}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(f"device={dev} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M train={len(tr)} val={len(va)}")

    for ep in range(start, a.epochs):
        model.train(); t0 = time.time(); tot = cnt = 0
        for i, raw in enumerate(tl):
            if a.max_batches and i >= a.max_batches: break
            b = to_batch(raw, dev)
            pred, _ = model(b["xy"], b["gap"], b["imu"], b["imu_mask"], b["agent"], b["ds"])
            g = b["gap"][..., None].float()
            loss = (F.huber_loss(pred / SCALE, b["xy"] / SCALE, reduction="none", delta=0.5) * g).sum() / (g.sum() * 2 + 1e-9)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); cnt += 1
        sched.step()
        res = evaluate(model, vl, dev)
        print(f"\nepoch {ep+1}/{a.epochs}  train_loss={tot/max(cnt,1):.4f}  {time.time()-t0:.0f}s")
        print_table(res, "val")
        # model selection: mean ratio model/linear across datasets (no dataset can hide)
        score = sum(v["model"] / max(v["linear"], 1e-9) for v in res.values()) / max(len(res), 1)
        ck = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, best=best, val=res)
        if score < best:
            best = score; ck["best"] = best
            torch.save(ck, a.out.replace(".pt", "_best.pt"))
        torch.save(ck, a.out)   # always keep last for --resume
        with open(a.out.replace(".pt", "_log.jsonl"), "a") as f:
            f.write(json.dumps(dict(epoch=ep + 1, loss=tot / max(cnt, 1), score=score, val=res)) + "\n")
    print(f"\ndone. best mean(model/linear) = {best:.3f}")


if __name__ == "__main__":
    main()