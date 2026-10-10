import argparse, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from hetrod_dataset import HetroDDataset, POS_SCALE
from hetrod_model import TrajectoryLSTM

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", dev)

if args.smoke:
    train_recs, val_recs, args.epochs = ["02", "03"], ["04"], 3
else:
    train_recs = [f"{i:02d}" for i in range(2, 40)]    # 02-39
    val_recs   = [f"{i:02d}" for i in range(40, 45)]   # 40-44 (45-49 kept for test)

train_ds = HetroDDataset(train_recs)                   # random gaps every epoch
val_ds   = HetroDDataset(val_recs, seed=0)             # fixed gaps -> reproducible
tl = DataLoader(train_ds, args.bs, shuffle=True, drop_last=True)
vl = DataLoader(val_ds, args.bs)
print("train windows:", len(train_ds), "val windows:", len(val_ds))

model = TrajectoryLSTM(in_dim=14).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

def masked_sq(pred, tgt, miss):
    se = ((pred - tgt) ** 2).sum(-1)                   # (B, W)
    return (se * miss).sum() / miss.sum().clamp(min=1)

os.makedirs("checkpoints", exist_ok=True)
best = 1e9
for ep in range(1, args.epochs + 1):
    model.train(); tot = n = 0
    for x, y, m in tl:
        x, y, m = x.to(dev), y.to(dev), m.to(dev)
        loss = masked_sq(model(x), y, m)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item(); n += 1

    model.eval(); se = cnt = 0
    with torch.no_grad():
        for x, y, m in vl:
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            se += (((model(x) - y) ** 2).sum(-1) * m).sum().item()
            cnt += m.sum().item()
    val_rmse = (se / cnt) ** 0.5 * POS_SCALE           # meters
    sched.step(val_rmse)
    print(f"ep {ep:02d} | train mse {tot/n:.4f} | val RMSE {val_rmse:.2f} m")
    if val_rmse < best:
        best = val_rmse
        torch.save(model.state_dict(), "checkpoints/hetrod_best.pt")
print("best val RMSE (m):", round(best, 2))
