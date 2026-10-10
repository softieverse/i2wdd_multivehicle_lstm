"""
app.py -- Gradio GUI for the unified trajectory imputer.
Colab : import app; app.demo.launch(share=True)
HF Space: this file is the entry point (app.py), checkpoint next to it.
"""
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr

from impute import PROFILES, impute_df, load_model, make_predictor, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = next((p for p in (os.environ.get("CKPT", ""),
                         os.path.join(HERE, "runs", "unified_v4.pt"),
                         os.path.join(HERE, "unified_v4.pt")) if p and os.path.exists(p)), None)
if CKPT is None:
    raise FileNotFoundError("unified_v4.pt not found (looked in ./runs and ./)")
MODEL = load_model(CKPT)
print("loaded", CKPT)

PROFILE_NAMES = list(PROFILES)


def _predictor(profile):
    ds, agent = PROFILES[profile]
    return make_predictor(MODEL, ds, agent), ds


def _read(file):
    if file is None:
        raise gr.Error("Upload a CSV first.")
    path = file if isinstance(file, str) else file.name
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise gr.Error(f"Could not read the CSV: {e}")


def _plot_fill(out, grid, unit):
    fig, ax = plt.subplots(figsize=(7, 5))
    obs = out["imputed"].to_numpy() == 0
    ax.plot(out["x"], out["y"], "-", color="#4c78a8", lw=1, alpha=0.5, label="completed track")
    ax.plot(out["x_hermite"][~obs], out["y_hermite"][~obs], ".", color="#54a24b", ms=5, label="Hermite baseline (gaps)")
    ax.plot(out["x"][obs], out["y"][obs], "o", color="black", ms=3, label="observed")
    ax.plot(out["x"][~obs], out["y"][~obs], "o", color="#e45756", ms=4, label="model-imputed")
    ax.set_xlabel(f"x ({unit})"); ax.set_ylabel(f"y ({unit})")
    ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def run_fill(file, profile, hz):
    df = _read(file)
    predict, ds = _predictor(profile)
    try:
        out, info, grid = impute_df(df, predict, ds, in_hz=float(hz or 5.0))
    except ValueError as e:
        raise gr.Error(str(e))
    unit = info["unit"] if "lat" not in out else "m"
    msg = (f"**Filled {info['n_missing']} of {info['n']} points** "
           f"({info['missing_pct']:.1f}% missing, longest gap {info['longest_gap_s']:.1f} s). "
           f"Output is on a 5 Hz grid; observed points are unchanged.")
    if info["warnings"]:
        msg += "\n\n" + "\n".join(f"- ⚠️ {w}" for w in info["warnings"])
    tmp = os.path.join(tempfile.mkdtemp(), "imputed.csv")
    out.to_csv(tmp, index=False)
    return _plot_fill(out, grid, unit), msg, out.round(4), tmp


def run_test(file, profile, hz, gap_s, trials):
    df = _read(file)
    predict, ds = _predictor(profile)
    try:
        m, ex = simulate(df, predict, ds, gap_s=float(gap_s), trials=int(trials), in_hz=float(hz or 5.0))
    except ValueError as e:
        raise gr.Error(str(e))
    r, u = m["rmse"], m["unit"]
    gain = (1 - r["model"] / r["hermite"]) * 100 if r["hermite"] > 0 else 0.0
    table = (f"**{m['trials']} random {m['gap_s']:.1f} s gaps cut from your track "
             f"({m['points']} deleted points). RMSE on the deleted points:**\n\n"
             f"| method | RMSE ({u}) |\n|---|---|\n"
             f"| **model** | **{r['model']:.4f}** |\n"
             f"| Hermite | {r['hermite']:.4f} |\n"
             f"| linear | {r['linear']:.4f} |\n\n"
             f"Model vs Hermite: **{gain:+.1f}%** error change (positive = model better).")
    fig, ax = plt.subplots(figsize=(7, 5))
    t, sl = ex["truth_full"], ex["sl"]
    ax.plot(t[:, 0], t[:, 1], "-", color="lightgray", lw=3, label="ground truth")
    ax.plot(ex["herm"][sl, 0], ex["herm"][sl, 1], "--", color="#54a24b", label="Hermite")
    ax.plot(ex["filled"][sl, 0], ex["filled"][sl, 1], "o-", color="#e45756", ms=4, label="model")
    keep = np.isfinite(ex["piece"]).all(1)
    ax.plot(ex["piece"][keep, 0], ex["piece"][keep, 1], "o", color="black", ms=3, label="observed")
    ax.set_xlabel(f"x ({u})"); ax.set_ylabel(f"y ({u})")
    ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Example trial")
    fig.tight_layout()
    return fig, table


with gr.Blocks(title="Trajectory gap imputer") as demo:
    gr.Markdown(
        "# Trajectory gap imputer\n"
        "Upload a track with missing points; the LSTM encoder-decoder fills them in.\n\n"
        "**CSV format:** header row with `x,y` (metres, or pixels for pedestrians) **or** `lat,lon`; "
        "optional `t` column (seconds). Missing points = empty cells or absent timestamps. "
        "Needs >= 6 s of track (30 samples at 5 Hz). Best for gaps up to ~3 s; "
        "output is on a 5 Hz grid."
    )
    with gr.Tab("Fill my data"):
        with gr.Row():
            f1 = gr.File(label="CSV with gaps", file_types=[".csv"], type="filepath")
            with gr.Column():
                p1 = gr.Dropdown(PROFILE_NAMES, value=PROFILE_NAMES[0], label="Agent / data type")
                h1 = gr.Number(value=5, label="Input sample rate (Hz), used only if no `t` column")
                b1 = gr.Button("Impute missing values", variant="primary")
        m1 = gr.Markdown()
        pl1 = gr.Plot()
        tb1 = gr.Dataframe(label="Completed track (imputed=1 marks filled points)", interactive=False)
        dl1 = gr.File(label="Download filled CSV")
        b1.click(run_fill, [f1, p1, h1], [pl1, m1, tb1, dl1])
    with gr.Tab("Test accuracy"):
        gr.Markdown("Upload a track that is (mostly) **complete**. Random gaps are cut out, "
                    "imputed, and compared with the real values.")
        with gr.Row():
            f2 = gr.File(label="Complete CSV", file_types=[".csv"], type="filepath")
            with gr.Column():
                p2 = gr.Dropdown(PROFILE_NAMES, value=PROFILE_NAMES[0], label="Agent / data type")
                h2 = gr.Number(value=5, label="Input sample rate (Hz), used only if no `t` column")
                g2 = gr.Slider(1.0, 3.0, value=2.0, step=0.2, label="Gap length (s)")
                n2 = gr.Slider(5, 50, value=20, step=5, label="Number of random gaps")
                b2 = gr.Button("Run test", variant="primary")
        pl2 = gr.Plot()
        m2 = gr.Markdown()
        b2.click(run_test, [f2, p2, h2, g2, n2], [pl2, m2])

if __name__ == "__main__":
    demo.launch()
