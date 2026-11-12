"""Build every figure and data table for the consolidated supervisor report.

Reads only cached results in this repository (no GPU, no checkpoints needed):
  data_results/{v1_base_pogs,v2_kalman,realtime}/summary.csv
  model/foundation_study/artefacts/**/{latency.json,tracking.json}
Writes figures/*.pdf and tables/*.tex next to this file.

    python make_figures.py
"""
import csv, glob, json, os, shutil
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # repository root
DEV = ROOT                                                # reproduction workstream
V2 = os.path.join(ROOT, "model", "foundation_study")      # model-substitution study
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(HERE, "tables"); os.makedirs(TAB, exist_ok=True)

# Reference palette (dataviz skill, light mode; print medium)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------- 1. latency
def collect_latency():
    """fp16 median latency + peak VRAM per model, at 1280x720 (pickplace)."""
    rows = {}
    for p in glob.glob(os.path.join(V2, "artefacts", "*", "*", "bench", "latency.json")):
        d = json.load(open(p))
        for m in d.get("measurements", []):
            if m.get("status") != "ok":
                continue
            key = (m["model"], m["dtype"])
            # prefer the 1280x720 scene; otherwise keep whatever exists
            hi = m.get("scene_resolution") == "1280x720"
            if key not in rows or (hi and not rows[key]["hi"]):
                rows[key] = dict(stage=m["stage"], ms=m["median_ms"], p95=m["p95_ms"],
                                 mb=m["peak_delta_mb"], hi=hi, res=m.get("scene_resolution"))
    return rows


MODEL_INFO = {   # display name, role, paper/venue, POGS-compatible (open-vocab + masks)
    "detic":            ("Detic", "Detection", "ECCV 2022", True),
    "locateanything3d": ("LocateAnything3D", "Detection", "CVPR 2026", False),
    "rfdetr":           ("RF-DETR", "Detection", "ICLR 2026", False),
    "vfmdetr":          ("VFM-DETR", "Detection", "NeurIPS 2025", None),
    "yolo":             ("YOLO11x", "Detection", "Ultralytics", False),
    "dino_vits14":      ("DINOv2 ViT-S/14", "Features", "TMLR 2024", True),
    "dinov3":           ("DINOv3 ViT-B/16", "Features", "arXiv 2025", True),
    "dino_txt":         ("dino.txt (DINOv2 Meets Text)", "Feat.+Align.", "CVPR 2025", True),
    "clip_vitb16":      ("CLIP ViT-B/16", "Alignment", "ICML 2021", True),
    "mobileclipv2":     ("MobileCLIPv2", "Alignment", "TMLR 2025", True),
    "siglip2":          ("SigLIP2-base", "Alignment", "DeepMind 2025", True),
}


def fig_latency(lat):
    # Measured fp16 medians; Detic and SigLIP2 come from the in-pipeline logs
    # (COMPARISON_STUDY.md 4b/4c); LocateAnything3D from its own harness.
    entries = []
    for key in ["yolo", "rfdetr"]:
        r = lat.get((key, "fp16"))
        if r: entries.append((key, r["ms"], "measured"))
    entries.append(("detic", 148.0, "log"))          # 128-154 ms, in-pipeline fp32
    entries.append(("locateanything3d", 430.0, "harness"))  # 0.43-1.64 s / detection
    for key in ["dino_vits14", "dinov3", "dino_txt", "clip_vitb16", "mobileclipv2"]:
        r = lat.get((key, "fp16"))
        if r: entries.append((key, r["ms"], "measured"))
    entries.append(("siglip2", 24.0 * 2.39, "log"))   # 2.39x CLIP embedding pass, 1280x720

    role_col = {"Detection": C1, "Features": C2, "Feat.+Align.": C2, "Alignment": C3}
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(entries))[::-1]
    for yi, (k, ms, src) in zip(y, entries):
        name, role, _, ok = MODEL_INFO[k]
        hatch = None if src == "measured" else "////"
        ax.barh(yi, ms, height=0.62, color=role_col[role], hatch=hatch,
                edgecolor="white", linewidth=0.8)
        lab = f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms"
        if k == "locateanything3d": lab = "430-1640 ms"
        if k == "detic": lab = "128-154 ms"
        ax.text(ms * 1.08, yi, lab, va="center", fontsize=7.5, color=INK2)
        tag = "" if ok else ("  (no masks / closed vocab)" if ok is False else "")
        ax.text(-0.01, yi, name + tag, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8, color=INK if ok else MUTED)
    ax.set_yticks([])
    ax.set_xscale("log"); ax.set_xlim(8, 4000)
    ax.set_xlabel("Median latency per frame, ms (log scale), RTX 4070, 1280x720, fp16")
    ax.axvline(200, color=CRIT, lw=1, ls="--")
    ax.text(215, y[0], "5 Hz budget\n(200 ms / frame)", color=CRIT,
            fontsize=7, va="center")
    ax.legend(handles=[Patch(color=C1, label="Object detection"),
                       Patch(color=C2, label="Visual features"),
                       Patch(color=C3, label="Feature alignment"),
                       Patch(facecolor="white", edgecolor=INK2, hatch="////",
                             label="from pipeline logs / own harness")],
              loc="upper center", bbox_to_anchor=(0.38, -0.2), ncol=4, fontsize=7.2)
    ax.grid(axis="y", visible=False)
    save(fig, "latency_models.pdf")
    return entries


def fig_latency_vram(lat):
    pts = [(k, lat[(k, "fp16")]) for k in
           ["yolo", "rfdetr", "dino_vits14", "dinov3", "dino_txt", "clip_vitb16", "mobileclipv2"]
           if (k, "fp16") in lat]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    role_col = {"Detection": C1, "Features": C2, "Feat.+Align.": C2, "Alignment": C3}
    for k, r in pts:
        name, role, _, _ = MODEL_INFO[k]
        ax.scatter(r["ms"], r["mb"], s=46, color=role_col[role], edgecolor="white",
                   linewidth=1.5, zorder=3)
        off = {"mobileclipv2": (-8, 8), "clip_vitb16": (6, -10), "yolo": (-10, -12),
               "rfdetr": (6, 4)}.get(k, (5, 3))
        ax.annotate(name.split(" (")[0], (r["ms"], r["mb"]), xytext=off,
                    textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("fp16 median latency (ms)")
    ax.set_ylabel("Peak extra VRAM (MB)")
    ax.axhline(12282, color=CRIT, lw=0.8, ls="--")
    ax.text(14, 12282 * 0.75, "RTX 4070 total (12 GB)", fontsize=7, color=CRIT)
    ax.set_xlim(12, 400); ax.set_ylim(40, 20000)
    save(fig, "latency_vram.pdf")


def tab_latency(lat):
    order = ["yolo", "rfdetr", "dino_vits14", "dinov3", "dino_txt", "clip_vitb16", "mobileclipv2"]
    lines = []
    for k in order:
        a, b = lat.get((k, "fp32")), lat.get((k, "fp16"))
        if not b: continue
        name, role, venue, ok = MODEL_INFO[k]
        def fm(r, key, fmt):
            return fmt.format(r[key]) if r else "--"
        lines.append(f"{role} & {name} & {venue} & {fm(a,'ms','{:.1f}')} & {fm(b,'ms','{:.1f}')} & "
                     f"{fm(b,'p95','{:.1f}')} & {fm(b,'mb','{:,.0f}')} & {1000/b['ms']:.0f} \\\\")
    open(os.path.join(TAB, "latency_rows.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote latency_rows.tex")


# ------------------------------------------------- 2. model-study tracking noise
def fig_tracking_noise():
    scenes = [f"pendulum_v2_{a:02d}" for a in (5, 10, 15, 20, 25, 30, 40, 50, 60)]
    rows = {"00_baseline_detic_dinov2_clip": ("Row 00 baseline", C1),
            "01_locateanything_dinotxt": ("Row 01 (= baseline re-run)", C2),
            "03_vfmdetr_siglip2": ("Row 03 SigLIP2 alignment", C3)}
    data, null = defaultdict(dict), {}
    for sc in scenes:
        for r in rows:
            p = os.path.join(V2, "artefacts", r, sc, "track", "tracking.json")
            if not os.path.exists(p): continue
            d = json.load(open(p))
            data[r][sc] = d["summary"]["centroid_err_mm_mean"]
            if sc not in null:
                g = [fr["gt_displacement_mm"] for fr in d["frames"] if fr["gt_displacement_mm"] > 1.0]
                null[sc] = float(np.mean(g)) if g else np.nan
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x = np.arange(len(scenes)); w = 0.26
    for i, (r, (lab, col)) in enumerate(rows.items()):
        vals = [data[r].get(s, np.nan) for s in scenes]
        ax.bar(x + (i - 1) * w, vals, width=w - 0.03, color=col, label=lab)
    ax.plot(x, [null[s] for s in scenes], color=INK, lw=2, marker="o", ms=4,
            label='"Object never moved" predictor')
    ax.set_xticks(x, [s.split("_")[-1].lstrip("0") + "°" for s in scenes])
    ax.set_xlabel("Pendulum swing amplitude (v2 scenes)")
    ax.set_ylabel("Mean centroid error (mm)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_ylim(0, 560); ax.grid(axis="x", visible=False)
    save(fig, "modelstudy_tracking_noise.pdf")

    # seed variance on pickplace (baseline, identical config)
    seeds = {}
    for sc, s in [("pickplace", 42), ("pickplace_seed13", 13), ("pickplace_seed7", 7)]:
        base = os.path.join(V2, "artefacts", "00_baseline_detic_dinov2_clip", sc, "track")
        p = os.path.join(base, "tracking_iters15.json")
        if not os.path.exists(p): p = os.path.join(base, "tracking.json")
        if os.path.exists(p):
            seeds[s] = json.load(open(p))["summary"]["centroid_err_mm_mean"]
    return data, null, seeds


# ---------------------------------------------------------- 3. pogs-dev results
def mean_by(rows, keys, val):
    acc = defaultdict(list)
    for r in rows:
        v = f(r[val])
        if not np.isnan(v):
            acc[tuple(r[k] for k in keys)].append(v)
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in acc.items()}


def fig_factor_heatmap():
    v1 = read_csv(os.path.join(DEV, "data_results", "v1_base_pogs", "summary.csv"))
    v2 = read_csv(os.path.join(DEV, "data_results", "v2_kalman", "summary.csv"))
    rows = v1 + [r for r in v2 if r["method"] not in {x["method"] for x in v1}]
    conds = [("yaw15", 1, "Yaw 15°"), ("base", 1, "Yaw 30° (base)"), ("yaw90", 1, "Yaw 90°"),
             ("tilt30", 1, "Tilt 30°"), ("tilt60", 1, "Tilt 60°"), ("combined45", 1, "Yaw+tilt 45°"),
             ("base", 2, "Speed 2×"), ("base", 4, "Speed 4×"),
             ("occ_none", 1, "No occluder"), ("occ_hand", 1, "Hand, partial"),
             ("occ_full", 1, "Hand, heavy"), ("obj_mustard", 1, "Mustard bottle"),
             ("obj_cereal", 1, "Cereal box"), ("multi", 1, "Two objects moving")]
    methods = ["Static (no motion)", "POGS", "POGS + KF", "POGS + KF + check"]
    stats = mean_by([r for r in rows if r["depth"] == "stereo"],
                    ["condition", "speed", "method"], "adds_success")
    M = np.full((len(conds), len(methods)), np.nan)
    for i, (c, sp, _) in enumerate(conds):
        for j, m in enumerate(methods):
            s = stats.get((c, str(sp), m))
            if s: M[i, j] = s[0] * 100
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(SEQ)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 55 else INK)
    ax.set_xticks(range(len(methods)), ["Static", "POGS", "POGS\n+KF", "POGS\n+KF+check"])
    ax.set_yticks(range(len(conds)), [c[2] for c in conds])
    ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(methods)), minor=True)
    ax.set_yticks(np.arange(-.5, len(conds)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
    ax.set_title("ADD-S success (%), mean of 3 seeds, 5 Hz camera", fontsize=8.5, color=INK2)
    save(fig, "factor_heatmap.pdf")


def fig_realtime():
    rt = read_csv(os.path.join(DEV, "data_results", "realtime", "summary.csv"))
    st = mean_by([r for r in rt if r["speed"] == "1"], ["method"], "adds_success")
    hz = mean_by([r for r in rt if r["speed"] == "1"], ["method"], "tracker_hz")
    order = ["POGS", "POGS, 10 iters", "POGS + KF + check", "POGS + KF + check + latency comp.",
             "POGS + KF + check, 10 iters", "POGS + KF + check, 10 iters + latency comp."]
    short = ["POGS (25 it)", "POGS (10 it)", "KF+check (25 it)", "KF+check+LC (25 it)",
             "KF+check (10 it)", "KF+check+LC (10 it)"]
    order = [m for m in order if (m,) in st]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True,
                                 gridspec_kw=dict(wspace=0.08))
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        s = st[(m,)]
        a1.barh(yi, s[0] * 100, xerr=s[1] * 100, height=0.6, color=C1,
                error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        a1.text(min(s[0] * 100 + s[1] * 100 + 2, 101), yi, f"{s[0]*100:.0f}%",
                va="center", fontsize=7, color=INK2)
        h = hz.get((m,), (np.nan,))[0]
        a2.barh(yi, h, height=0.6, color=C2 if h < 5 else C3)
        a2.text(h + 0.08, yi, f"{h:.1f} Hz", va="center", fontsize=7, color=INK2)
    a1.set_yticks(y, [short[[*order].index(m)] for m in order])
    a1.set_xlim(0, 112); a1.set_xlabel("ADD-S success at 30 Hz camera, 1× (%)")
    a2.axvline(5, color=CRIT, lw=1, ls="--")
    a2.text(5.08, y[-1] - 0.45, "5 Hz target", color=CRIT, fontsize=7)
    a2.set_xlim(0, 6.2); a2.set_xlabel("Achieved tracker rate (Hz)")
    for a in (a1, a2): a.grid(axis="y", visible=False)
    save(fig, "realtime.pdf")


def fig_status():
    items = [
        ("Base POGS re-implementation\n(train, group, track)", "done"),
        ("Synthetic scene generation\n(54-case benchmark)", "done"),
        ("Gaussian + pose metrics\n(ADD-S, pos/rot error)", "done"),
        ("Object grouping visuals\n(object-only splats)", "partial"),
        ("With / without object\nsplat comparison", "missing"),
        ("Tool perturbation recovery\n(±5° steps to 30°, KL div.)", "missing"),
        ("In-grasp recovery rate\n(target 80%)", "missing"),
        ("5 Hz tracking frequency", "partial"),
        ("2D foundation-model latency\n(11 models)", "partial"),
        ("Model substitution\nin train/track", "partial"),
        ("Env: Py 3.12 / CUDA 13 /\nPyTorch 2.14", "partial"),
    ]
    col = {"done": GOOD, "partial": WARN, "missing": CRIT}
    lab = {"done": "✓ Done", "partial": "◐ Partial", "missing": "✗ Not started"}
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.axis("off")
    n = len(items); ncol = 3
    for i, (t, s) in enumerate(items):
        r, c = divmod(i, ncol)
        x0, y0 = c * 1.0, -r * 1.0
        ax.add_patch(plt.Rectangle((x0 + 0.03, y0 + 0.08), 0.94, 0.84, color="#f4f3f0", lw=0))
        ax.add_patch(plt.Rectangle((x0 + 0.03, y0 + 0.08), 0.05, 0.84, color=col[s], lw=0))
        ax.text(x0 + 0.13, y0 + 0.62, t, fontsize=7.2, va="center", color=INK)
        ax.text(x0 + 0.13, y0 + 0.2, lab[s], fontsize=7, va="center", color=INK2,
                family="DejaVu Sans")
    ax.set_xlim(0, ncol); ax.set_ylim(-3.0, 1.0)
    save(fig, "status_matrix.pdf")


def copy_existing():
    for n in ["grouping.pdf", "scan_views.jpg", "conditions.jpg", "dash_rt_pogs.jpg",
              "dash_rt_kf.jpg", "dash_cereal_pogs.jpg", "training.pdf", "speed.pdf",
              "rt_error.pdf", "budget.pdf"]:
        src = os.path.join(ROOT, "report", "reproduction", "figures", n)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(FIG, n))
    print("copied reproduction figures")


if __name__ == "__main__":
    lat = collect_latency()
    fig_latency(lat); fig_latency_vram(lat); tab_latency(lat)
    data, null, seeds = fig_tracking_noise()
    print("null:", {k: round(v, 1) for k, v in null.items()})
    print("seeds:", seeds)
    fig_factor_heatmap(); fig_realtime(); fig_status(); copy_existing()
