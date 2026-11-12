"""Write a row's RESULTS.md from the artefacts that row produced.

Every number in the report is read back from an artefact file on disk. Nothing
is recomputed here and nothing is typed in by hand, so the report cannot drift
away from what was actually measured.

Two constraints are enforced in the text rather than left to the reader:

* "beats baseline" can only ever mean "beats row 00 on this data". The authors
  released no POGS checkpoint and no capture scene, so there is no published
  number to compare against and the report must not imply otherwise.
* Metrics are labelled ``deterministic`` or ``stochastic``. POGS seeds nothing
  of its own, so latency, VRAM and detector IoU are reproducible while tracking
  pose error is a single seeded sample.

Usage::

    python report.py --row 00_baseline_detic_dinov2_clip
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402

NO_BASELINE_CLAIM = (
    "This row is compared against row 00 on the same synthetic data. It is "
    "**not** a comparison against POGS as published: the authors released no "
    "trained POGS checkpoint and no capture scene, so no published baseline "
    "exists to measure against."
)

#: Which row is being reported. ``build()`` sets this; ``section_patches`` needs
#: it to locate that row's applied-patch record, and threading it through every
#: section would be noise for the four sections that do not care.
PATCH_ROW = [None]


def load(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return None


def fmt(value, spec: str = "", dash: str = "not run") -> str:
    return dash if value is None else format(value, spec) if spec else str(value)


def section_scene_matrix(per_scene: list[dict]) -> str:
    """The scene x metric table: what this row did, on every input.

    The angle ladder is the point, so the scenes appear in order rather than
    alphabetically and a missing rung is visibly missing rather than absent. A
    cell says `not run` rather than being dropped, because "we did not measure
    this" and "we measured this and got nothing" are different claims.
    """
    if not per_scene:
        return "_no scenes run yet_"

    out = ["| scene | res | PSNR | gaussians | centroid err | frames | status |",
           "|---|---|---|---|---|---|---|"]
    for s in per_scene:
        name = s["scene"]
        tr, tk = s["train"] or {}, s["track"] or {}
        summ = tk.get("summary") or {}
        res = ""
        try:
            w, h = cfg.scene_resolution(cfg.scene_path(name))
            res = f"{w}x{h}"
        except SystemExit:
            res = "-"

        psnr = f"{tr['psnr']:.2f}" if tr.get("psnr") is not None else "-"
        gauss = f"{tr['gaussian_count']:,}" if tr.get("gaussian_count") else "-"
        err = (f"{summ['centroid_err_mm_mean']:.1f} mm"
               if summ.get("centroid_err_mm_mean") is not None else "-")
        frames = (f"{summ.get('frames_scored', '?')}/"
                  f"{summ.get('frames_total', '?')}"
                  if summ else "-")

        done = [bool(tr), bool(tk)]
        status = "measured" if all(done) else ("partial" if any(done) else "not run")
        # A lost track still produces a full set of per-frame scores, so without
        # this it reports as "measured" like any other run. The error figure is
        # then the drift of an estimate that stopped following the object, which
        # is not comparable with a run that held -- and is misleadingly *low*
        # when the loss happened early.
        if summ.get("lost_at_frame") is not None:
            status = f"**lost@{summ['lost_at_frame']}**"
        out.append(f"| `{name}` | {res} | {psnr} | {gauss} | {err} | "
                   f"{frames} | {status} |")

    out += [
        "",
        "`frames` is moving frames scored over total frames in the sequence. The "
        "error figure is displacement error against the generator's ground "
        "truth, averaged only over frames where the object actually moved; "
        "stationary frames would dilute it toward zero.",
    ]
    return "\n".join(out)


def section_environment(env: dict) -> str:
    if not env:
        return "_not recorded_"
    rows = [
        ("GPU", env.get("gpu", "cpu")),
        ("GPU memory", f"{env.get('gpu_total_mb', '?')} MB"),
        ("CUDA", env.get("cuda", "-")),
        ("torch", env.get("torch", "-")),
        ("cudnn", env.get("cudnn", "-")),
        ("Python", env.get("python", "-")),
        ("Platform", env.get("platform", "-")),
    ]
    out = ["| item | value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out)


def section_latency(bench: dict) -> str:
    if not bench:
        return "_bench.py did not run_"
    ms = bench.get("measurements", [])
    if not ms:
        return "_no measurements_"
    out = [
        "| model | stage | POGS uses | dtype | resolution | median ms | p95 ms | peak MB | params |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in ms:
        if m["status"] != "ok":
            out.append(f"| `{m['model']}` | {m['stage']} | "
                       f"{'yes' if m['pogs_uses'] else 'no'} | {m['dtype']} | "
                       f"{m['resolution']} | **blocked** | - | - | - |")
            continue
        params = f"{m['params'] / 1e6:.1f}M" if m.get("params") else "-"
        out.append(
            f"| `{m['model']}` | {m['stage']} | "
            f"{'**yes**' if m['pogs_uses'] else 'no'} | {m['dtype']} | "
            f"{m['resolution']} | {m['median_ms']:.2f} | {m['p95_ms']:.2f} | "
            f"{m['peak_delta_mb']:.0f} | {params} |"
        )
    p = bench.get("protocol", {})
    total = bench.get("total_pipeline_median_ms_fp16")
    out += [
        "",
        f"Protocol: {p.get('warmup')} warmup, {p.get('iters')} timed iterations, "
        f"`cudnn.benchmark`, `cuda.synchronize()` on both sides of the timed "
        f"region, seed {p.get('seed')}.",
        "",
        f"**Total measured pipeline (fp16): {fmt(total, '.2f', 'n/a')} ms** "
        f"({bench.get('measured_models', 0)} models measured, "
        f"{bench.get('blocked_models', 0)} blocked).",
        "",
        "Metrics in this section are **deterministic** -- no training is "
        "involved, so they carry no seed variance.",
    ]
    return "\n".join(out)


def weights_cell(probe_rec: dict, info: dict) -> str:
    """One word for whether a model's weights are obtainable.

    A row's whole point is to substitute a model, so "blocked" and "gated" are
    results that belong in the capability table, not just the errors section.
    """
    if info.get("blocked"):
        return "**BLOCKED**"
    status = probe_rec.get("status")
    if status == "gated":
        return "gated (licence)"
    if status == "available":
        return "yes"
    return status or "unprobed"


def section_patches(row_meta: dict, patched: bool) -> str:
    """The patch table, when a row modified POGS.

    Never printed for a row that did not, because "stock POGS" is the strongest
    thing a report can say and it should be said by omission, not by a heading
    that happens to be empty.
    """
    if not patched:
        return ""
    reason = row_meta.get("patch_reason", "")
    out = ["**This row patches POGS.** Stock POGS hardcodes the models it uses, "
           "so the substitution is not expressible as configuration:"]
    if reason:
        out += ["", reason]
    # row_common.sh records the applied patch text per row; the file is found by
    # path rather than assumed, so a report built before the run still reads
    # "not recorded" instead of a stale diff from another row.
    applied = os.path.join(cfg.ARTEFACTS, PATCH_ROW[0], "patches_applied.txt") \
        if PATCH_ROW[0] else None
    if applied and os.path.exists(applied):
        with open(applied, encoding="utf-8", errors="replace") as fh:
            out += ["", "```diff", fh.read().rstrip(), "```"]
    return "\n".join(out)


def section_capability(row_meta: dict, val: dict, probe: dict) -> str:
    lines = []
    det = row_meta.get("detection", "?")
    lines.append("| role | model | open-vocab | instance masks | weights |")
    lines.append("|---|---|---|---|---|")
    for role in ("detection", "features", "alignment"):
        name = row_meta.get(role)
        if not name:
            continue
        info = cfg.MODELS.get(name)
        # From the probe artefact where there is one, so the table says what the
        # network actually returned rather than what the registry hopes for.
        pr = next((r for r in (probe or {}).get("results", [])
                   if r["model"] == name), {})
        if info:
            vocab = "yes" if info.get("open_vocab") else "**no**"
            # Masks are a detector concern. Reporting them for a feature or
            # alignment model would be meaningless noise, so the column is
            # only filled in for detection.
            masks = ("yes" if info.get("masks") else "**no**") \
                if role == "detection" else "n/a"
            lines.append(f"| {role} | `{name}` | {vocab} | {masks} | "
                         f"{weights_cell(pr, info)} |")
        else:
            state = pr.get("status", "unknown")
            lines.append(f"| {role} | `{name}` | {state} | {state} | {state} |")

    lines += ["", "POGS feeds its detector a `custom_vocab` and requires "
                  "per-instance masks. A model that cannot accept a text "
                  "vocabulary cannot serve that contract however fast it is."]

    if val and val.get("summary"):
        s = val["summary"]
        lines += [
            "",
            "**Detector accuracy on the authors' real capture** "
            "(the only transferable accuracy number in this study, and "
            "scene-independent: the same 14 frames say the same thing for "
            "every scene in the matrix):",
            "",
            f"- mean IoU **{s['iou_mean']:.4f}**, median {s['iou_median']:.4f}, "
            f"best {s['iou_best']:.4f}, worst {s['iou_worst']:.4f}",
            f"- GT labels overlapped: **{s['coverage'] * 100:.1f}%** "
            f"across {s['labels_per_frame']:.1f} labels per frame",
            # n_frames and area_ceiling sit at the top level of the artefact,
            # not in its summary. This read was `s['frames']` and raised
            # KeyError the first time a row actually had a summary to format --
            # dormant until the artefact path above was corrected.
            f"- {val.get('n_frames', '?')} real ZED frames, area ceiling "
            f"{(val.get('area_ceiling') or 0) * 100:.1f}% of frame",
        ]
    else:
        lines += ["", "**Detector IoU: not run.** Run "
                      "`bash run.sh validate`, which scores DETIC over the "
                      "authors' 14 real frames."]

    if not cfg.MODELS.get(det, {}).get("latency_measurable", True):
        lines += [
            "",
            f"**`{det}` has no latency figure.** It only loads inside the POGS "
            "environment, so `bench.py` cannot time it standalone. Every latency "
            "number in section 5a covers the feature and alignment models only.",
        ]
    return "\n".join(lines)


def section_training(train: dict) -> str:
    if not train:
        return "_training did not complete_"
    rows = [
        ("Iterations", f"{train.get('iterations', '?')} / {cfg.POGS_TRAIN_ITERS}"),
        ("Final PSNR", fmt(train.get("psnr"), ".3f")),
        ("Gaussians", fmt(train.get("gaussian_count"), ",d")),
        ("Depth loss", fmt(train.get("depth_loss"), ".5f")),
        ("DINO loss", fmt(train.get("dino_loss"), ".5f")),
        ("DETIC masks cached", fmt(train.get("detic_masks"), ",d")),
        ("Wall clock", fmt(train.get("wall_clock_s"), ".0f", "-") + " s"),
    ]
    out = ["| item | value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    if train.get("smoke"):
        out += ["", f"Smoke test at {train['smoke'].get('iterations')} "
                    f"iterations: {'passed' if train['smoke'].get('ok') else 'FAILED'}."]
    if "determinism" in train:
        d = train["determinism"]
        verdict = "reproducible" if d.get("match") else "**NOT reproducible**"
        out += ["", f"Determinism check (two {d.get('iterations')}-iteration runs, "
                    f"seed {d.get('seed')}): {verdict}, max loss difference "
                    f"{d.get('max_abs_delta', '?')}."]
    return "\n".join(out)


def section_tracking(track: dict) -> str:
    if not track:
        return ("_tracking did not run_\n\nRun `bash run.sh track`. It needs a "
                "finished training run first: `eval_setup` reads the checkpoint "
                "next to the run's `config.yml`.")
    s = track.get("summary") or {}
    out = [
"| item | value |",
        "|---|---|",
        f"| frames | {track.get('n_frames', '?')} |",
        f"| optimizer iterations / frame | {track.get('opt_iters', '?')} |",
        f"| seed | {track.get('seed', cfg.POGS_SEED)} |",
        f"| gaussians tracked / total | {track.get('n_gaussians_kept', '?')} / "
        f"{track.get('n_gaussians_total', '?')} |",
        f"| HDBSCAN clusters (eps {track.get('cluster_eps', '?')}) | "
        f"{track.get('clusters', '?')} |",
    ]

    if "centroid_err_mm_mean" in s:
        out += [
            f"| mean centroid error | {s['centroid_err_mm_mean']:.1f} mm |",
            f"| max centroid error | {s['centroid_err_mm_max']:.1f} mm |",
            f"| frames scored | {s.get('frames_scored', '?')} |",
        ]
    else:
        out.append(f"| mean centroid error | _{s.get('reason', 'not recorded')}_ |")

    if s.get("lost_at_frame") is not None:
        out += [
            f"| **track lost at frame** | **{s['lost_at_frame']}** of "
            f"{track.get('n_frames', '?')} |",
            "",
            f"**This run lost the object at frame {s['lost_at_frame']}.** The "
            "error above is not a tracking accuracy figure: scoring continues "
            "after the loss, so it measures the drift of an estimate that is no "
            "longer following anything. An early loss leaves the estimate near "
            "its frame-0 position, which scores the same as assuming the object "
            "never moved and can therefore look like the best result in the "
            "study rather than the worst.",
        ]

    out += [
        "",
        s.get("note", ""),
        "",
        "Error is **displacement error**: `|(est_i - est_0) - (gt_i - gt_0)|` "
        "against the generator's ground truth, per tracked part. Frame 0 is "
        "exactly zero by construction. Differencing distances instead of "
        "displacements would under-report error for sideways motion, which is "
        "most of what the mug does here.",
        "",
        "This metric is a **single seeded sample** (`n=1`) and is the only "
        "figure in this report affected by POGS's unseeded stochasticity: "
        "`dino_feats` init, `random_quat_tensor`, the random CLIP tile scale and "
        "the datamanager's `randperm` are all unseeded by POGS itself.",
    ]
    return "\n".join(x for x in out if x is not None)


def section_verdict(row_meta: dict, bench: dict, val: dict) -> str:
    name = row_meta.get("detection", "?")
    info = cfg.MODELS.get(name)
    lines = ["**vs row 00 on this data**", ""]
    if info and not info.get("open_vocab"):
        lines.append(f"- **Cannot serve POGS's `custom_vocab` contract.** "
                     f"`{name}` is a closed-set model, so it cannot accept an "
                     f"arbitrary text vocabulary. Latency is reported; "
                     f"capability is not improved.")
    if info and info.get("masks") is False:
        lines.append(f"- **No instance masks.** `{name}` emits boxes only, and "
                     f"POGS consumes per-instance masks.")
    if not lines[2:]:
        lines.append("- Open-vocabulary masks available; a full capability "
                     "comparison is possible.")
    if val and val.get("summary") and val["summary"]["coverage"] == 0:
        lines.append("- **Detector produced no overlap with any GT label on the "
                     "real frames.** This is a finding about the model, not "
                     "about our data.")
    if row_meta.get("blocked") is not None and row_meta.get("status", "").startswith("blocked"):
        lines.append(f"- **Row blocked.** {row_meta.get('status')}")
    lines += ["", NO_BASELINE_CLAIM]
    return "\n".join(lines)


def build(row: str) -> str:
    PATCH_ROW[0] = row
    root = os.path.join(cfg.ARTEFACTS, row)
    probe = load(os.path.join(root, "probe", "weights.json"))

    # Artefacts are per scene: artefacts/<row>/<scene>/{bench,train,track}/.
    # real_14 lives outside the matrix because it is validation-only and has no
    # poses to train on, so it keeps its own slot.
    scenes = [s for s in cfg.SCENES
              if os.path.isdir(os.path.join(root, s))]
    if not scenes:
        scenes = [None]  # nothing run yet; report the single-scene shape

    # Same resolution order as compare.py, from the one list in study_config:
    # the stage that writes this is `detect_check.py --real`, whose output slot
    # is real_14/, not validate/. See cfg.detector_iou_paths.
    val = cfg.load_detector_iou(root)

    per_scene = []
    for scene in scenes:
        base = os.path.join(root, scene) if scene else root
        per_scene.append({
            "scene": scene or "(none)",
            "bench": load(os.path.join(base, "bench", "latency.json")),
            "train": load(os.path.join(base, "train", "training.json")),
            "track": load(os.path.join(base, "track", "tracking.json")),
        })

    bench = per_scene[0]["bench"] if per_scene else None
    train = per_scene[0]["train"] if per_scene else None
    track = per_scene[0]["track"] if per_scene else None
    row_meta = cfg.ROWS.get(row, {})

    parts = [
        f"# {row}",
        "",
        f"*Generated by `report.py`. Every number is read from an artefact "
        f"file; none is entered by hand.*",
        "",
        NO_BASELINE_CLAIM,
        "",
        *([f"> **Which models ran in train/track:** {row_meta['pipeline_note']}", ""]
          if row_meta.get("pipeline_note") else []),
        "## 1. Environment",
        "",
        section_environment((bench or {}).get("environment")),
        "",
        "## 2. Capability",
        "",
        section_capability(row_meta, val or {}, probe or {}),
        "",
        "## 4. Every scene",
        "",
        section_scene_matrix(per_scene),
        "",
        "## 5. Detail: "
        + (per_scene[0]["scene"] if per_scene and per_scene[0]["scene"] else "n/a"),
        "",
        "### 5a. Latency and memory",
        "",
        section_latency(bench),
        "",
        "### 5b. Training",
        "",
        section_training(train),
        "",
        "### 5c. Tracking",
        "",
        section_tracking(track),
        "",
    ]

    patch_text = section_patches(row_meta, bool(row_meta.get("patched")))
    if patch_text:
        parts += ["## 6. What was patched, and why", "", patch_text, ""]

    parts += ["## 7. Errors", ""]
    errors = []
    for s in per_scene:
        for m in (s["bench"] or {}).get("measurements", []):
            if m.get("status") != "ok":
                errors.append((s["scene"], m["model"], m.get("dtype", "-"),
                               m.get("error", "?")))
    if errors:
        parts.append("| scene | model | dtype | error |")
        parts.append("|---|---|---|---|")
        parts += [f"| `{sc}` | `{m}` | {d} | {e} |" for sc, m, d, e in errors]
    else:
        parts.append("_none_")

    parts += ["", "## 8. Verdict", "", section_verdict(row_meta, bench,
                                                         val or {}), ""]
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a row's RESULTS.md")
    ap.add_argument("--row", default=cfg.BASELINE_ROW)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    text = build(args.row)
    out = args.out or os.path.join(cfg.STUDY_ROOT, args.row, f"RESULTS{cfg.RESULTS_SUFFIX}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # utf-8 explicitly: the default is cp1252 on Windows and the study is
    # written on Windows, so an em dash would silently become mojibake in the
    # one file nobody is supposed to hand-edit.
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {out}  ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())