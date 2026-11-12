"""Merge every row's RESULTS.md into one comparison table.

This is the file the study is actually read from. It pulls numbers out of each
row's artefact JSON rather than scraping its markdown, so a row that was never
run shows as missing rather than silently omitted.

Usage::

    python compare.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402


def load(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return None


def _blocked_models(bench: dict) -> list[str]:
    """Models that failed to load, once each, with the precision named.

    A model is reported bare only when every precision it was tried at failed.
    If one succeeded, the failing dtype is appended, because "rfdetr" and
    "rfdetr (fp32)" are different findings: the first says the weights are
    unavailable, the second says the model ran and only one precision did not.
    """
    tried: dict[str, list[str]] = {}
    failed: dict[str, list[str]] = {}
    for m in bench.get("measurements", []):
        name, dtype = m["model"], m.get("dtype", "?")
        tried.setdefault(name, []).append(dtype)
        if m.get("status") != "ok":
            failed.setdefault(name, []).append(dtype)

    out = []
    for name, dtypes in failed.items():
        if len(dtypes) == len(tried[name]):
            out.append(name)
        else:
            out.append(f"{name} ({', '.join(dtypes)})")
    return out


def row_summary(row: str) -> dict:
    root = os.path.join(cfg.ARTEFACTS, row)

    # Per-scene artefacts, with a fallback to the pre-matrix flat layout so a
    # report built from an older run is still readable rather than empty.
    scenes = {}
    for scene in cfg.SCENES:
        base = os.path.join(root, scene)
        if os.path.isdir(base):
            scenes[scene] = {
                "bench": load(os.path.join(base, "bench", "latency.json")) or {},
                "train": load(os.path.join(base, "train", "training.json")) or {},
                "track": load(os.path.join(base, "track", "tracking.json")) or {},
            }
    if not scenes:
        scenes["(none)"] = {
            "bench": load(os.path.join(root, "bench", "latency.json")) or {},
            "train": load(os.path.join(root, "train", "training.json")) or {},
            "track": load(os.path.join(root, "track", "tracking.json")) or {},
        }

    # Resolved by study_config so this reader cannot drift from the writer
    # again; see cfg.detector_iou_paths for why the obvious path is the wrong one.
    val = cfg.load_detector_iou(root)

    # Latency is per scene because POGS feeds the models images of that scene's
    # size, so which scene leads is part of the measurement, not a detail. The
    # primary (pickplace-family) scene leads where it was benchmarked; otherwise
    # the first scene that actually carries a bench/, because a row benchmarked
    # only on a pendulum rung should report that number rather than nothing.
    #
    # The scene and its resolution are carried out of here and printed in the
    # table: falling back silently would let one row's 1280x720 figure sit
    # beside another's 640x480 in the same column, which for DINO is a factor of
    # roughly the pixel ratio and would read as a model difference.
    bench_scene = None
    primary = cfg.primary_scene()
    order = ([primary] if primary in scenes else []) + \
            [s for s in scenes if s != primary]
    for s in order:
        if scenes[s]["bench"].get("measurements"):
            bench_scene = s
            break
    if bench_scene is None:
        bench_scene = primary if primary in scenes else next(iter(scenes))
    bench = scenes[bench_scene]["bench"]

    per_model = {}
    for m in bench.get("measurements", []):
        if m.get("status") == "ok" and m.get("dtype") == "fp16":
            per_model[m["model"]] = m["median_ms"]

    det = cfg.ROWS.get(row, {}).get("detection", "?")
    info = cfg.MODELS.get(det, {})
    probe = load(os.path.join(root, "probe", "weights.json")) or {}
    weight_status = {r["model"]: r.get("status") for r in probe.get("results", [])}
    row_models = {m for m in (det, cfg.ROWS.get(row, {}).get("features"),
                              cfg.ROWS.get(row, {}).get("alignment")) if m}
    return {
        "row": row,
        "detection": det,
        "features": cfg.ROWS.get(row, {}).get("features", "-"),
        "alignment": cfg.ROWS.get(row, {}).get("alignment", "-"),
        "open_vocab": info.get("open_vocab"),
        "masks": info.get("masks"),
        "patched": bool(cfg.ROWS.get(row, {}).get("patched")),
        "gated": sorted(m for m in row_models if weight_status.get(m) == "gated"),
        "blocked_models": sorted(m for m in row_models
                                 if weight_status.get(m) == "blocked"),
        "total_ms": bench.get("total_pipeline_median_ms_fp16"),
        "per_model": per_model,
        # Which scene the latency row above was measured on, and at what input
        # size, so a cross-row comparison at mismatched resolution is visible
        # rather than silent. The resolution comes from the artefact itself, not
        # from the scene name, so it reports what was run.
        "bench_scene": bench_scene if bench.get("measurements") else None,
        "bench_res": (bench.get("protocol") or {}).get("scene_resolution"),
        # Once per model, not once per measurement: fp32 and fp16 are separate
        # entries, so a model blocked in both was listed twice ("dinov3,
        # dinov3"). Where only one precision failed, the dtype is named -- the
        # bare model name reads as "no weights" when the truth can be "loaded
        # fine at fp16", which is how rfdetr appeared both here and in the
        # latency table above.
        "blocked": _blocked_models(bench),
        # How much of the pipeline the total actually covers. A row whose
        # feature and alignment models were both blocked still reports a
        # "total", and without this it sits in the same column as a complete one.
        "n_measured": bench.get("measured_models"),
        "n_blocked": bench.get("blocked_models"),
        "iou_mean": (val.get("summary") or {}).get("iou_mean"),
        "iou_coverage": (val.get("summary") or {}).get("coverage"),
        # Per-scene tracking error, keyed by scene. Millimetres, from the
        # tracker's displacement metric: a sub-millimetre distinction is
        # invisible on a 0.44 m travel and the whole question is whether a swap
        # is better by a few mm.
        "err_by_scene": {
            s: ((d["track"].get("summary") or {}).get("centroid_err_mm_mean"))
            for s, d in scenes.items()
        },
        # The frame the tracker lost the object at, per scene, or None if it
        # held. Carried into the table because an error figure from a lost run
        # is not a measurement of anything: `track_offline.py` keeps scoring
        # after the loss, so a run that dies at frame 0 never moves its estimate
        # and scores the "assume no motion" error -- which on a small-amplitude
        # scene is the *lowest* number in the column. Sorting this table without
        # the flag ranks failures first.
        "lost_by_scene": {
            s: ((d["track"].get("summary") or {}).get("lost_at_frame"))
            for s, d in scenes.items()
        },
        "err_mm_max": max(
            [v for v in
             ((d["track"].get("summary") or {}).get("centroid_err_mm_max")
              for d in scenes.values()) if v is not None] or [None]
        ),
        "scenes_run": [s for s, d in scenes.items() if d["track"] or d["train"]],
        "opt_iters": next(
            (d["track"].get("opt_iters") for d in scenes.values()
             if d["track"].get("opt_iters")), None),
    }


def tick(flag) -> str:
    if flag is None:
        return "?"
    return "yes" if flag else "**no**"


def build() -> str:
    rows = [row_summary(r) for r in cfg.ROWS]
    done = [r for r in rows if r["total_ms"] is not None]
    base = next((r for r in done if r["row"] == cfg.BASELINE_ROW), None)

    out = [
        "# POGS model-substitution study - comparison",
        "",
        "*Generated by `compare.py` from each row's artefacts. Do not edit by "
        "hand; regenerate with `python shared/compare.py`.*",
        "",
        "> **Scope.** Every row is measured against row 00 on the same "
        "synthetic data. The POGS authors released no trained POGS checkpoint "
        "and no capture scene, so there is no published baseline and no row "
        "here claims to beat POGS as published.",
        "",
        "## Rows",
        "",
        "| row | detection | features | alignment | open-vocab | masks | POGS | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        status = "measured" if r["total_ms"] is not None else (
            "blocked models" if r["blocked"] else "not run")
        # Weight availability is a per-row fact, not a footnote: a row whose
        # detector has no weights cannot produce a number at all, and reading
        # that off the row name alone takes a lookup.
        if r["blocked_models"]:
            status += f" ({', '.join(r['blocked_models'])} BLOCKED)"
        elif r["gated"]:
            status += f" ({', '.join(r['gated'])} gated)"
        out.append(
            f"| `{r['row']}` | `{r['detection']}` | `{r['features']}` | "
            f"`{r['alignment']}` | {tick(r['open_vocab'])} | "
            f"{tick(r['masks'])} | "
            f"{'patched' if r['patched'] else '**stock**'} | {status} |"
        )

    out += [
        "",
        "`stock` means the row runs an unmodified POGS tree, which is what makes "
        "row 00 the reference. `patched` means stock POGS hardcoded the models it "
        "uses and the substitution had to be expressed as a patch; each row's "
        "`RESULTS.md` section 6 carries the diff.",
    ]

    out += ["", "## Latency - deterministic", ""]
    if not done:
        out.append("_no row has been benchmarked yet_")
    else:
        out += ["| row | scene | input | roles measured | total fp16 (ms) "
                "| vs baseline | per-model fp16 (ms) |",
                "|---|---|---|---|---|---|---|"]
        for r in done:
            delta = ""
            # Only ratio against the baseline when both were measured on the
            # same input size. A 1280x720 total over a 640x480 one is a
            # resolution ratio with a model ratio hidden inside it.
            if base and r is not base and base["total_ms"]:
                if r["bench_res"] == base["bench_res"]:
                    ratio = r["total_ms"] / base["total_ms"]
                    delta = (f"{ratio:.2f}x"
                             + (" (faster)" if ratio < 1 else " (slower)"))
                else:
                    delta = "n/a - different input"
            per = ", ".join(f"{k} {v:.1f}" for k, v in
                            sorted(r["per_model"].items(), key=lambda kv: -kv[1]))
            n_roles = len(r["per_model"])
            roles = f"{n_roles}/3"
            if r["n_blocked"]:
                roles = f"**{roles}**"
            out.append(f"| `{r['row']}` | `{r['bench_scene'] or '-'}` "
                       f"| {r['bench_res'] or '?'} | {roles} "
                       f"| {r['total_ms']:.2f} | {delta or '-'} | {per} |")

        # The baseline's total covers 2 of 3 roles because DETIC only loads
        # inside the POGS environment and bench.py cannot time it standalone.
        # Ratioing a 3-role total against it therefore charges the alternative
        # for a detector the baseline was never billed for. The column above
        # makes the mismatch visible; this says what it means.
        if base and any(len(r["per_model"]) != len(base["per_model"])
                        for r in done if r is not base):
            out += [
                "",
                f"**`vs baseline` compares totals over different numbers of "
                f"roles.** The baseline's covers "
                f"{len(base['per_model'])} of 3 - `"
                f"{cfg.ROWS.get(cfg.BASELINE_ROW, {}).get('detection', 'detic')}`"
                " only loads inside the POGS environment, so `bench.py` cannot "
                "time it standalone - while a row with its own benchable "
                "detector covers 3. The ratio therefore charges that row for a "
                "stage the baseline was never billed for. Compare per-model "
                "figures role by role instead; the detector's in-pipeline cost "
                "is in each row's training log, not here.",
            ]
        if any(r["n_blocked"] for r in done):
            out += [
                "",
                "**A bold `roles measured` means the row's `total` is not a "
                "pipeline total.** Models that failed to load are absent from "
                "the sum, so a row missing its feature model looks cheap rather "
                "than unmeasured. The missing ones are listed under *Blocked "
                "models* below.",
            ]
        if base is None:
            out += [
                "",
                f"**`{cfg.BASELINE_ROW}` was not benchmarked on this scene "
                "set**, so there is nothing to ratio against and `vs baseline` "
                "is empty for every row. Its figures are in `RESULTS.md`, "
                "measured on `pickplace` at the same 1280x720.",
            ]

        measured_res = {r["bench_res"] for r in done if r["bench_res"]}
        if len(measured_res) > 1:
            out += [
                "",
                "**These rows were not all benchmarked on the same input size** "
                f"({', '.join(sorted(measured_res))}), because a row is "
                "benchmarked on the scenes it was run on. Latency scales with "
                "input, so compare the per-model figures within one `input` "
                "value and not down the `total` column; `vs baseline` is left "
                "blank where the sizes differ.",
            ]

    out += ["", "## Detector accuracy on the authors' real capture - "
            "deterministic", "",
            "| row | mean IoU | GT labels overlapped |",
            "|---|---|---|"]
    for r in rows:
        if r["iou_mean"] is None:
            out.append(f"| `{r['row']}` | not run | not run |")
        else:
            cov = (f"{r['iou_coverage'] * 100:.1f}%"
                   if r["iou_coverage"] is not None else "?")
            out.append(f"| `{r['row']}` | {r['iou_mean']:.4f} | {cov} |")

    out += ["", "## Tracking error per scene - stochastic", ""]
    out += ["Centroid error in mm, lower is better. **-** means that row has not "
            "been tracked on that scene yet. **`lost@N`** means the tracker lost "
            "the object at frame N, so that cell is not a measurement - see "
            "below.", "",
            "| row | " + " | ".join(f"`{s}`" for s in cfg.SCENES) + " | scenes |",
            "|---" * (len(cfg.SCENES) + 2) + "|"]
    for r in rows:
        cells = []
        for s in cfg.SCENES:
            v = r["err_by_scene"].get(s)
            if v is None:
                cells.append("-")
                continue
            lost = r["lost_by_scene"].get(s)
            cells.append(f"{v:.1f}" if lost is None
                         else f"{v:.1f} **lost@{lost}**")
        n_lost = sum(1 for s in cfg.SCENES
                     if r["lost_by_scene"].get(s) is not None)
        count = f"{len(r['scenes_run'])}/{len(cfg.SCENES)}"
        if n_lost:
            count += f", **{n_lost} lost**"
        out.append(f"| `{r['row']}` | " + " | ".join(cells) + f" | {count} |")

    out += [
        "",
        "Centroid error is **displacement error against the generator's ground "
        "truth**, `|(est_i - est_0) - (gt_i - gt_0)|`, per tracked part, "
        "averaged only over frames where the object actually moved. Lower is "
        "better. Every row must use the same `opt_iters` "
        f"({cfg.POGS_TRACK_OPT_ITERS}, as in stock POGS demos) for the "
        "comparison to mean anything.",
        "",
        "The pendulum columns are a difficulty ladder: the cylinder's peak "
        "displacement grows with the angle, so a row that holds up at 60 degrees "
        "is doing something a row that only holds at 5 is not. `pickplace` is "
        "the occlusion case, a different kind of difficulty rather than a "
        "bigger one.",
        "",
        "**A `lost@N` cell cannot be compared with one that held, in either "
        "direction.** The tracker goes on scoring after it loses the object, so "
        "the figure is the drift of an estimate that is no longer following "
        "anything. Lost late, that inflates the error; lost early, it *deflates* "
        "it, because an estimate frozen near frame 0 scores the same as assuming "
        "the object never moved - and on a small-amplitude scene that is the "
        "lowest number the column can produce. Read those cells as failures "
        "whatever their value, and do not rank on them.",
    ]

    out += ["", "## Blocked models", ""]
    any_blocked = False
    for r in rows:
        if r["blocked"]:
            any_blocked = True
            out.append(f"- `{r['row']}`: {', '.join(f'`{b}`' for b in r['blocked'])}")
    if not any_blocked:
        out.append("_none_")

    out += [
        "",
        "## Reading this table",
        "",
        "- **Latency and detector IoU are deterministic.** They involve no "
        "training, so they carry no seed variance and small differences are "
        "real.",
        "- **Centroid error and PSNR are stochastic.** POGS seeds nothing of "
        "its own - `dino_feats` init, `random_quat_tensor`, the datamanager's "
        "`randperm` calls and HDBSCAN clustering are all unseeded - so these "
        "are one sample at `machine.seed = 42`.",
        "- **A model that is not open-vocabulary cannot serve POGS**, however "
        "fast it is. POGS passes a `custom_vocab` to its detector and consumes "
        "per-instance masks.",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge all rows into one report")
    ap.add_argument("--out", default=os.path.join(cfg.STUDY_ROOT, f"RESULTS{cfg.RESULTS_SUFFIX}.md"))
    args = ap.parse_args()

    text = build()
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {args.out}  ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())