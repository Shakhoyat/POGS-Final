# model_study_results — results-only archive

A copy of `model_study/` carrying **only the measurements and their provenance**, in
the original folder structure. 365 files, ~7 MB, extracted 2026-11-08 from a
13 GB source tree. Every file is byte-identical to its original (md5-verified),
and all 263 JSON files parse.

This is an **evidence archive, not a runnable checkout** — see *What is not here*.

---

## Start here

| file | what it is |
|---|---|
| `COMPARISON_STUDY.md` | **Read this first.** The comparison analysis: what each row actually ran, why the tracking metric carries no signal on the pendulum scenes, and the per-combination verdicts. Numbers recomputed from the JSON in this archive. |
| `RESULTS.md` | merged comparison, v1 scene set (`pendulum_05…60` + `pickplace`) |
| `RESULTS_v2.md` | merged comparison, v2 scene set (`pendulum_v2_*` + `pickplace_v2`) |
| `RESULTS_handle.md` | merged comparison, `pickplace_handle` only |
| `README.md` | the original study README: design, protocol, scene matrix, known risks |
| `<row>/RESULTS*.md` | per-row reports, one per scene set. **Section 6 of each carries the patch diff; the banner at the top states which models really ran in train/track** — that caveat is not in the merged tables. |

---

## Layout

```
model_study_results/
├── COMPARISON_STUDY.md   RESULTS.md   RESULTS_v2.md   RESULTS_handle.md
├── README.md             MANIFEST.md (this file)
├── 00_baseline_detic_dinov2_clip/   RESULTS.md, RESULTS_v2.md, RESULTS_handle.md
├── 01_locateanything_dinotxt/       (same three)
├── 02_rfdetr_dinov3_mobileclip/     (same three)
├── 03_vfmdetr_siglip2/              (same three)
├── 04_yolo/                         (same three)
├── artefacts/            287 files — the machine-readable results
│   └── <row>/<scene>/
│       ├── bench/latency.json              per-model median/p95 ms, peak VRAM, params
│       ├── bench/locateanything_detect.json  row 01 only: per-frame box IoU
│       ├── train/training.json             iterations + checkpoint path
│       ├── train/<scene>/pogs/<ts>/config.yml   the full nerfstudio TrainerConfig
│       ├── track/tracking.json             per-frame centroid error + ground truth
│       ├── capture_detect.json             DETIC mask counts over the capture ring
│       ├── real_14/detector_real_iou.json  DETIC IoU on the authors' real frames
│       ├── probe/weights.json              per-model availability (available/gated/blocked)
│       └── sweep.txt, patches_applied.txt
├── incomplete/           13 files — partial runs, kept as evidence
├── inputs/<scene>/       44 files — ground truth only
│   ├── transforms.json           capture-ring camera poses + intrinsics
│   └── tracking/gt_poses.json    per-frame GT centroid, quaternion, mask fraction
└── weights/manifest.json  URL, byte count and sha256 per weight file
```

Artefact counts per row: 00 → 135, 01 → 68, 02 → 21, 03 → 50, 04 → 12.
The spread is the study's coverage, not an extraction artefact: row 00 was swept
most, row 04 was benchmarked only.

---

## The three files that carry most of the signal

**`artefacts/<row>/<scene>/track/tracking.json`** — the only per-frame data here.
Each frame record holds `gt_centroid`, `gt_displacement_mm`, `est_centroids` and
`centroid_err_mm`, plus `phase` on the pickplace scenes. `summary` adds
`centroid_err_mm_mean/max`, `frames_scored` and **`lost_at_frame`**.

> `lost_at_frame` is not null in 6 of 48 runs, and scoring continues after the
> loss. An early loss leaves the estimate frozen near frame 0, which scores the
> same as assuming the object never moved — so it produces an artificially
> **low** error, not a high one. `01/pendulum_05` reports 9.0 mm, the lowest
> number in the study, with `lost_at_frame: 0`. Filter on this field before any
> aggregation.

**`artefacts/<row>/<scene>/train/<scene>/pogs/<ts>/config.yml`** — the
authoritative record of which models a run used. `network=OpenCLIPNetworkConfig`
vs `SigLIP2NetworkConfig` is how you tell a real substitution from a baseline
rerun; the row name does not tell you. Where a scene has several timestamped
runs, the one named in `track/tracking.json`'s `config` field is the one that was
tracked.

**`inputs/<scene>/tracking/gt_poses.json`** — exact ground truth (closed form to
3e-8 m, pivot drift 0). Needed to compute a no-motion baseline, which is what
shows the pendulum errors to be noise.

---

## Three things to know before reusing these numbers

1. **Rows 01 and 02 did not substitute anything in train/track.** Their configs
   show the baseline's DETIC + DINOv2 + OpenCLIP ViT-B-16. Their tracking numbers
   are baseline reruns, which makes them a *variance estimate*, not a result for
   LocateAnything3D, dino.txt, RF-DETR, DINOv3 or MobileCLIPv2. Only row 03
   (SigLIP2 alignment) is a genuine substitution. Those five models were measured
   standalone — see `bench/latency.json` and `01_*/LOCATEANYTHING_DETECT.md`.
2. **Tracking error is n=1 and the noise is larger than any row difference.**
   Baseline `pickplace` at seeds 7/13/42 gives 668.7 / 159.8 / 63.4 mm — SD
   325 mm, against between-row differences of 17–60 mm. Those three runs are
   under `artefacts/00_*/pickplace_seed7`, `_seed13` and `pickplace`.
3. **Latency and detector IoU are deterministic and reusable as-is.** No training
   involved; repeat measurements of the same model differ by under 1 ms.

---

## What is not here

Excluded to keep the archive portable. None of it is a measurement.

| excluded | size | why |
|---|---|---|
| `weights/` blobs | ~13 GB | re-downloadable; `weights/manifest.json` keeps URL + sha256 |
| `*.ckpt` (63 checkpoints) | GBs | trained models, regenerable from config + inputs |
| `inputs/` frames: `img/`, `depth/`, `depth_png/`, `obj_id/`, `poses/`, `*.ply` | ~4 GB | scene pixel data; the JSON ground truth is kept |
| `POGS/` (5 trees), `deps/` | large | source checkouts, unmodified upstream + patches |
| `*/outputs/*/dino.npy` | 131 MB | cached DINO features |
| `artefacts/**/clusters.npy` (48) | 58 MB | HDBSCAN assignments per gaussian — **not** regenerable without the checkpoints. Ask if you need them. |
| `train.log` (51 files) | ~32 MB | raw logs. The model identity in them is also in `config.yml`; the DETIC and embedding throughput figures quoted in `COMPARISON_STUDY.md` §4b/§4c came from these, so pull them across if you want to re-derive those. |
| `shared/*.py`, `run.sh`, `patch/` | small | pipeline code. **This archive cannot be re-run.** Each patch's full diff is preserved in `artefacts/*/patches_applied.txt`. |

Regenerating the merged tables needs `shared/compare.py`, which is not here. The
reports in this archive were generated before extraction and are current.

---

## Reading the artefacts

```bash
# every tracking result, with its lost-track flag  (49 files / 48 runs)
python - <<'EOF'
import json, glob, os
for p in sorted(glob.glob("artefacts/*/*/track/*.json")):
    d = json.load(open(p)); s = d["summary"]
    parts = p.replace(os.sep, "/").split("/")   # glob yields \ on Windows
    row, scene = parts[1], parts[2]
    print(f"{row:30s} {scene:18s} {s['centroid_err_mm_mean']:7.1f} mm"
          f"  iters={d['opt_iters']} seed={d['seed']}"
          f"{'  LOST@%s' % s['lost_at_frame'] if s.get('lost_at_frame') is not None else ''}")
EOF
```

```bash
# which alignment encoder each run really used
grep -l "SigLIP2NetworkConfig" artefacts/*/*/train/*/pogs/*/config.yml
```

Two caveats when aggregating: `opt_iters` is 15 everywhere **except**
`artefacts/00_*/pickplace/track/tracking.json`, which is 30 — its 15-iteration
counterpart is `tracking_iters15.json` in the same folder, and that is the one to
use for comparison. And `pendulum_v2_01` is a null control (1°, 2.6 mm of motion);
its 42.5 mm error is the tracker's resting noise floor, not a result.
