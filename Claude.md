# POGS re-implementation -- working notes

## Scope
- Paper: *Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot
  Manipulation of Irregularly Shaped Objects*, ICRA 2025 (`2025-ICRA-POGS-CRv6.pdf`).
- Reference code: `POGS-author-github/` (nerfstudio splatfacto + gsplat + warp).
- In scope: scan -> train -> group -> track (Sec. IV-A..D). Out of scope: grasp
  planning (Contact-GraspNet), tool servoing, the physical robot.

## Ground rules
1. `v1` = paper/release behaviour. Every default lives in `common/config.py`, with
   the release source in each docstring. Change a default only to match the release.
2. `v2`/`v3` may only add to v1 through configuration or the motion-prior hook of
   `common.evaluate.track_sequence`. Never copy the tracking loop.
3. Ground truth is for evaluation only (`assign_gt_ids`, `match_clusters_to_gt`,
   `cluster_purity`). Never use it to build groups, select objects or initialise.
4. Every result table must include the `Static (no motion)` baseline.
5. A checkpoint is one file (`common/checkpoint.py`) and includes the DINO PCA
   basis. `DinoExtractor` refuses to run without a basis.
6. Run `.venv/bin/python -m pytest -q tests` after touching `common/`.

## Pipeline (what matches the release)
| Stage | Release | Here |
|---|---|---|
| Training | splatfacto, 4000 steps, DINO from step 100, CLIP + grouping from 2500 | same (`common/trainer.py`); gsplat `DefaultStrategy` with splatfacto settings |
| Grad accumulation | `{'color':10,'shs':10,'lerf':3}`, but only `lerf` matches a group name | `lerf: 3` only |
| DINO | ViT-S/14, long side 1260, /10, PCA-64, per-Gaussian feature + 4-layer MLP | same; PCA saved in the checkpoint |
| Grouping | Detic masks (+ negative mask), contrastive loss, cuML HDBSCAN | sim instance masks filtered like Detic (floor/table -> negative mask); sklearn HDBSCAN on a capped subsample (CPU budget) |
| Object selection | interactive click, table removed below `table_z + 0.015` | geometric selection after the same support-plane cut; optional CLIP queries |
| Tracking | ROI per object, DINO L2 + depth (3.7, ignore 0.26 m), Adam 0.004 -> 0.0008, 25 iters, MATCH_RESOLUTION 500, init 100 + 70 | same (`common/tracking.py`) |

## Known deviations (intentional, documented)
- Per-frame depth: the paper uses DINO + depth; the release's `step()` defaults to
  `use_depth=False`. v1 reports both (`POGS (paper ...)`, `POGS (release ...)`).
- No camera-pose optimiser (synthetic poses are exact).
- `render_mask` falls back to half the peak accumulation instead of crashing when
  the 0.9 mask is empty (counted in diagnostics as `roi_mask_fallbacks`).
- Quaternions are re-normalised inside the init loop (the release does not; harmless).
- `apply_group_deltas` composes the K group transforms first and gathers them with
  `index_select` (same maths as the Warp kernel); the tracker transforms only the
  Gaussians it renders. Together ~8x faster than gathering per Gaussian.
- Grouping masks keep the negative mask last. The release's random permutation
  can move it elsewhere, which looks like a bug.

## Benchmark (`scene_generator/pick_place_benchmark.py`)
One 35-view capture of a floor scene (low table + mug, mustard bottle, soup can,
apple, cereal box), one trained model, and many tracking conditions so a modified
tracker can be compared with base POGS per condition. A robot gripper picks the
target, carries it (rotation profile per condition) and places it on the table;
60 frames at 5 Hz, 889x500 static camera. All five objects are tracked;
non-targets must stay put. Success = the paper's 3 cm / 5 deg, per frame.

Factors (one changed at a time from `base` = mug, yaw 30, gripper, 1x, 3 mm depth),
each with 3 random trajectories (seeds):
| Factor | Levels | Tests |
|---|---|---|
| rotation | yaw 15/30/90, tilt 30/60 (unscanned underside), yaw+tilt 45 | Sec. III assumptions |
| speed | frame stride 1/2/4 on `base` and `yaw90` (load time) | "objects had to be moved slowly" (Sec. VI) |
| occlusion | none / gripper / + hand partial (58%) / + hand heavy (100% peak) | Sec. VI occlusion limitation |
| depth | clean / 3 mm / 10 mm + 5% holes (load time) | reliance on stereo depth |
| object | mug / mustard bottle / cereal box | shape generality |
| multi-object | one / two objects moving | simultaneous tracking |

Protocol: only the manipulated object(s) are tracked (the operator clicks the
object to follow, Sec. IV-C). `run_suite(track_all=True)` tracks every selected
object; then the static ones get dragged when the gripper or carried object
passes through their ROI (seen in smoke tests), reported as "Static drift".

Metrics: lead with ADD-S success (< 10% of object diameter) and the lost-frame
rate (> 10 cm); the paper's 3 cm / 5 deg rule is reported but saturates (~11%)
because the gripper tilts the held object. `rescore_runs` back-fills new metrics
into cached rows from their telemetry.

Scene v2 (`SCENE_VERSION = 2`): procedural materials (`scene_generator/materials.py`:
grain, wood, two-tone labels, printed patches, tiles, brushed metal; roughness /
metallic drive the specular). Textures are evaluated in each primitive's own frame;
v1 evaluated them in world coordinates, so a moving object slid under a fixed
pattern (corrupting the DINO features POGS tracks). v2 also puts the table top
exactly at `TABLE_TOP_Z` (it was 6 mm higher, so placed objects sank into it) and
moves the soup can and mug off the table footprint. Data, checkpoints and cached
runs carry the version and are never mixed across versions.

Real-time protocol (`common.evaluate.track_realtime`, `experiments/realtime/`): the
same trajectories rendered at 30 Hz. The tracker starts on frame 0 at t = 0 and
each tracked frame costs its MEASURED compute time; when free it takes the newest
frame and skips the rest (`pick_next_frame`); the Kalman prior predicts over the
true gap. Scored at every camera frame on the last finished estimate
(`estimate_sources`); KF methods also report "+ latency comp." (that estimate
moved to the current time with the filter velocity). Initialisation is not timed
(object at rest). Timings depend on the GPU and on anything else running on it:
run alone, and report the hardware.

Rules: a method name is a key of the shared run cache
(`data_results/benchmark_runs`), so one name = one configuration. Compare methods
on the same cases only, and report seed std next to every mean. A speed claim
needs an equally cheap baseline (same iterations per frame), not just 25-iter POGS.
