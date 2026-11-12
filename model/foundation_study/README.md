# POGS model-substitution study

Which 2D foundation models, if any, beat POGS's own choice?

| role | row 00 (baseline) | row 01 | row 02 | row 03 | row 04 |
|---|---|---|---|---|---|
| **detection** | DETIC | LocateAnything3D | RF-DETR | VFM-DETR | YOLO |
| **features** | DINOv2 ViT-S/14 | DINOv2 Meets Text | DINOv3 | VFM-DETR | DINOv2 ViT-S/14 |
| **alignment** | CLIP ViT-B-16 | DINOv2 Meets Text | MobileCLIPv2 | SigLIPv2 | CLIP ViT-B-16 |

Rows 01 and 03 have merged cells in the source table, so those models span two
roles: dino.txt does dense features *and* text alignment in one model, and
VFM-DETR does detection *and* features.

## Status: where each row actually stands

| row | folder | weights | POGS tree | can run end to end? |
|---|---|---|---|---|
| 00 DETIC + DINOv2 + CLIP | ready | all 3 available | **stock** | yes |
| 01 LocateAnything3D + dino.txt | ready | LocateAnything3D available, dino.txt gated | patched | with `HF_TOKEN` |
| 02 RF-DETR + DINOv3 + MobileCLIP2 | ready | RF-DETR + MobileCLIP2 available, DINOv3 gated | patched | with `HF_TOKEN` |
| 03 VFM-DETR + SigLIP2 | ready | SigLIP2 available, **VFM-DETR blocked** | stock | no, blocked model |
| 04 YOLO | ready | all available | stock | latency + gap only |

Two rows are gated on an HF licence, one on a model whose weights do not exist
publicly. Those are properties of the upstream projects, not of this study, and
each is recorded with its reason rather than worked around.

---

## What you can and cannot conclude here

**Can:** compare these rows against each other, on identical data, with a fixed
protocol.

**Cannot:** compare against POGS as published. The authors released **no trained
POGS checkpoint and no capture scene** - their Drive `checkpoints/` folder holds
four Contact-GraspNet models in TensorFlow format, which is the grasp network,
not the 3DGS model. There is no published baseline to measure against. This
constraint is enforced in the report text, not left to the reader.

## The catch that shapes the whole study

POGS passes a `custom_vocab` to its detector and consumes **per-instance
masks**. That is what lets it track an arbitrary object rather than a fixed
class list. Of the alternatives in the table:

| model | instance masks | open-vocabulary | can serve POGS? |
|---|---|---|---|
| DETIC | yes | yes | **yes - baseline** |
| dino.txt | n/a (feature model) | yes | as alignment, yes |
| RF-DETR | yes (`-Seg-*`) | no, COCO only | no |
| DINOv3 | n/a (feature model) | no | as feature, yes |
| MobileCLIP2 | n/a (alignment model) | yes | **yes, no adapter needed** |
| SigLIP2 | n/a (alignment model) | yes | **yes, no adapter needed** |
| YOLO | no, boxes only | no | no |
| LocateAnything3D | **no**, emits 3D boxes | yes | no |
| VFM-DETR | unknown - no weights | unknown | cannot be evaluated |

Two results are already clear without running anything, and they are worth
stating because they are the kind of thing that takes a GPU day to discover:

- **MobileCLIP2 and SigLIP2 drop straight into POGS's alignment slot.** Both
  load through the existing `open_clip` call with a different architecture
  string; no adapter, no patched tree. Row 02's alignment is genuinely free.
- **LocateAnything3D cannot serve POGS's mask contract.** Its weights are public
  and ungated (`nvidia/LocateAnything-3B`, ~9.5 GB), but it is a 3B Eagle-2.5 VLM
  emitting 3D boxes through Chain-of-Sight decoding, not per-instance 2D masks.
  Row 01 records the gap rather than bridging it.

Rows 02, 03 and 04 are therefore reported as **latency plus a documented
capability gap**, not as drop-in replacements. No adapter engineering was
attempted; the gap is the finding.

## The second finding: POGS hardcodes its models

Stock POGS cannot be told which models to use through configuration:

- `dino_dataloader.py:23`  -  `dino_model_type = "dinov2_vits14"` is a **class
  attribute**. `cfg` is read only for `image_shape` (lines 95-96), so no config
  value reaches it.
- `detic_dataloader.py:76-77`  -  the DETIC config path and weights URL are
  literals.
- `pyproject.toml`  -  pins `transformers==4.44.0`, which predates DINOv3.

So a substitution is a patch, not a flag. Rows 01 and 02 carry one in
`patch/`, applied by their `run.sh` at setup time and logged to
`artefacts/<row>/patches_applied.txt` so the report can print the real diff.
Every patch defaults to upstream behaviour, so an unpatched and a patched tree
produce identical results when the environment says nothing.

Row 00 stays **stock**. That is what makes it the reference: if the baseline
needed a patch to run, the comparison would have no fixed point.

One gotcha the patches had to handle: every `.py` in the upstream tree uses CRLF
line endings (46 of 46 files checked), so `patch` needs `--binary` and the patch
files themselves must be CRLF. Without that they fail with "different line
endings" while looking otherwise correct.

## Layout

```
model_study/
|-- RESULTS.md                       the merged comparison. Read this one.
|-- shared/
|   |-- row_common.sh                the run.sh body every row sources
|   |-- study_config.py              paths, resolutions, model registry, rows
|   |-- download_weights.py          fetch + verify + manifest, per row
|   |-- prefetch.py                  warm model caches so timing is not a download
|   |-- probe_weights.py             availability, so a row can report `blocked`
|   |-- bench.py                     latency / VRAM
|   |-- detect_check.py              does the detector find the mug?
|   |-- validate_detector.py         IoU on the authors' real capture
|   |-- track_offline.py             headless tracking, scored against GT
|   |-- test_track_offline.py        host-independent checks (no GPU needed)
|   |-- test_detect_check.py         host-independent checks (no GPU needed)
|   |-- test_download_weights.py     host-independent checks (no GPU needed)
|   |-- test_scenes.py               the whole scene matrix loads (no GPU needed)
|   |-- report.py                    -> per-row RESULTS.md
|   `-- compare.py                   -> the merged RESULTS.md
|-- inputs/
|   |-- pendulum_05 .. pendulum_60/  9 angles, the difficulty ladder
|   |   |                            36 capture views + 60 tracking frames
|   |   |                            640x480, exact closed-form ground truth
|   |   `-- tracking/                left/, depth/, obj_id/, gt_poses.json
|   |-- pickplace/                   PRIMARY - mug carried by a gripper
|   |   `-- tracking/                120 frames, same layout
|   `-- real_14/                     authors' test_data -> detector IoU ONLY
|-- deps/                            nerfstudio + Detic + detectron2 submodules
|-- weights/                         model weights, one folder per model
|-- artefacts/                       machine-readable results per row
|-- 00_baseline_detic_dinov2_clip/   DETIC      + DINOv2  + CLIP
|   |-- POGS/                        pristine official tree - UNMODIFIED
|   |-- run.sh
|   `-- RESULTS.md
|-- 01_locateanything_dinotxt/       LocateAnything3D + dino.txt (features AND alignment)
|   |-- POGS/                        patched at setup from patch/
|   |-- patch/                       the diff, applied and logged by run.sh
|   |-- run.sh
|   `-- RESULTS.md
|-- 02_rfdetr_dinov3_mobileclip/     RF-DETR + DINOv3 + MobileCLIP2
|   |-- POGS/  patch/  run.sh  RESULTS.md
|-- 03_vfmdetr_siglip2/              VFM-DETR (BLOCKED) + SigLIP2
|   |-- POGS/  run.sh  RESULTS.md
`-- 04_yolo/                         YOLO detection only
    |-- POGS/  run.sh  RESULTS.md
```

Each row is self-contained: its own pristine POGS tree, its own `run.sh`, its own
`RESULTS.md`. Copy the whole `model_study` folder to the GPU host once, then run
whichever row you want:

```bash
cd model_study/00_baseline_detic_dinov2_clip && bash run.sh
cd model_study/02_rfdetr_dinov3_mobileclip && bash run.sh
```

Row `run.sh` files are 30-50 lines each and source `shared/row_common.sh`, which
holds the whole pipeline. One implementation, not five: a fix to the training or
tracking path lands in every row at once instead of the five copies drifting.

POGS trains on `capture/` (the 36-view ring, with `transforms.json` and
`sparse_pc.ply`) and tracks on `tracking/` (one fixed camera). Separate folders
deliberately: the capture ring must stand far enough back to clear the table,
which would shrink the mug below the size the tracking views achieve, so the two
use different focal lengths.

## The scene matrix

Every row runs the same 10 scenes, because the angle is the study's independent
variable:

| scene | capture | tracking | resolution | focus object | difficulty axis |
|---|---|---|---|---|---|
| `pendulum_05` .. `pendulum_60` | 36 views | 60 frames | 640x480, f=420 | red cylinder, 0.96% of frame | peak displacement grows 1.3 cm -> 15 cm |
| `pickplace` | 36 views | 120 frames | 1280x720, f=1400 | red mug, 2.3% of frame | occlusion up to 36.1% of the silhouette |

Artefacts are scoped per scene: `artefacts/<row>/<scene>/{bench,train,track}/`,
so a sweep never overwrites itself and a missing rung is visibly missing rather
than silently absent. `real_14` keeps its own slot outside the matrix because it
is validation-only.

```bash
bash run.sh sweep              # train + track on all 10
bash run.sh sweep bench        # a different stage per scene
SCENE=pendulum_60 bash run.sh train
```

A sweep records `artefacts/<row>/sweep.txt` with per-scene outcome and minutes,
and exits non-zero if any scene failed. It does **not** stop at the first
failure - a 40-run sweep that dies at run 27 with no record of the other 39 is
worse than one that finishes and reports.

The pendulum ladder is the cheap, exact end of the range: ground truth matches
closed form to 3e-8 m and pivot drift is exactly 0. `pickplace` is the hard end.
A row that holds at 60 degrees is doing something a row that only holds at 5 is
not.

## Weights

One downloader for all 12 registry models, per row:

```bash
python shared/download_weights.py --row 00_baseline_detic_dinov2_clip
python shared/download_weights.py --all        # ~13 GB
python shared/download_weights.py --verify     # check, download nothing
```

- Resumable via a `.part` file and HTTP `Range`, so LocateAnything3D's 4.7 GB
  shard does not restart on every blip
- `--verify` exits non-zero with `--strict` if a required file is missing
- `weights/manifest.json` records URL, byte count and sha256 per file, so a run
  on the GPU host can prove it used the same bytes this machine downloaded
- Gated models (`dinov3`, `dino_txt`) need `HF_TOKEN`; the token is sent only to
  `huggingface.co`, never to Meta or GitHub
- Sits outside `artefacts/` on purpose: artefacts are measurements and get
  regenerated, weights are 13 GB and must never be deleted by a re-run

`probe_weights.py` reads its URL table from the downloader rather than keeping a
second copy. An earlier version kept two and they drifted: `mobileclipv2` and
`locateanything3d` were listed with no URL, so both were reported blocked when
MobileCLIP2 is in fact public and ungated.

## The two scenes, and why there are two

**`pickplace` is the primary input.** A mug is lifted 0.15 m off a table by a
gripper, carried 0.44 m across it with a 15 deg yaw, and set down again - 120
frames in four phases. This is the scenario POGS exists for, and it fixes the
three weaknesses that made the pendulum a poor *model* test:

| | pendulum (9 angles) | pickplace |
|---|---|---|
| detector's class | no LVIS/COCO match - no "pendulum" | **mug**: COCO `cup`, LVIS `mug` |
| silhouette | 0.96% of frame | **2.3-3.6%** |
| background | 85% pure black | a room: wall, floor, table |
| difficulty axis | peak displacement 1.3 -> 15 cm | **occlusion up to 36.1%** |
| occlusion | none | **gripper hides up to 36.1% of the mug's silhouette** |
| ground truth | exact (closed form) | exact |
| seed cloud | byte-identical between runs | byte-identical between runs |

The gripper's jaws close on +/-Y, the axis the tracking camera looks along, so a
jaw sits between the lens and the mug's face. A side grip on +/-X is the
physically ordinary choice and was tried first, but it only occluded the mug's
edges - 11.4% of the silhouette. The mug's own visible share of the silhouette is
2.3-3.3% in the worst frames, which is deliberate: it is the hard version of the
occlusion test.

All nine pendulum angles are in the matrix, not just one. Their ground truth is
exact (centroids match closed form to 3e-8 m, pivot drift exactly 0) and
`default_rng(0)` makes each seed cloud byte-identical between runs, so they stay
the right instrument for "did the pipeline change?" - and, as a ladder, they also
answer "does the answer hold as the task gets harder?", which a single angle
cannot.

`deps/` holds the three submodules the training path actually needs. The other
three (`ur5py`, `raftstereo`, `contact_graspnet`) are robot-arm and grasp paths
this study never touches, so they are excluded to keep the transfer small.
`run.sh` recreates them as symlinks on the destination machine, because symlinks
made on Windows do not survive the move.

## Running it

Move `model_study` to the GPU host, then pick a row:

```bash
cd model_study/00_baseline_detic_dinov2_clip
bash run.sh
```

Stages, identical in every row because they all source `shared/row_common.sh`:

```bash
bash run.sh setup      # conda env, deps, POGS install, patches, import check
bash run.sh weights    # download this row's weights
bash run.sh prefetch   # warm caches so timing is not a download
bash run.sh smoke      # 50 iterations twice, to verify determinism
bash run.sh detect     # does the detector find the object? ~1 min. Run early.
bash run.sh train      # full 4000
bash run.sh bench      # latency and memory
bash run.sh validate   # detector IoU on the authors' real frames
bash run.sh track      # headless tracking, scored against ground truth
bash run.sh report     # RESULTS.md + merged comparison
bash run.sh sweep      # train + track on all 10 scenes
SKIP_TRAIN=1 bash run.sh all    # everything except training and tracking
SCENE=pendulum_60 bash run.sh train     # one scene, one stage
bash run.sh sweep bench                # a different stage per scene
bash run.sh track --max-frames 20      # flags pass through to the tracker
```

`setup` creates its own conda env per row (`pogs_env`, or `pogs_env_r01` etc. where
a newer `transformers` or an extra package is needed), symlinks `deps/` into
`POGS/pogs/dependencies/`, applies that row's patches, and then runs an
import-chain check that exits non-zero rather than letting training start and
fail obscurely.

Every `run.sh` and `row_common.sh` passes `bash -n` on Linux. The sweep is driven
end to end with fakes on `PATH` standing in for `ns-train` and `python`, so the
real dispatch, the real per-scene artefact paths and the real failure handling
run unmodified. That harness is what caught three bugs `bash -n` cannot see:

- `BASH_SOURCE[0]` inside a sourced file names the *sourced* file, so `HERE`
  resolved to `shared/` and `POGS` to a directory that does not exist. Every row
  now sets `ROW_DIR` before sourcing.
- `INPUT` and `ART` were computed once at entry, so a sweep's per-scene `SCENE`
  never reached them and all ten scenes trained on the first one - reported as
  ten successes. They are now derived by `set_scene` at the point of use.
- `stage_train`, `stage_smoke` and `stage_detect` ended failures with `die`,
  which exits the shell, so the first failing scene killed the sweep before
  `sweep.txt` was written. They `return` now, and the sweep records and continues.

### A suggested order

Row 00 first, and stop at the first two gates:

```bash
cd model_study/00_baseline_detic_dinov2_clip
bash run.sh setup      # ~30 min; fails here if the CUDA stack is wrong
bash run.sh detect     # ~1 min; the real go/no-go
```

Then `smoke`, then either one scene (`bash run.sh train track`) or the whole
ladder (`bash run.sh sweep`). Start with `SCENE=pendulum_30 bash run.sh train
track` before committing to ten runs - it is the exact, cheap scene, so a failure
there is the pipeline and not the difficulty.

Nothing about rows 01-04 is worth debugging until row 00 has proven the pipeline
works end to end on the 4070.

### GPU cost of the full ladder

10 scenes x 4000 iterations. Budget roughly 5-7 hours per row for
`bash run.sh sweep`, so the five-row study is on the order of a day and a half of
4070 time, plus row 01's ~10 GB of weight downloads. `bash run.sh sweep bench`
gives the latency table for every scene in minutes and costs no training, so it is
the cheap first pass across all rows.

### Detection: run this before the long run

POGS trains on DETIC masks. `FullImageDatamanager` builds a `DeticDataloader` at
`full_images_datamanager.py:174` and feeds its masks into the CLIP/lerf loss from
`lerf_step = 2500`. If DETIC returns nothing on our capture, POGS asserts
`"No masks found"` at line 437 - and that is most of the way into a 40-minute
run.

`bash run.sh detect` answers that in about a minute and **aborts the run** if the
answer is no. `detect_check.py` calls POGS' own `DeticDataloader.predict()`, so it
applies the stock config, the LVIS vocabulary, `SCORE_THRESH_TEST = 0.5` and the
`H*W/3.5` area ceiling that training applies, and cannot drift from it.

```bash
python shared/detect_check.py                    # our capture, stock LVIS
python shared/detect_check.py --vocab mug cup    # prompted, for diagnosis
python shared/detect_check.py --real             # authors' frames -> IoU
python shared/detect_check.py --limit 3          # quick smoke
```

Read the result carefully. `frames_with_mug_sized_mask` counts masks between 0.5%
and 10% of frame, and the mug here is 2.3-3.6%  -  so a healthy count is
*suggestive*, not proof, because the gripper and table are the same order of
magnitude. `bash run.sh track` is what actually confirms the mug is trackable,
since that is where the ground-truth mask selects which gaussians to follow.

If capture mode reports nothing, `--vocab mug cup` distinguishes the two causes:
prompts that work means a vocabulary problem, prompts that fail means the mug is
genuinely hard to detect, which is a finding about the model rather than a bug in
the data.

The detector's latency does not appear in the latency table, because DETIC only
runs inside the POGS environment. That gap is stated in each report rather than
left as a silent omission.

### Tracking

`pogs/scripts/track_main_demo.py` cannot be pointed at generated data: it
hardcodes a ZED calibration file and matrix, opens a `viser` server plus a
nerfstudio `Viewer`, and blocks on `input()` for a human crop at the clustering
step. `shared/track_offline.py` runs the same `Optimizer` headlessly so a run is
reproducible rather than dependent on when a human pressed enter.

Two decisions worth knowing:

- **No interactive crop.** HDBSCAN runs over all gaussians via POGS' own
  `model.cluster(eps)`. Which gaussians are the mug comes from projecting the
  generator's `obj_id` ground-truth mask onto frame 0 and keeping the gaussians
  inside it. Taking "the largest cluster" instead would often select the gripper,
  which is a separate instance this study is deliberately not tracking.
- **Error is displacement error**, `|(est_i - est_0) - (gt_i - gt_0)|`, per part,
  against the generator's ground truth. Differencing *distances* looks simpler but
  only measures error along the ray from ground truth to estimate, so a 50 mm
  sideways slide reads as near zero - and sideways is most of what the mug does.
  `test_track_offline.py` asserts this with a 10 mm and a 50 mm case.

`opt_iters` is fixed at 15, matching `track_main_demo.py`. A row that is faster
must not be allowed to buy its error back with extra iterations, so compare rows
at equal `opt_iters` only.

Verify the bookkeeping without a GPU - 95 checks across four files:

```bash
python shared/test_scenes.py            # 52: every scene loads, both frame
                                       #     schemas score, real_14 is unusable
python shared/test_track_offline.py     # 16: camera rebasing, error metric
python shared/test_detect_check.py      # 15: masks, IoU, channel order
python shared/test_download_weights.py  # 12: row resolution, token leakage
```

None of them can test DETIC, a trained model, or a real download, so a clean pass
means the surrounding logic is sound and the remaining risk is genuinely the
models and the CUDA install.

## Requirements

The install sequence in `row_common.sh` follows the authors' README verbatim:

- conda env `pogs_env`, Python 3.10 (per-row envs where a newer
  `transformers` or an extra package is needed)
- `torch==2.6.0`, `torchvision==0.21.0`, CUDA 11.8
- `gsplat` (cu118 wheel), `tiny-cuda-nn`, `warp-lang`
- `cuml` / `cudf-cu11==25.4.*` - **mandatory**, `pogs.py` imports
  `cuml.cluster.hdbscan` for the clustering step
- `numpy==1.26.4`, `fast_simplification==0.1.9`
- ~15 GB free for the conda env, CUDA wheels and nerfstudio outputs; plus the
  row's weights (84 MB for row 00, ~10 GB for row 01)

A preinstalled newer torch will be overridden. `stage_setup` fails loudly on a
broken import chain rather than letting training start and fail obscurely.

Optional, for the two gated rows:

```bash
export HF_TOKEN=...    # enables dino.txt (row 01) and DINOv3 (row 02)
```

## Measurement protocol

Latency numbers from a naive loop are meaningless, so `bench.py`:

- sets `cudnn.benchmark` and runs a warmup pass, so autotuning and lazy
  allocation are not charged to the measurement
- calls `cuda.synchronize()` on both sides of the timed region, since kernel
  launches are asynchronous and the clock would otherwise stop early
- reports **median and p95**, because a mean hides the tail that matters
- resets peak-memory stats per model, since peak VRAM is set by the largest
  single model rather than the sum
- measures at the resolutions POGS actually runs: DETIC 896, DINOv2 up to 1260
  (stride 14), CLIP /2 - from `dino_dataloader.py` and
  `full_images_datamanager.py`, not from guesswork
- runs fp32 and fp16 separately, because vendor tables are usually fp16 and
  mixing them makes the comparison dishonest

## Determinism and variance

POGS **seeds nothing of its own**: `dino_feats` initialisation,
`random_quat_tensor`, the datamanager's `randperm` calls and cuml HDBSCAN
clustering are all unseeded. So:

| metric | variance |
|---|---|
| latency, VRAM | **deterministic** - no training involved |
| detector IoU on real frames | **deterministic** - inference on fixed frames |
| PSNR, pose error | **stochastic** - one sample at `--machine.seed 42` |

`run.sh smoke` trains twice at the same seed and diffs the loss traces, so
determinism is *verified* rather than assumed. If the traces diverge, the report
marks tracking as noisy.

## Data provenance

- `inputs/pickplace/` - **ours**, generated by `synth_scene_gen/generate_pickplace.py`,
  verified POGS-compatible (32 checks: per-frame intrinsics, POSIX paths, real
  metric depth, GT matching the render transform to 0.0 m, no clipping, occlusion
  present but not total).
- `inputs/pendulum_30/` - **ours**, generated by `synth_scene_gen`, 162 checks.
  Kept as the deterministic regression track.
- `inputs/real_14/` - **the authors'**, from the Drive link in their README.
  Real ZED capture, 14 frames, `seg` labels. **Never trained on.** Its only job
  is to make detector accuracy a measurement that transfers to real imagery,
  since accuracy measured on our raytraced frames would not.

## Known risks

- **DETIC may still find little.** `custom_vocab` defaults to the LVIS
  vocabulary. "mug" is a plausible LVIS/COCO match, so this should be far better
  than the pendulum, but a flat-shaded synthetic mug is still not a photograph.
  This is the single largest risk to the whole experiment, and `bash run.sh detect`
  settles it in about a minute - before, not after, the long run. If it fails on
  both the synthetic and the real frames, that is a finding about the model, not a
  bug in the data.
- **`dino_txt` and `DINOv3` are `gated`** - HuggingFace licence acceptance plus
  `HF_TOKEN`. Rows 01 and 02 run their other models without a token and record
  the gated one as such, so neither row produces a full comparison unless a token
  is set.
- **VFM-DETR is `blocked`** - no weights located: zero HuggingFace search hits,
  no repository, no NeurIPS 2025 page under that name. Nearest matches are
  VFM-Det (arXiv 2408.13031, vehicle detection, 2024) and VFMTok (NeurIPS 2025,
  a tokenizer). Row 03 runs SigLIP2 only. If you have the actual paper, that is
  the one thing that would unblock it.
- **LocateAnything3D is available but not POGS-compatible.** `nvidia/LocateAnything-3B`
  is ungated, but it emits 3D boxes and POGS consumes 2D instance masks. Row 01
  reports the gap rather than bridging it, and bridging it would mean writing the
  adapter the study was meant to measure the need for.
- **LocateAnything3D is 9.5 GB.** Row 01's weight download is by far the largest
  and is only needed if you intend to measure its latency.
- **Two rows patch POGS.** Rows 01 and 02 carry a patch in `patch/`, applied at
  setup. The patches default to upstream behaviour and are verified to apply
  cleanly against a pristine tree, but they are still not stock POGS, and the
  comparison against row 00 crosses that line.
- **Occlusion is hard, and that cuts both ways.** At up to 36.1% of the
  silhouette hidden and 2.3-3.3% still visible, DETIC has little to go on. If the
  baseline scores badly here, that is a legitimate finding about the model under
  occlusion - but it also means a null result cannot distinguish "the model is
  bad" from "the scene is too hard to be informative". `pendulum_30` exists as the
  easy end of the range; a third, mid-occlusion scene would separate the two if
  the baseline comes back ambiguous.
- **The mug may span several HDBSCAN clusters.** `track_offline.py` keeps all of
  them and says so in its output rather than picking one, but a split mug means
  the rigid-body optimiser is fitting two pieces.
- **Submodule transfer.** `deps/` must arrive intact; `run.sh` checks and fails
  loudly rather than half-installing.
- **`custom_vocab` is not set.** Stock POGS leaves it empty, so DETIC uses the
  LVIS vocabulary. That is the honest baseline, but if the mug is not found it is
  worth one diagnostic run with `custom_vocab=["mug", "cup"]` before concluding
  anything about the model.