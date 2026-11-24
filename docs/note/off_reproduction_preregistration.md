# 8.OFF pre-registration: does the OFFICIAL POGS code show the same roll bias?

Written 2026-11-25 ≈ 00:40 Dhaka, before the official environment exists and before any official-code number is
computed. Extension priority (1) of `plan_author_reply.md`. Committed before any official-code run.

## Why
Every result so far (E2, B1, G1, G2) uses our re-implementation. The most likely reviewer objection is "the bias
is an artefact of the re-implementation". The public release contains no POGS captures or splat checkpoints (ledger
8.OFF: `test_data` and "trained models" are Contact-GraspNet material), so the check must use the official code on
our simulated capture.

## Code and inputs
- Official repository `uynitsuj/POGS` @ `3d29f04` (2025-05-16), nerfstudio submodule `5003d0e` (v1.1.5-26),
  unmodified, pinned at `~/third_party_pinned/POGS-official`. Our `third_party/POGS-author-github/pogs` is
  byte-identical to it.
- Data: the drill scene `tool_table` capture (35 views, RGB + metric depth, exact poses), exported to the
  nerfstudio format (`transforms.json`, images, float depth in metres, seed point cloud as `points3D` ply).
- Training: official `pogs` method config (4000 steps, official optimisers, DINO/CLIP/grouping schedule, camera
  optimiser as configured), official DINO extractor and CLIP encoder.
- Tracking loss: official `RigidGroupOptimizer.get_optim_loss` with the official config (ROI on, blur kernel,
  `use_dino=True`; depth on/off reported as two variants, as in G1).

## Deviations (environment, not method), stated in the paper if this result is used
1. torch 2.14.1+cu130, CUDA 13.0 nvcc, Python 3.12 instead of torch 2.6 / CUDA 11.8 / Python 3.10: system gcc 15
   cannot host CUDA 11.8's nvcc and there is no conda. gsplat at the official pin (1.4.0) if it JIT-builds,
   otherwise 1.5.3 (stated).
2. Detic is not run (detectron2 does not build on this stack). Its cache (`outputs/<data>/detic.npy`) is filled
   with the simulator's instance masks put through Detic's own post-processing (3×3 erosion, masks larger than
   H·W/3.5 dropped, background mask last). Our re-implementation uses the same masks, so the comparison isolates
   the training and tracking code.
3. cuML is not installed; `cuml.cluster.hdbscan.HDBSCAN` and `cuml.neighbors.NearestNeighbors` are provided by the
   `hdbscan` package and scikit-learn with the official call arguments.
4. The operator's click (interactive viewer) is replaced by a scripted click at the centroid of the drill group
   found by our geometric selection; the official "crop to click" code then selects the cluster. The object frame
   is the official default (group centroid, identity rotation).
5. If tiny-cuda-nn does not build with CUDA 13 + gcc 15: a pure-torch stand-in for `tcnn.Network` /
   `tcnn.Encoding(HashGrid)` (nerfstudio's torch `HashEncoding` + `MLP`). It affects only the CLIP and instance
   (grouping) heads, not the DINO features or geometry used by the tracker. Stated if used.

## Frames (same as G1; rendered by our simulator)
- Tracking camera with the frame-0 gripper, cast shadows on (s = 1) and off; the six shadow strengths
  s ∈ {0, 0.25, 0.5, 0.75, 1, 1.25} of the G1 dose-response.
- The 12 held-out ring views (r 0.87 m, elevation 30°), shadows on, no gripper.
- Roll = rotation about the drill object x axis (bit axis) through the group centroid, grid ±20° at 1° (dose: 0.5°),
  applied through the official `part_deltas` (wxyz quaternion) so the official render path is used.
- The official observation path: `PosedObservation` + `add_roi` from the official `calculate_roi` at the true pose,
  official DINO feature function.

## Predictions and pass rules (form copied from G1 P2, P4, P5)
- **OR1 (P2 form).** Official loss (DINO only, release default `use_depth=False`), tracking camera with gripper and
  shadow: drill roll argmin ≤ −5°.
- **OR2 (P4 form).** Over the six shadow strengths: Spearman(s, argmin) ≤ −0.8 and argmin(s=1) − argmin(s=0) ≤ −4°.
- **OR3 (agreement, descriptive with one rule).** Over the 13 views (tracking + 12 ring), the signed argmin of the
  official pipeline and of our re-implementation (G1 seed 0, same views) have Spearman ≥ 0.5 and median absolute
  difference ≤ 3°. Two different splats are compared, so this is an agreement check, not an identity test.
- Exploratory: the same with `use_depth=True`; the official mask-free rendered-background variant is not part of
  the official code and is not run here.

## Reading
- OR1 and OR2 pass → the bias and its shadow dependence are properties of POGS, not of our re-implementation; the
  paper says so and lists the deviations above.
- OR1 or OR2 fail → reported as a fail in the paper (generality section and limitations) with the official numbers;
  the claim is narrowed to "our re-implementation" unless a pre-registered follow-up explains the difference. No
  threshold is changed after the fact.
- An environment failure (training does not converge: train-view PSNR < 30 dB, or the drill is not separated into
  its own cluster) is reported as "not testable", with the reason, not as a fail.

## Budget
One GPU process; ≤ 3 GPU hours in total (training ≈ 30–60 min, sweeps ≈ 30 min); never during lab hours
(`tools/supervisor.sh labhours`). Outputs: `data_results/off_repro/` (summary committed, bulk local);
official training outputs under `~/pogs_external/off/` (local, not in git).
