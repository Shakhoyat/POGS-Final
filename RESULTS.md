# Results

Benchmark run of 2026-11-08/09: one trained scene (4000 steps, 392k Gaussians,
train-view PSNR 31.3 dB), 54 tracking cases (12 rendered conditions x 3 random
trajectories, plus load-time speed / depth settings), RTX 4070. Protocol: only the
manipulated object(s) are tracked (operator click, Sec. IV-C). Full per-factor
tables: `data_results/{v1_base_pogs,v2_kalman,v3_objects}/summary.md`; every run
is cached in `data_results/benchmark_runs/`.

**Metrics.** *ADD-S success* = fraction of frames with ADD-S below 10% of the
object diameter (standard, symmetry-aware). *Lost* = fraction of frames with
position error > 10 cm. The paper's *3 cm & 5 deg* rule is also reported but is
nearly constant (~11%) for every method: rotation error is dominated by the
gripper-induced tilt described below, so it cannot separate methods here.
Means are over 3 seeds; ± is the seed standard deviation.

## Main findings

1. **Speed (v2).** Base POGS breaks down when the object moves faster per
   tracked frame (2x: 83% success, 4x: 40% with 56% of frames lost); the cause
   is the bounded per-frame step of a 25-step Adam run. POGS + KF + check (a
   damped constant-velocity Kalman warm start, kept only where it lowers the
   tracking loss) holds 100% at 2x and 84% with 0% lost at 4x (seeds: 60 / 93 /
   100% vs 40 / 40 / 40%).
2. **Compute (v2).** At 10 optimiser steps per frame (3.5-3.9 FPS vs 1.7 FPS)
   the Kalman warm start restores 25-step accuracy on the base condition
   (100% vs 84% for plain 10-step POGS). Over all cases the 10-step filter
   (72%) roughly matches 25-step POGS (73%) at 2x the frame rate. The advantage
   does not survive the hardest motion (4x at 10 steps: no method above 27%).
3. **Occlusion is unsolved.** A hand passing in front of the object makes every
   method lose it (43-46% success, ~54% lost) and nothing in v1/v2 helps. Even
   the gripper alone tilts the tracked mug by ~30 deg while it is held: its depth
   sits in front of the object and the depth term pulls the pose. This is the
   paper's stated limitation (Sec. VI) and the clearest target for a next method.
4. **Large rotations / unscanned faces.** Yaw 90 (58%) and tilt 60 (79%, high
   seed variance) degrade POGS; the Kalman prior does not fix them.
5. **Not discriminative here:** depth noise up to 10 mm + 5% holes (no effect:
   the 26 cm depth-ignore threshold absorbs it), the mustard bottle, and two
   objects moving at once (all 100%). These levels are at the ceiling.
6. **Cereal box fails** (32%): once the pose lags, the rendered box overlaps
   floor pixels and the depth term drags it into the background.
7. **Paper ablations:** per-frame depth is essential (No Depth: 32%; the
   release's depth-only-at-init variant: 33%). DINO vs plain RGB makes little
   difference on these flat-coloured synthetic objects (both 100%, median 1.3
   vs 2.4 cm), unlike the paper's real textured objects.

## Tables

### Per factor: ADD-S success ± seed std / lost-frame rate (25 iterations)

| Factor | Level | Static (no motion) | POGS | POGS (release: no per-frame depth) | POGS + KF | POGS + KF + check |
|---|---|---:|---:|---:|---:|---:|
| rotation | yaw 15 | 21% ±1% / 73% | 100% ±0% / 0% | 31% ±1% / 23% | 100% ±0% / 0% | 100% ±0% / 0% |
| rotation | yaw 30 | 20% ±0% / 75% | 100% ±0% / 0% | 33% ±1% / 22% | 100% ±0% / 0% | 100% ±0% / 0% |
| rotation | yaw 90 | 20% ±0% / 75% | 58% ±12% / 32% | 39% ±9% / 46% | 61% ±17% / 19% | 53% ±3% / 46% |
| rotation | tilt 30 | 20% ±0% / 74% | 84% ±28% / 16% | 39% ±7% / 34% | 83% ±29% / 16% | 83% ±30% / 17% |
| rotation | tilt 60 | 20% ±0% / 67% | 79% ±37% / 21% | 31% ±1% / 36% | 64% ±32% / 32% | 66% ±31% / 33% |
| rotation | yaw+tilt 45 | 21% ±1% / 73% | 74% ±30% / 4% | 33% ±2% / 26% | 82% ±32% / 3% | 80% ±35% / 3% |
| speed | 1x | 20% ±0% / 75% | 100% ±0% / 0% | 33% ±1% / 22% | 100% ±0% / 0% | 100% ±0% / 0% |
| speed | 2x | 20% ±0% / 73% | 83% ±29% / 16% | 52% ±8% / 23% | 100% ±0% / 0% | 100% ±0% / 0% |
| speed | 4x | 20% ±0% / 73% | 40% ±0% / 56% | 24% ±4% / 56% | 27% ±0% / 60% | 84% ±21% / 0% |
| speed | yaw 90, 1x | 20% ±0% / 75% | 58% ±12% / 32% | 39% ±9% / 46% | 61% ±17% / 19% | 53% ±3% / 46% |
| speed | yaw 90, 2x | 20% ±0% / 74% | 48% ±5% / 51% | 34% ±14% / 53% | 60% ±12% / 19% | 63% ±17% / 31% |
| speed | yaw 90, 4x | 20% ±0% / 71% | 31% ±4% / 58% | 20% ±0% / 62% | 24% ±4% / 62% | 49% ±10% / 42% |
| occlusion | none | 20% ±0% / 76% | 100% ±0% / 0% | 68% ±13% / 0% | 100% ±0% / 0% | 100% ±0% / 0% |
| occlusion | gripper | 20% ±0% / 75% | 100% ±0% / 0% | 33% ±1% / 22% | 100% ±0% / 0% | 100% ±0% / 0% |
| occlusion | gripper + hand (partial) | 20% ±0% / 76% | 43% ±0% / 55% | 38% ±5% / 54% | 44% ±1% / 43% | 44% ±2% / 54% |
| occlusion | gripper + hand (heavy) | 21% ±1% / 70% | 44% ±1% / 54% | 33% ±3% / 54% | 45% ±0% / 52% | 43% ±0% / 54% |
| depth | clean | 20% ±0% / 75% | 100% ±0% / 0% | 34% ±3% / 22% | 100% ±0% / 0% | 100% ±0% / 0% |
| depth | stereo 3 mm | 20% ±0% / 75% | 100% ±0% / 0% | 33% ±1% / 22% | 100% ±0% / 0% | 100% ±0% / 0% |
| depth | noisy 10 mm + holes | 20% ±0% / 75% | 100% ±0% / 0% | 38% ±6% / 18% | 100% ±0% / 0% | 100% ±0% / 0% |
| object | mug | 20% ±0% / 75% | 100% ±0% / 0% | 33% ±1% / 22% | 100% ±0% / 0% | 100% ±0% / 0% |
| object | mustard bottle | 21% ±1% / 73% | 100% ±0% / 0% | 67% ±5% / 0% | 100% ±0% / 0% | 100% ±0% / 0% |
| object | cereal box | 23% ±1% / 69% | 32% ±1% / 66% | 31% ±1% / 68% | 42% ±7% / 54% | 66% ±31% / 34% |
| multi-object | one moving | 20% ±0% / 75% | 100% ±0% / 0% | 33% ±1% / 22% | 100% ±0% / 0% | 100% ±0% / 0% |
| multi-object | two moving | 20% ±0% / 69% | 100% ±0% / 0% | 31% ±1% / 16% | 100% ±0% / 0% | 100% ±0% / 0% |

### Compute budget: all 54 cases

| Method | Iters/frame | FPS | ADD-S success | Lost (>10 cm) | Median pos (cm) | Success 3 cm & 5 deg |
|---|---:|---:|---:|---:|---:|---:|
| POGS | 25 | 1.7 | 73% | 24% | 11.4 | 11% |
| POGS + KF | 25 | 1.7 | 74% | 20% | 8.7 | 11% |
| POGS + KF + check | 25 | 1.6 | 80% | 17% | 7.1 | 12% |
| POGS, 10 iters | 10 | 3.9 | 61% | 35% | 12.8 | 11% |
| POGS + KF, 10 iters | 10 | 3.9 | 69% | 26% | 14.3 | 12% |
| POGS + KF + check, 10 iters | 10 | 3.5 | 72% | 23% | 11.3 | 13% |
| POGS, 5 iters | 5 | 6.6 | 43% | 48% | 14.8 | 14% |
| POGS + KF, 5 iters | 5 | 6.6 | 38% | 52% | 25.5 | 13% |
| POGS + KF + check, 5 iters | 5 | 5.5 | 57% | 36% | 11.3 | 14% |

### Speed factor at a reduced budget (ADD-S success / lost)

| Level | POGS | POGS, 10 iters | POGS + KF, 10 iters | POGS + KF + check, 10 iters | POGS, 5 iters | POGS + KF + check, 5 iters |
|---|---:|---:|---:|---:|---:|---:|
| 1x | 100% ±0% / 0% | 84% ±28% / 15% | 100% ±0% / 0% | 100% ±0% / 0% | 49% ±6% / 44% | 81% ±34% / 17% |
| 2x | 83% ±29% / 16% | 42% ±5% / 49% | 36% ±8% / 53% | 69% ±28% / 27% | 23% ±0% / 54% | 34% ±7% / 47% |
| 4x | 40% ±0% / 56% | 27% ±0% / 58% | 27% ±0% / 60% | 27% ±0% / 56% | 27% ±0% / 60% | 27% ±0% / 60% |
| yaw 90, 1x | 58% ±12% / 32% | 51% ±6% / 46% | 55% ±12% / 34% | 64% ±5% / 12% | 38% ±11% / 49% | 48% ±11% / 44% |
| yaw 90, 2x | 48% ±5% / 51% | 36% ±10% / 52% | 28% ±2% / 62% | 44% ±5% / 49% | 23% ±0% / 59% | 29% ±8% / 50% |
| yaw 90, 4x | 31% ±4% / 58% | 27% ±0% / 60% | 27% ±0% / 62% | 27% ±0% / 58% | 24% ±4% / 62% | 24% ±4% / 62% |

### Paper ablations (base condition, 3 seeds)

| Method | ADD-S success | Lost | Median pos (cm) | Rot err (deg) |
|---|---:|---:|---:|---:|
| Static (no motion) | 20% | 75% | 20.85 | 15.0 |
| POGS | 100% | 0% | 1.28 | 33.2 |
| No Depth | 32% | 22% | 6.62 | 38.1 |
| No DINO (RGB) | 100% | 0% | 2.41 | 33.6 |


## History of this run

- The first v2 filter (undamped) diverged by metres once tracking slipped: a
  warm-started optimiser's output is not independent of the prediction. Those
  runs were discarded; v2 now uses velocity damping (tau 1 s), speed limits
  (0.5 m/s, 3 rad/s) and the optional loss check. The "fuse" variant was dropped.
- Tracking all five objects instead of the manipulated one drags static
  objects through their ROIs (12 cm drift in a smoke test).

## Paper reference numbers (physical robot, not comparable to simulation)

Table I (object reset, POGS, Tier 1): mean placement error 2.5 / 3.4 / 4.1 cm
(jigsaw / iron / shoe), measured with calipers on real placements. Table II (tool
servoing): recovery from 15 deg perturbations 48/50, from 30 deg 40/50, success =
tool tip within 3 cm and 5 deg. Our benchmark reports tracking error against exact
simulator ground truth, with the same 3 cm / 5 deg criterion as a per-frame success rate.

