# Overnight 2026-11-24/25: what changed (for the user and the supervisor)

All tests below were pre-registered (committed before they ran) unless marked exploratory. Full rows: `docs/LEDGER.md`
(8.E4T … 8.OR6) and the Findings inbox. Paper: `docs/paper_rss/main.pdf` (RSS 2027 Stage 1 format, 5 pages).

## 1. The most important result: the official POGS code
I installed and ran the official POGS release on our simulated drill capture (Python 3.10 as officially specified;
torch 2.14/CUDA 13; real cuML; Detic replaced by the simulator masks; tiny-cuda-nn replaced by a pure-torch stand-in for
the grouping heads only; one release bug worked around: eval passes off).

| Test | Result |
|---|---|
| OR1: official loss has the −10° roll minimum with the gripper's shadow | **FAIL** (+2°) |
| OR2: official loss follows the shadow dose-response | **FAIL** (ρ 0.00) |
| OR3: official and our argmins agree across 13 views without the gripper | pass (ρ 0.53, median 2°) |
| OR4: official tracker has the roll error before contact | **pass** (median 19.4°; Static 0°) |
| OR5: an ROI-only initialisation removes it | FAIL (−7.8° after init, −19.7° at end of approach) |
| OR6: official tracker on 40 in-grasp trials, both rules | 0/40 both rules (= Static); ours release 10/40, ours paper 24/40 roll-free. OR6a FAIL (differs from ours by 10), OR6b pass |

Exploratory, decisive for interpretation:
- **Our tracker reproduces the official trajectory**: over the approach the two trackers' roll differs by a median of
  0.7° (ρ 0.87). (Caveat: all O4 seeds share one approach trajectory.)
- **Swap**: our tracking code on the official splat gives −1.5° (official code +1.5°); on our splat −10.5°. The
  objectives agree; the splats differ. Our drill groups over-cover the true silhouette (730–1100 px vs 330).
- **The official objective mostly prefers the truth** (40 of 60 approach frames), yet the tracker drifts: the roll
  landscape is nearly flat and asymmetric, and as the gripper descends its minimum moves to large negative roll for
  both splats (`docs/paper_rss/figs/roll_landscape.pdf`). The estimate slides at Adam's step limit (4–5°/frame).

## 2. Other pre-registered results tonight
- E4 (boundary-gradient prediction of the bias): E4a pass (ρ 0.80), E4b pass (sign 48/48), E4c FAIL (area-only
  predictor as good).
- G1c (dose-response on 4 objects without gripper): FAIL — scene self-shadows do not move the minimum.
- E5 (masked tracker's drift at the true pose along the lift): FAIL — it is dynamic.
- E1a (pruning over-covering Gaussians): PR1/PR3 FAIL, PR2 pass — pruning removed only ~30% of the over-coverage.
- F1 (origin of the floor): F1a pass (whole-image init minimum −7°), F1b FAIL (ROI minima on a ±20° grid).

## 3. What this means for the paper (decision needed)
Robust and reproduced: **the roll-free recovery rule hides a large roll error that both the official tracker and ours
carry before contact**, plus the null-model and relative-rule checks. The mechanism common to both pipelines is a
nearly flat, asymmetric roll landscape that an occluding gripper tilts. The static −10° shadow-driven offset (current
title) is a property of our splats and is NOT reproduced by the official splat.

The current title ("Shadows Bias Feature-Field Object Trackers Through an Implicit Silhouette Term") now overstates the
shadow claim. Options:
1. "A Roll-Free Recovery Rule Hides Tool-Axis Drift in Feature-Field Object Tracking" (evaluation-led; safest).
2. "Flat Roll Landscapes Make Feature-Field Trackers Drift Under Gripper Occlusion" (mechanism-led).
3. Keep the title and narrow the abstract (not recommended: the official splat does not show the shadow bias).

## 4. Actions for you
- **OpenReview account now** (RSS: profiles without an institutional email can take up to two weeks to moderate).
- **Push**: commits since ≈ 03:00 are local only; GitHub pushes go through VS Code's credential prompt, which did not
  answer while you were away. Opening VS Code (or `git push origin research-plan-rev2`) pushes them; the supervisor
  loop retries every 30 min.
- Decide the title/framing; the RSS draft is otherwise ready to iterate.
- Email to Justin: the correction draft is in `docs/note/plan_author_reply.md`; tonight's official-code result is
  worth mentioning privately if he replies (Branch D).
