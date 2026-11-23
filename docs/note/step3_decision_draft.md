# Step 3 decision (draft for the user, 2026-11-23)

## What we know now
- Inside the drill, distilled DINO features are unbiased at the true pose in all 16 views tested and with or
  without shadows (B1e). The view-dependent ±5° and the shadow's −4° appear only when background pixels enter
  the loss.
- The background pixels are also what makes tracking work. Remove them (masks) or replace them with a
  rendered background, and the tracker loses the drill as soon as it moves (E2d).

## Would step 3 as planned (3-D-aware feature fine-tuning, FiT3D-style) help?
**Probably not with this bias.** 3-D-aware fine-tuning makes the features on the object more consistent across
views. Our measurements say the object features are already unbiased. The bias comes from how background
pixels, compared with a zero render, act as a silhouette term. Better object features do not change that term,
whose behaviour is set by the live background (table texture, shadows) next to the object outline. Expected
gain: small, at the cost of many GPU hours and a new training pipeline. I recommend **not** running it as
planned.

## Options that target the mechanism (all ground-truth-free)
| Option | Idea | Main risk | Cost |
|---|---|---|---|
| (a) explicit silhouette term against a **depth-foreground mask** | live foreground = pixels where live depth is closer than the rendered static background by > τ; IoU / BCE against the rendered object mask. Depth does not see shadows | the gripper is foreground too and touches the drill; needs a gripper exclusion (known robot model or a second rendered group) or a robust loss | 1 day code + 1 O4 run (~1 h at 2 shards) |
| (b) DINO background term with **shading-normalised features** (e.g. per-pixel feature normalisation, or a feature-space projection that removes the brightness direction) | keep the silhouette signal, remove the shadow's effect on background feature norms | the view-dependent ±5° part is not shading and may remain | ½ day + sweeps + 1 run |
| (c) **robust background term** (truncated or Huber on ‖live‖ outside the outline) | caps the influence of the high-norm shadow pixels | may weaken the restoring force as well (B1f) | ½ day + 1 run |
| (d) stop method work; write the note | the note already has a clean story: rule mismatch, Static-checked metric, the bias and its mechanism, why four substitutes fail | no method contribution | — |

## Recommendation
(d) now, with (a) as the one bounded method test if you want one. It is the only option that keeps a
pose-sensitive silhouette signal while removing the illumination channel, which is the cause we identified.
Pre-register it with the 8.E2c rule (beat POGS and the depth-only control) and give it one run. If it fails,
the note stands as written.

Questions for you / the supervisor:
1. Is an analysis paper (RA-L or a workshop) an acceptable outcome, or is a method gain required?
2. May tracking use a known gripper model (robot kinematics) to exclude its pixels? That is ground-truth-free
   on a real robot, but it adds an input POGS does not use.

## Decision (user, 2026-11-23)
Step 3 skipped. Write the note now. Run the depth-silhouette test (option a) **only if the supervisor says
yes to question 2**, pre-registered in `docs/note/dsil_preregistration.md` (GPU cap 6 h, pass rule 8.E2c vs a
reference POGS with the same gripper exclusion). Both questions go to the supervisor
(`docs/note/supervisor_message_draft.md`).
