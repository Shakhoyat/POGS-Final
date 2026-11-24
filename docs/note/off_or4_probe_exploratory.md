# 8.OFF-P (exploratory, no pass rule): does the official objective prefer the drifted pose?

Written 2026-11-25 ≈ 02:58 Dhaka after OR4 (pass: official tracker |roll| 19.4° at the end of the approach, growing
from −7.4° right after initialisation) and before this probe runs. Same as E2 for our tracker: at every approach and
lift frame of the OR4 run, evaluate the official loss (DINO, ROI, no per-frame depth, as `step`) at the tracker's
estimate and at the ground-truth pose. loss(est) < loss(GT) on most frames → the objective itself prefers the
drifted pose (a bias, as in our implementation); loss(est) > loss(GT) → the tracker fails to descend (optimisation /
basin / iteration budget).
