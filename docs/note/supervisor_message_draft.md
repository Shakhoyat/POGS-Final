# Message to the supervisor (draft; the user sends it)

Subject: POGS reproduction: a mechanism finding, two questions

Short version: under the paper's own roll-free rule, our re-implementation matches the 30° tier in one
direction (+30°: 39/50 vs 40/50) and falls short elsewhere. The gap comes from a systematic ~10° roll bias about
the drill bit. We traced it to its cause: the DINO loss gets much of its pose restoring force from an implicit
silhouette term, the live background compared with a zero render. Shadows and viewpoint bias that term. The
evidence:
- ground truth exact;
- the drill's own features are unbiased;
- removing the background collapses recovery (1/0/4/4 vs 9/12/16/23 of 25);
- depth alone recovers 0.
Six simple fixes each fail for a different reason.

1. Is a mechanism/analysis paper on feature-field tracking an acceptable outcome? If so, which venue do you
   consider realistic (e.g. RA-L, or an ICRA/IROS workshop)?
2. May the tracker use a known gripper model (commanded pose + mesh, i.e. robot kinematics) to exclude gripper
   pixels? POGS lists gripper masking and end-effector regularisation as future work (Sec. VI). We would apply
   the same exclusion to the POGS baseline for fairness. This gates one pre-registered test of an explicit,
   shadow-free silhouette term (GPU cap 6 h).
