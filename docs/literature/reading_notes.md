# Reading notes: reference papers in `ref-paper/` (read in full text, 2026-11-22)

Each entry: what was read, the facts we rely on (with section/table), and what it changes for us.
Tags: [V] read in the PDF text this pass. Numbers are the authors' own, on their own protocols.

## Yu et al. 2025, POGS (ICRA 2025), arXiv 2503.05189 -> `2025_ICRA_Yu_POGS.pdf`
- [V] Tracking = per-object SE(3) by gradient descent on DINO-feature loss + depth loss (Sec. IV-D); no uncertainty,
  no symmetry handling, no failure detection. Stereo depth from TRI model at ~30 Hz (Sec. IV-C).
- [V] Assumptions (Sec. III): rigid; start within 90 deg / 25 cm of scan pose; all surfaces seen in scan except table contact;
  uniform light; no full mutual occlusion; low specularity.
- [V] Tool servoing (Sec. V-B, Table II): drill + ArUco target; 5 trials/tier x 5 target poses x {CW, CCW};
  Tier 1 CW 24/25, CCW 24/25 (48/50, 6.01 s mean); Tier 2 CW 20/25, CCW 20/25 (40/50, 12.66 s); follow-target 24/25, 21/25.
  Tier 1 = target on table plane, in-grasp change "up to 15 deg"; Tier 2 = target moves in 3D (the text ties 40/50 to 30 deg).
  Success = drill tip within 3 cm and 5 deg of the target, closed-loop, no frame budget, no baselines.
- [V] **"We leave tool rotation about its target axis (i.e. drill bit axis) unconstrained"** (Sec. V-B). So the paper's 5 deg
  rule does NOT score roll about the bit axis. Our pre-registered O4 rule uses full geodesic rotation, which includes roll.
  This alone can explain a large part of our 10-24 deg "floor" vs. their 5 deg success. -> E0 (re-score with roll excluded).
- [V] Perturbation is "about the grasp axis" (Sec. V-B), which is not necessarily the bit axis.
- [V] Limitations (Sec. VI): 5 Hz on an RTX 4090 (~140 ms DINO ViT-S); objects moved slowly; partial occlusion by hand/gripper
  hurts; authors propose an end-effector regularisation term and gripper masking as future work (=> both are "obvious next
  steps", low novelty, and a scoop risk).
- [V] Object reset (Table I): ablations No-Depth / No-DINO (RGB) collapse; mean placement error 1.8-4.1 cm.

## Tuna et al. 2024, X-ICP (IEEE T-RO 40), arXiv 2211.16335 -> `2024_TRO_Tuna_X-ICP_localizability-aware-registration.pdf`
- [V] Point-to-plane ICP; Hessian A' = sum (p x n, n)(p x n, n)^T. Eigen-analysis done **separately on the rotation and
  translation 3x3 blocks** because their scales differ (Sec. V-A). Eigenvalues are *not* used directly: they vary with
  sensor/point density, so fixed eigenvalue thresholds need per-environment tuning (Sec. II-B, VII-E; Zhang et al. needed
  threshold 120 vs 200 by site, Fig. 17).
- [V] Localizability = per-correspondence contribution |F V| (moment-normalised Jacobian rows projected on eigenvectors),
  filtered at cos 80 deg, strong at cos 45 deg; three categories full/partial/none via counts kappa1=250, kappa2=180, kappa3=35.
- [V] Mitigation: equality constraints v^T (x - x0) = 0 through Lagrange multipliers; none -> keep the prior (odometry) along
  that direction; partial -> constrained update from resampled high-contribution pairs; full -> unconstrained.
- [V] Baselines it beats: Zhang et al. 2016 "degeneracy factor + solution remapping" (min eigenvalue threshold, project the
  update onto well-constrained eigenvectors) and Hinduja et al. 2019 (relative condition number, partial factors).
  Real data: underground mine, construction site, park; APE 2.05 m vs 3.36 / 5.79 m (Table I).
- [V] Admitted limits: sensitive to the initial guess; categorisation is discrete; future work = fold into a factor graph.
- **For us.** (1) The port is not "use the Hessian" (old: Zhang 2016) but a *scale-free* observability measure for a
  photometric/feature render loss, where there are no point-normal pairs: use per-pixel Jacobian rows of the DINO+depth
  residual as "information pairs" and normalise rotation by object radius. (2) Mandatory baselines for any
  observability-aware update: solution remapping (Zhang 2016) and a fixed-threshold variant. (3) X-ICP keeps the *prior*
  along degenerate directions; for an in-hand tool the natural prior is gripper kinematics / constant-velocity, which is
  what the filter supplies.

## Zhang, Fischer, Larsson, Ilg 2026, Compact pose graph (arXiv 2610.11815, Oct 2026) -> `2026_arXiv_Zhang_compact-pose-graph-tracking.pdf`
- [V] Correspondences (RoMa v2 / LoFTR / UNOPose) -> RANSAC + weighted Umeyama relative pose -> per-edge covariance
  Sigma = sigma^2 H^-1, H = sum J^T J with right-multiplied SO(3) perturbation, sigma^2 = weighted residual / (3N-6)
  (Eq. 8, Supp. Sec. 9). Pose graph over relative-pose edges only, iSAM2, Cauchy loss; K=5 loop closures by
  rotation-radius candidate set + farthest-frame sampling.
- [V] Table 1 (H100, ADD/ADD-S AUC 0-0.1 m): YCBInEOAT ours 94.4 / 88.8 (ADD-S / ADD), FoundationPose (CAD) 96.4 / 93.1,
  BundleSDF 93.8 / 87.0, BundleTrack(R) 93.8 / 88.0; HO3D ours 96.4 / 92.2; YCB-Video 98.2 / 96.5; HOI4D 92.2 / 80.3
  (FoundationPose 84.4 / 66.5). Speed 10 fps (H100), 9 fps (4090). FoundationPose tracking 22.9 fps average on H100
  (10-33 fps by dataset, Table 9) -- not 32 Hz everywhere.
- [V] **Ablation ladder (Table 6)**: Identity / Trace / full Covariance / Oracle. Trace and covariance both help; full
  anisotropic covariance gives *no consistent benefit over trace on two of four benchmarks*; "an anisotropic covariance only
  helps when its estimated directions are correct". Gains larger with a weaker matcher (LoFTR: up to +4.3 ADD).
- [V] Spearman(uncertainty trace, true error) rotation 0.717 (HO3D), 0.651 (YCBInEOAT); translation 0.713 / 0.599 (Fig. 6).
- [V] Failure (Fig. 5): hand covers label of a rotationally symmetric object -> rot error 37.66 deg with 1.22 deg predicted
  (~31x overconfident); thin object end-on -> 176.96 deg vs 36.92. Limitation (Sec. 6): "local approximation ... can capture a
  flat, poorly-constrained direction but not a well-separated alternative solution"; future work = learned multi-modal pose
  distributions.
- **For us.** This is the closest prior art for "Hessian covariance in object tracking" and it already ran the
  Identity/Trace/Covariance/Oracle ladder. Our differences must be: (a) the information comes from a dense
  foundation-feature render loss of a persistent splat (no correspondences, no RANSAC); (b) we target exactly its documented
  failure -- separated alternative modes on (near-)symmetric objects -- with a weak-axis multimodal belief; (c) calibration is
  evaluated with proper scores (NLL, coverage, NEES), not only Spearman. Copy its protocol (first-frame alignment, AUC
  0-0.1 m, same masks for all methods, per-dataset fps) so numbers are comparable.

## Liang et al. 2026, RoPT (arXiv 2610.28962, Oct 2026; NYU + GM) -> `2026_arXiv_Liang_RoPT-robust-pose-tracking.pdf`
- [V] FoundationPose refiner+scorer (V0) + EdgeTAM mask -> centroid translation prior (V1) + GTSAM iSAM2 constant-velocity
  factor graph, Huber observation factor, 4 hypotheses H0-H3 (prev / predicted / predicted-R + mask-t / prev-R + mask-t)
  (V2) + SH lighting fit on frame 0 (V3) + failure-triggered MASt3R render-to-image matching + PnP-RANSAC (V4). Needs CAD.
- [V] **Fixed observation sigmas {0.02 x3, 0.10 x3}, motion noise 0.05** (App. A). Failure trigger: FoundationPose score
  > 140 (typical ~100) or jump > 50 mm / 15 deg -- hand-tuned thresholds.
- [V] YCBInEOAT with frame-rate downsampling 1/2/4/8/16x (Table 1, 3): RoPT ADD 92.02 / 92.05 / 91.58 / 91.64 / 90.85;
  FoundationPose 91.97 / 91.94 / 65.81 / 56.74 / 37.97 (Table 3 V0) -- Table 1 reports FPose 38.87 at 16x (two tables
  differ slightly). FPose++ 67.37 at 16x. Mean per-frame motion at 16x: 23.2 mm and 5.8 deg.
- [V] Ablation: mask frontend alone (V1) hurts ADD at 1x (69.9 vs 92.0) -- a mask-centroid prior is harmful unless gated.
  Lighting (V3) ~ +0.1. Runtime (Table 2, RTX 6000 Ada): FoundationPose 23.3 ms (42.9 fps), RoPT 53.7 ms (18.6 fps).
- [V] Own real benchmark: FANUC CRX, Astra 2 at 25 fps, robot TCP as ground truth, peak speeds 10/15/20 cm/s
  (App. C.1, Table 5): RoPT ADD 60.1 vs FoundationPose 54.0. => robot-kinematics ground truth is an accepted protocol.
- **For us.** (1) The frame-downsampling protocol (1-16x) is the accepted way to test "fast motion" on YCBInEOAT; adopt it
  (E8). (2) Fixed observation noise -> a per-frame, direction-aware measurement covariance from the render-loss information
  matrix is a concrete, testable delta over RoPT. (3) Its failure trigger is a hand-tuned score; a calibrated NIS/chi-square
  gate is the principled alternative. (4) Recovery via render-to-image matching works because the renderer gives 3D
  coordinates for every pixel -- a splat gives the same for free (re-detection baseline for us).

## Opra, Ghafari, Stewart, Stachniss 2026 (arXiv 2607.23468, Jul 2026; Woven by Toyota + Bonn; IEEE (c) notice) -> `2026_arXiv_Opra_robust-tracking-built-in-recovery.pdf`
- [V] Built on ICG+ (region + depth + texture, Gauss-Newton with **Tikhonov regularisation per rotation/translation block**,
  Eq. 3: "helps stabilise ... underconstrained" DoF). Adds SuperPoint-TensorRT (1800 kp), a 3D point-distance residual
  (Tukey, 1/d^2 weights), keyframe queue + icosahedron keyframe store, **cycle-consistency reliability check** (>= 10 inliers
  with model-frame distance < eps_cc), and global recovery (16 crops x 20 keyframes -> TEASER++ -> best inliers -> refine ->
  re-check). Recovery ~210 ms vs 2.69 s for CNOS + FoundationPose re-init.
- [V] YCBInEOAT (Table II): ours ADD 93.14 / ADD-S 96.57 at 57.6 fps (RTX 4090), FoundationPose 93.09 / 96.42 at 41.3 fps,
  ICG+ 90.79 / 96.35 at 111 fps. Tomato soup can (axially symmetric): ICG+ ADD 79.83 vs ours 90.69 vs FP 91.71.
- [V] Own dataset (6 seqs, 2491 frames, stereo depth, hand-annotated): full system ADD 83.67 vs 50.90 without recovery vs
  FoundationPose 45.58 (Table I); weakest object = textureless rotationally symmetric mug (66.09 ADD). Output frozen 6 frames
  after recovery to account for latency (honest real-time accounting).
- [V] Limits: recovery only from views already seen; identity switch with duplicate instances.
- **For us.** (1) Reliability signal = match inliers, not a pose covariance -> calibrated uncertainty is a different,
  complementary claim. (2) Tikhonov damping per block is the "isotropic damping" baseline our anisotropic damping must beat.
  (3) Symmetric objects (mug, soup can) remain the weak spot of every 2026 tracker -> our narrow claim lives there.
  (4) Adopt their "freeze output for the recovery latency" rule in the real-time protocol.

## Li et al. 2026, RRTrack (arXiv 2607.23669, Jul 2026; ZJU + TeleAI) -> `2026_arXiv_Li_RRTrack-recoverable-pose-tracking.pdf`
- [V] CUTIE VOS mask + frozen FoundationPose refiner; rendered-mask agreement P_t = |M n R|/|M|, S_t = |M n R|/|R| and VOS
  entropy gate memory writes; "snap" translation from mask centre + median depth; lost / stagnation tests; DINOv2 ViT-S/14
  CLS retrieval from 256 offline rendered templates + 64 online anchors, top-3 each, refine + rank, accept if P* > tau.
  All thresholds from an EMA-MAD adaptive manager (alpha 0.05, 100-frame buffer).
- [V] **Synthetic-only quantitative evaluation** (Isaac Sim: Franka-YCB, agile quadcopter, Dingo AGV; 22 videos, 38 episodes,
  72 lost-reappear events); real-world drone results are qualitative only (no ground truth). RTX 4090, 55.2 fps.
- [V] **Event metric** (Sec. IV-A-b): per lost-reappear event, recovered if any of the 10 frames after reappearance has
  ADD-S AR > 0.9 (0.1 d); post-recovery AUC/AR over the 10 frames after the first success; unrecovered = 0. Also median delay.
  Full model success 0.722, w/o dual-bank 0.681, w/o adaptive threshold 0.639 with delay 9 vs 2 frames (Table IV).
- [V] Equal-subset mean ADD-S AR 73.5 vs FoundationPose 44.2 (Table II). Future work: "symmetry-aware and
  probability-distribution-based pose estimation ... for symmetric objects".
- **For us.** (1) A synthetic-only benchmark + qualitative real demo was accepted framing for this paper (venue unknown);
  it does not lift the bar for ICRA/CoRL/CVPR, where we still need real quantitative data. (2) Adopt the event-level recovery
  metric verbatim for comparability, then add what it lacks: a null model (Static / frozen estimate scored with the same rule),
  survival curves for time-to-recover, and seed/scene-level intervals. (3) Their stated future work is our core claim -- a
  scoop signal; move fast and cite.

## Kong, Lee, Lee, Kim 2026, UGS-Loc (arXiv 2603.16538, Mar 2026; Yonsei) -> `2026_arXiv_Kong_UGS-Loc-3DGS-pose-refinement-uncertainty.pdf`
- [V] Task: camera relocalisation against a scene splat (Scaffold-GS), refining an APR/SCR/retrieval prior. GS-CPR-style
  pipeline (render -> MASt3R 2D-2D matches -> lift with rendered depth -> PnP-RANSAC) plus (a) Monte Carlo refinement:
  m = 8 particles, N = 2 iterations, each particle locally optimised then weighted by sum(match confidence x (1 - U));
  final SSIM re-weighting; (b) **Fisher information over the splat's *scene parameters*** (FisherRF-style, diagonal Laplace
  approximation, aggregated over training views) rendered into a per-pixel depth-uncertainty map U that biases RANSAC sampling.
- [V] Note: the Fisher information is about **geometry/Gaussian parameters, not about the pose**. They never compute a pose
  information matrix or pose covariance. => "Fisher information of the render loss w.r.t. object pose in a splat tracker"
  is not claimed by this paper (still cite it and FisherRF).
- [V] Results: Cambridge average ACE+GS-CPR 15 cm / 0.33 deg -> ACE+UGS-Loc 10.5 / 0.21 (~30% lower translation);
  7-Scenes 2 cm/2 deg recall 95.6 vs 93.1. Ablation (Tables E/F): Monte Carlo refinement gives most of the gain; uncertainty
  PnP alone ~11%. 1.1 s per iteration (RTX A5000) -- not real time.
- **For us.** (1) A per-Gaussian geometric-reliability weight (from Fisher over splat parameters, computable once after
  training) could down-weight unreliable splat regions in the tracking residual; cheap ablation, possible component.
  (2) "Multi-start / particles + local optimisation" is accepted in splat refinement -> our equal-compute multi-start
  baseline is mandatory and must be beaten on the same compute.

## Lu, Sijan, Noh, Chen, Popov 2026, EnDKF (arXiv 2605.03105, May 2026; U. Hawaii) -> `2026_arXiv_Lu_EnDKF-ensemble-directional-Kalman-filter.pdf`
- [V] Ensemble Kalman filter generalised to manifolds via exp/log/projection maps; gain chosen by stochastic optimisation
  (Adam) of the ensemble generalised variance instead of ensemble covariances; quaternion attitude with
  Fisher-von Mises-Langevin noise; state (v, p, angular velocity, attitude) in R3xR3xS3xS3; optional constant bias states.
- [V] Evaluation: synthetic constant-velocity twin experiment (100 MC runs) and a Blender digital-twin head rotating at
  constant yaw, FoundationPose measurements at 10 fps. Ideal model: attitude error -0.78% vs measurements; perturbed model
  with bias states: attitude -16.8%, position -7.0% (Tables I-II). Attitude error is *under*-predicted by the filter
  (Fig. 1: "not conservative"). No real data (stated future work); no comparison to an MEKF/IEKF.
- **For us.** Confirms (i) filters on top of a pose detector give small gains on attitude unless they model a bias;
  (ii) attitude covariance from a generic filter is overconfident. A measurement-specific (per-frame, per-direction) R is
  the missing ingredient. Not a competitor at a top venue; use as related work for directional filtering only.

## Andrews & Morgansen 2026, DQ-UKF observability (arXiv 2605.02054, May 2026; U. Washington) -> `2026_arXiv_Andrews_dual-quaternion-observability-visual-pose.pdf`
- [V] Lie-algebraic nonlinear observability (rank of the observability codistribution dO built from Lie derivatives of
  h along the process model) for 6-DoF relative pose + velocity with (a) 3D marker positions, (b) unit-vector (pinhole)
  measurements. Theorems 2-3: locally (weakly) observable iff three markers are non-collinear -- recovers the P3P collinear
  degeneracy. Remarks: P3P's up-to-four global solutions = global multiplicity while each is locally observable; "the prior
  estimate plays the disambiguating role of a fourth point" (Sec. V); "nearly unobservable" (near-collinear, high noise)
  is the practical problem.
- [V] Dual-quaternion Lie-group UKF (left-invariant error, sigma points retracted on DH_u). Simulation only (5 markers,
  occlusion schedule 4-3-2-3-4 markers at 30 Hz): with 2 markers the covariance grows only in some directions -- partial
  observability of a subspace (Fig. 4). Baseline = OpenCV PnP.
- **For us.** Template for the theory section: (1) local vs global ambiguity is exactly our split -- the information matrix
  H handles *local* weak directions; separated symmetric modes are a *global multiplicity* that needs a multimodal belief.
  (2) State it formally: for an object with a continuous symmetry axis a, the render loss is invariant along exp(t a^) so
  a is in null(H) for any image -> unobservable for any estimator; for a near-symmetric tool lambda_min(H) is set by the
  asymmetric features' image footprint. (3) Use rank/eigen-spectrum of H (instantaneous) and of the filter information
  (with process model) -- the latter is what a sequential estimator needs.

## Wen, Yang, Kautz, Birchfield 2024, FoundationPose (CVPR 2024), arXiv 2312.08344 -> `2024_CVPR_Wen_FoundationPose.pdf`
- [V] Unified estimation + tracking for novel objects, model-based (CAD) or model-free (~16 posed reference images ->
  neural SDF object field -> marching cubes mesh for fast rasterised RGB-D rendering). Trained only on ~1.2M synthetic
  images (Objaverse + GSO, LLM-prompted TexFusion textures, Isaac Sim path tracing), ~1 week on 4 V100.
- [V] Estimation: 42 x 12 = 504 global hypotheses -> transformer refiner (5 iterations) -> hierarchical pose ranker
  (self-attention across hypotheses), 1.3 s per object (RTX 3090). **Tracking = refiner only, 1 iteration, from the previous
  pose, ~32 Hz on RTX 3090**; no re-initialisation in the tracking benchmark ("to evaluate long-term robustness").
- [V] YCBInEOAT (Table 4, model-based, GT init): ADD-S 96.42 / ADD 93.09; with own estimation init 96.40 / 93.22;
  se(3)-TrackNet 95.53 / 92.66; BundleSDF 93.77 / 86.95; BundleTrack 92.53 / 87.34. Per object ADD: cracker 91.32, bleach
  91.45, sugar 94.14, **tomato soup can 91.71**, mustard 96.34. YCB-Video tracking (Table 5) all-frames 96.0 / 97.9;
  model-free 93.7 / 97.5.
- [V] Refiner trained on perturbations up to 2/2/5 cm and 20 deg -> its basin is ~20 deg; this is why it collapses under
  RoPT's 8-16x downsampling. Limitation: depends on external 2D detection; fails on textureless + occluded objects (Fig. 11).
- **For us.** (1) The baseline every reviewer will ask for; run it ourselves (model-free mode with reference images taken
  from our splat renders = same object knowledge as POGS). (2) Its tracking mode is a *learned* render-and-compare refiner
  with no uncertainty -> our calibrated belief is orthogonal and could even wrap it (stretch: show the information-matrix
  idea is tracker-agnostic). (3) Report its fps measured on our GPU, not 32 Hz.

## Wen et al. 2023, BundleSDF (CVPR 2023), arXiv 2303.14158 -> `2023_CVPR_Wen_BundleSDF.pdf`
- [V] Model-free causal tracking + reconstruction from RGB-D with a first-frame mask only: video segmentation (XMem-type)
  -> LoFTR matches + RANSAC coarse pose -> online pose graph over <= K = 10 memory frames (feature-correspondence Huber +
  point-to-plane + SDF unary term), Gauss-Newton 7 steps in CUDA -> concurrent Neural Object Field (hash-grid SDF + colour,
  hybrid SDF with "uncertain free space" for occluded/missing pixels) refines memory poses every ~6.7 s.
- [V] HO3D (13 videos, 20,428 frames): ADD-S 96.52 / ADD 92.62 (BundleTrack 92.39 / 66.01). YCBInEOAT (9 videos, 7,449
  frames): 93.77 / 86.95 (BundleTrack 93.01 / 87.26). BEHAVE (70 videos single-view, AUC to 0.5 m): 83.63 / 67.52.
  10.2 Hz on RTX 3090 (seg 18 ms, matching 24 ms, pose graph 56 ms). No re-initialisation on failure.
- [V] Protocol details worth copying: first frame's ground-truth pose defines the canonical frame (no CAD for the method);
  ADD/ADD-S AUC 0-0.1 m; same video masks for all baselines; BEHAVE camera chosen by least occlusion (offline, GT-based --
  disclosed). Failure (App. F): severe occlusion + segmentation error + low texture together; "when the object re-appears,
  the recovered pose is affected by symmetric geometry".
- **For us.** (1) Primary model-free baseline and the source of the evaluation protocol (HO3D + YCBInEOAT). (2) The
  "uncertain free space" idea = principled handling of occluder pixels; our gripper/hand mask ablation should cite it.
  (3) Symmetric re-acquisition failure is again documented -- supports the weak-axis multimodal belief.

## Murphy, Esteves, Jampani, Ramalingam, Makadia 2021, Implicit-PDF (ICML 2021), arXiv 2106.05965 -> `2021_ICML_Murphy_Implicit-PDF-SO3-distributions.pdf`
- [V] An MLP f(x, R) gives an unnormalised log density on SO(3); normalised on an equivolumetric HEALPix-Hopf grid
  (Yershova 2010; 72 * 8^k points, eval at ~2.36M); trained with NLL of a single ground-truth pose per image (no symmetry
  labels). Inference: grid argmax + gradient ascent with projection, or full distribution.
- [V] Metrics (Sec. 3.6): **average log-likelihood of the ground truth** (uniform on SO(3) = -2.29; invariant to having one
  or all equivalent ground truths), spread / MAAD (needs all symmetric ground truths), precision (median error, Acc@theta),
  top-k recall from connected-component modes.
- [V] SYMSOL I (textureless cone, cylinder, tet, cube, icosahedron) and **SYMSOL II (near-symmetric: marked sphere, cylinder
  with an off-centre dot, tetrahedron with one red face)**: IPDF LL 4.10 / 7.57 vs Bingham mixtures (Gilitschenski 2019,
  Deng 2020) and von Mises (Prokudin 2018) far lower (Table 1); mixtures suffer mode collapse and cannot represent a
  continuous orbit (cone/cylinder) well (spread 10-15 deg vs 1.4 deg, Table S1). ModelNet10-SO(3) top-2 Acc@15 0.868 vs top-1
  0.719. Cost: 18.3 fps at 37k queries, 2.4 fps at 2.36M (ResNet-50).
- **For us.** (1) Evaluation language for P3: report the log-likelihood of the true rotation (and of the true roll on the
  weak axis), with the uniform baseline, plus top-k recall. (2) The "marked cylinder" (SYMSOL II cylO) is exactly the
  near-symmetric drill/mug case: the density should collapse when the mark is visible and spread over a half-orbit when
  not -- a crisp, testable prediction for our weak-axis belief in simulation (our decal intervention). (3) We do not need
  a learned density: our likelihood comes from the render loss along the weak axis (1-D), so a 1-D grid on the orbit is
  cheap; but we must justify that exp(-L/T) is calibrated (temperature fitted on held-out sequences).
