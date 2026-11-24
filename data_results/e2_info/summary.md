# E2: information of the POGS loss about drill-bit roll (evaluation only)

40 trials, 1120 frames (POGS paper variant; probe does not alter tracking). Rotation scaled by radius 0.1 m.

| Quantity | At estimate | At ground truth |
|---|---|---|
| Spearman(log roll information, abs roll error) [P1: <= -0.5] | 0.22 (p=3.3e-14) | -0.20 (p=1.7e-11) |
| Weakest rotation direction within 15° of bit axis [P1: >= 70%] | 17% | 27% |
| Median roll information / strongest rotation information | 0.144 | 0.102 |
| Median share of bit axis in the 2-D weak rotation subspace (1 = fully weak) | 1.00 | 1.00 |
| Median quadratic-fit residual of the loss (1°/3 mm) | 0.23 | – |

## E1 (GPU part): loss at the estimate vs at the ground truth

| Phase | n | loss(est) < loss(GT) | median loss(est) − loss(GT) |
|---|---|---|---|
| approach | 160 | 100% | -0.0050 |
| lift | 240 | 67% | -0.0022 |
| rest_x | 120 | 97% | -0.0056 |
| perturbed | 600 | 87% | -0.0054 |

Reading: loss(est) < loss(GT) on most frames means the loss itself prefers the wrong pose (features / depth cannot separate roll) — not an optimiser failure. P1 verdict: FAIL (at the estimate).
Static baseline: not applicable (information of the loss, not a tracker comparison).
