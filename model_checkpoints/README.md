# Model checkpoints

`pick_place_floor_table/` -- the single trained POGS for the benchmark scene, shared
by v1, v2 and v3 (they track different objects in the same scene).

| File | Contents |
|---|---|
| `pogs_checkpoint.pt` | everything tracking needs: per-Gaussian tensors, feature field, DINO projection head, **DINO PCA basis**, training step, config (`model/pogs/checkpoint.py`) |
| `train_summary.json` | Gaussian count, train time, backend, train-view PSNR / depth error, loss log |
| `cluster_labels.npy` | HDBSCAN label per Gaussian (cache; deleted = re-clustered) |
| `object_clusters.json` | tracked groups, their matched GT object (evaluation only) and IoU |

`*.pt` files are git-ignored. Produce with `python -m experiments.v1_base_pogs.run_all`.
