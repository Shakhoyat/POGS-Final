# Model Checkpoints Directory

Stores trained 3D Gaussian Splats, semantic feature fields, and segmentation cluster labels.

## Structure:
- `v1_baseline/`:
  - `pogs_gaussians.pt`: PyTorch dict containing 3D Gaussian parameters (means, scales, rots, opacities, shs).
  - `pogs_fields.pt`: Weights for language & DINO feature field neural networks.
  - `cluster_labels.npy`: Persistent object cluster IDs.
  - `train_summary.json`: Training losses, time elapsed, final Gaussian count.
- `v2_kalman_opt/`:
  - Checkpoints for optimized/fine-tuned configurations.
