"""
Metrics (evaluation only; never used to build groups, select objects or initialise).

    metrics.pose      ADD / ADD-S, translation / rotation error, success criteria
    metrics.gaussian  splat KL divergence, rendered-mask IoU / leakage, PSNR / SSIM
    metrics.stats     Wilson intervals, target verdicts, mean +- std over seeds
"""
