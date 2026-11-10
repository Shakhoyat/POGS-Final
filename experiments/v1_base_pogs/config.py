"""
v1 configuration: base POGS with the official release's settings.

Everything comes from ``common.config`` defaults, which are the release's values
(see that module for sources). The only scene-specific settings are which
instance ids a detector would treat as background, and the support-plane height
used when selecting tracked clusters.

Method names double as keys of the shared run cache
(``experiments.pipeline.RUNS_DIR``): the same name must always mean the same
settings, so v2/v3 reuse these runs instead of repeating them.
"""

from __future__ import annotations

from common.config import POGSConfig
from scene_generator.pick_place_benchmark import FLOOR_ID, TABLE_ID

RESULTS_DIR = "data_results/v1_base_pogs"

# (method name, ``common.evaluate.variant_config`` variant)
V1_MAIN = [
    ("POGS", "POGS"),                                                   # paper: DINO + depth every frame
    ("POGS (release: no per-frame depth)", "POGS (release: no per-frame depth)"),
]
# The paper's ablations; run on the base condition only (they are not
# candidates for "better", just a sanity check of each loss term).
V1_ABLATIONS = [
    ("No Depth", "No Depth"),
    ("No DINO (RGB)", "No DINO"),
]


def get_v1_config(device: str = "cuda:0") -> POGSConfig:
    cfg = POGSConfig(device=device)
    # Detic would not report the floor or the support table as objects; they
    # join the negative mask (Sec. IV-B).
    cfg.train.mask_background_ids = (FLOOR_ID, TABLE_ID)
    cfg.runtime.output_dir = RESULTS_DIR
    return cfg
