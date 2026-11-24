"""Train the OFFICIAL POGS on the exported drill capture (8.OFF). Run with the official venv:

    ~/venvs/pogs-official/bin/python -m experiments.off_repro.official_train [--data ...] [--work ...]

1. Fill the official Detic cache (`<work>/outputs/<data name>/detic.npy` + `.info`) from the simulator instance
   masks, in the official train-split order, with the official post-processing (3x3 erosion rule, masks larger than
   H*W/3.5 dropped, background mask last, nearest downscale by clip_downscale_factor, flattened). Floor and table
   ids are excluded as in our re-implementation.
2. Run `ns-train pogs` (official method config, official defaults) with `--vis tensorboard` (no interactive viewer).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.off_repro import shims  # noqa: E402,F401  (must precede any pogs import)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

FLOOR_ID, TABLE_ID = 0, 9


def detic_like(ids: np.ndarray, downscale: int) -> np.ndarray:
    H, W = ids.shape
    raw = [ids == i for i in np.unique(ids) if i not in (FLOOR_ID, TABLE_ID)]
    small = [m for m in raw if m.sum() <= H * W / 3.5]
    comp = np.zeros((H, W), bool)
    for m in small:
        comp |= m
    bg = ~comp
    if not small:
        masks = np.zeros((0, 1, H, W), bool)
    else:
        masks = np.stack(small)[:, None]
    t = torch.from_numpy(masks)
    er = torch.conv2d((~t).float(), torch.ones(1, 1, 3, 3), padding=1) if len(t) else t.float()
    er = ~(er >= 2)
    filt = torch.cat([er, torch.from_numpy(bg)[None, None]], 0)
    h, w = H // downscale, W // downscale
    if downscale > 1:
        filt = F.interpolate(filt.to(float), (h, w), mode="nearest").to(bool).squeeze(1).view(-1, h * w)
    return filt.numpy()


def fill_detic_cache(data: str, work: str, downscale: int = 2) -> int:
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    dp = NerfstudioDataParserConfig(data=__import__("pathlib").Path(data), load_3D_points=True,
                                    orientation_method="none", center_method="none", auto_scale_poses=False,
                                    depth_unit_scale_factor=1.0).setup()
    outs = dp.get_dataparser_outputs("train")
    arrs = []
    for fn in outs.image_filenames:
        ids = np.load(os.path.join(data, "masks_sim", os.path.splitext(os.path.basename(fn))[0] + ".npy"))
        arrs.append(detic_like(ids, downscale))
    cache = os.path.join(work, "outputs", os.path.basename(os.path.normpath(data)))
    os.makedirs(cache, exist_ok=True)
    obj = np.empty(len(arrs), dtype=object)
    obj[:] = arrs
    np.save(os.path.join(cache, "detic.npy"), obj)
    with open(os.path.join(cache, "detic.info"), "w") as f:
        f.write(json.dumps({"sam": False, "custom_vocab": [], "downscale_factor": downscale}))
    json.dump([str(p) for p in outs.image_filenames], open(os.path.join(cache, "train_split.json"), "w"), indent=0)
    print(f"Detic cache: {len(arrs)} train images, masks per image {[a.shape[0] for a in arrs]}")
    return len(arrs)


def main(data: str, work: str, seed: int) -> None:
    data, work = os.path.expanduser(data), os.path.expanduser(work)
    os.makedirs(work, exist_ok=True)
    os.chdir(work)  # official caches live under ./outputs/<data name>
    fill_detic_cache(data, work)
    from nerfstudio.scripts.train import entrypoint
    sys.argv = ["ns-train", "pogs", "--data", data, "--output-dir", os.path.join(work, "runs"),
                "--vis", "tensorboard", "--machine.seed", str(seed), "--viewer.quit-on-train-completion", "True"]
    entrypoint()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/pogs_external/off/tool_table_ns")
    ap.add_argument("--work", default="~/pogs_external/off")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.data, a.work, a.seed)
