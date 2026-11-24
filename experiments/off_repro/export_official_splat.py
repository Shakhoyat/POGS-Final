"""8.OFF-S (exploratory): export the official splat + the official drill selection for use with OUR tracking code.
Official venv:  ~/venvs/pogs-official/bin/python -m experiments.off_repro.export_official_splat --config <config.yml>
Writes ~/pogs_external/off/official_splat_for_ours.pt (local, not in git)."""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.off_repro import shims  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

wp.init()
from experiments.off_repro.official_sweeps import FR, setup  # noqa: E402


def main(config: str) -> None:
    os.chdir(os.path.expanduser("~/pogs_external/off"))
    ck = sorted(glob.glob(os.path.join(os.path.dirname(config), "nerfstudio_models", "*.ckpt")))[-1]
    sd = torch.load(ck, map_location="cpu")["pipeline"]
    gp = {k.split("gauss_params.")[1]: v for k, v in sd.items() if ".gauss_params." in k}
    nn_sd = {k.split("_model.nn.")[1]: v for k, v in sd.items() if k.startswith("_model.nn.")}
    idx = json.load(open(os.path.join(FR, "index.json")))
    pipeline, opt, info = setup(config, idx["click_xyz"], idx["table_z"])
    keep = pipeline.model.keep_inds.detach().cpu()
    pca = torch.from_numpy(np.load("outputs/tool_table_ns/pca.npy"))
    out = os.path.expanduser("~/pogs_external/off/official_splat_for_ours.pt")
    torch.save({"gauss_params": gp, "nn": nn_sd, "keep": keep, "dino_pca": pca, "ckpt": ck, "setup": info}, out)
    print("exported", out, {k: tuple(v.shape) for k, v in gp.items()}, "nn", list(nn_sd), "keep", len(keep),
          "pca", tuple(pca.shape))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
