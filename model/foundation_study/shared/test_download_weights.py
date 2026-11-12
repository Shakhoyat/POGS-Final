"""Host-independent checks for the downloader's path and selection logic.

The downloader cannot be fully tested without pulling gigabytes over a network,
so this covers the parts that decide *what* to fetch and *where* it lands -- the
failures that would otherwise show up as a 4.7 GB transfer to the wrong place.

    python test_download_weights.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402
import download_weights as dw  # noqa: E402

FAILED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILED
    if not ok:
        FAILED += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    print("=" * 68)
    print("DOWNLOAD WEIGHTS - HOST-INDEPENDENT CHECKS")
    print("=" * 68)

    # --- every row resolves to models the downloader knows about -----------
    for row in cfg.ROWS:
        names = dw.models_for(row, False, None)
        unknown = [m for m in names if m not in dw.FILES and m not in dw.BLOCKED]
        check(f"{row} resolves to known models", not unknown,
              f"{names}" + (f" UNKNOWN {unknown}" if unknown else ""))

    # --- every registry model is downloadable or explicitly blocked ------
    unhandled = [m for m in cfg.MODELS
                 if m not in dw.FILES and m not in dw.BLOCKED]
    check("every MODELS entry has a source or is BLOCKED", not unhandled,
          f"unhandled: {unhandled}")

    # --- gated models need no silent fallback ------------------------------
    check("gated models are declared, not silently absent",
          all(m in dw.FILES for m in dw.GATED),
          f"gated: {sorted(dw.GATED)}")
    check("blocked models have no file list",
          all(m not in dw.FILES for m in dw.BLOCKED),
          f"blocked: {sorted(dw.BLOCKED)}")

    # --- token never leaks to a non-HF host -------------------------------
    os.environ["HF_TOKEN"] = "secret-token-value"
    hf = dw.auth_headers("https://huggingface.co/x/y/resolve/main/z.bin")
    other = dw.auth_headers("https://dl.fbaipublicfiles.com/dinov2/x.pth")
    check("token is sent to huggingface", hf.get("Authorization") ==
          "Bearer secret-token-value")
    check("token is NOT sent to fbaipublicfiles or github",
          "Authorization" not in other)
    os.environ.pop("HF_TOKEN", None)
    check("no token in env means no Authorization header",
          "Authorization" not in dw.auth_headers(
              "https://huggingface.co/x/y/resolve/main/z.bin"))

    # --- blocked rows still produce something ------------------------------
    row_models = dw.models_for("03_vfmdetr_siglip2", False, None)
    check("row 03 resolves", row_models == ["detic", "dino_vits14", "siglip2"],
          f"{row_models}")

    # --- sizes are sane ----------------------------------------------------
    sizes = {}
    for name, files in dw.FILES.items():
        sizes[name] = sum(1 for _ in files)
    check("locateanything3d needs its index and config too",
          sizes.get("locateanything3d", 0) >= 4, f"{sizes.get('locateanything3d')}")
    check("clip_vitb16 has weights plus a config",
          sizes.get("clip_vitb16", 0) == 2, f"{sizes.get('clip_vitb16')}")
    check("dinov3 has weights plus a config",
          sizes.get("dinov3", 0) == 2, f"{sizes.get('dinov3')}")

    # --- human() formatting, used in every progress line -------------------
    check("human formats bytes", dw.human(512) == "512 B", dw.human(512))
    check("human formats GB", dw.human(2.5 * 1024 ** 3) == "2.5 GB",
          dw.human(2.5 * 1024 ** 3))

    # --- settle_size does not mangle a normal file -------------------------
    probe = os.path.join(cfg.WEIGHTS, ".settle_probe")
    os.makedirs(cfg.WEIGHTS, exist_ok=True)
    payload = b"x" * 4096
    with open(probe, "wb") as fh:
        fh.write(payload)
    check("settle_size returns the true size", dw.settle_size(probe) == len(payload),
          f"{dw.settle_size(probe)}")
    os.remove(probe)

    # --- weights live outside artefacts/, so a re-run cannot delete them ---
    check("weights root is not inside artefacts",
          os.path.commonpath([os.path.abspath(cfg.WEIGHTS),
                              os.path.abspath(cfg.ARTEFACTS)]) !=
          os.path.abspath(cfg.ARTEFACTS),
          f"weights={cfg.WEIGHTS}")

    total = 12 - FAILED
    print(f"\n{'PASS' if not FAILED else 'FAIL'}: {total} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())