"""Check whether a model's weights can actually be obtained.

Two of the study's rows name models I could not verify when the folder was
built -- LocateAnything3D and VFM-DETR. Rather than discover that after
transferring a folder and installing an environment, this probes for the
weights and records *why* a model is unavailable.

A row with an unresolved model reports ``blocked`` with a reason. That is a
result, not a failure: "this substitution cannot be evaluated" is something the
comparison needs to say.

Usage::

    python probe_weights.py --models locateanything3d vfmdetr
    python probe_weights.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402
import download_weights  # noqa: E402

#: Public entry points, checked before anything heavy is attempted. A HEAD
#: request is enough to establish whether a weights URL exists, and costs
#: nothing on a machine with no weights cached yet.
#:
#: Sourced from ``download_weights.FILES`` rather than maintained separately. An
#: earlier version of this file carried its own table and had drifted: it listed
#: ``mobileclipv2`` and ``locateanything3d`` with no URLs at all, so the probe
#: reported both blocked when in fact MobileCLIP2 is public and ungated on
#: HuggingFace. Two tables for one fact is how that happened.
URLS: dict[str, list[str]] = {
    name: [url for _, url, _ in files]
    for name, files in download_weights.FILES.items()
}

#: No source exists at all. Never probed, never downloaded.
NO_SOURCE = {"vfmdetr"}

#: Models that need a licence or a token rather than an anonymous download.
GATED = {
    "dinov3": "HuggingFace licence acceptance + token required",
    "dino_txt": "HuggingFace licence acceptance + token required",
}

TIMEOUT = 20


def probe(name: str) -> dict:
    rec: dict = {"model": name, "urls": URLS.get(name, [])}
    # NO_SOURCE is checked before the registry so a model can be declared
    # source-less without also needing a (deliberately empty) FILES entry, and so
    # "blocked" wins over "unknown" -- blocked is the actionable word.
    if name not in URLS and name not in NO_SOURCE:
        return {**rec, "status": "unknown",
                "reason": "not in the probe table"}

    if name in NO_SOURCE or not rec["urls"]:
        return {**rec, "status": "blocked",
                "reason": (cfg.MODELS.get(name, {}).get("note")
                           or "no public weights URL identified; cannot be "
                              "evaluated without a source")}

    results = []
    for url in rec["urls"]:
        entry = {"url": url}
        try:
            req = urllib.request.Request(url, method="HEAD",
                                         headers=download_weights.auth_headers(url))
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                entry["status"] = "reachable"
                entry["code"] = resp.status
                length = resp.headers.get("Content-Length")
                if length and length.isdigit():
                    entry["size_mb"] = round(int(length) / 1024**2, 1)
        except urllib.error.HTTPError as exc:
            entry["status"] = "http-error"
            entry["code"] = exc.code
            entry["reason"] = str(exc.reason)
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "unreachable"
            entry["reason"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)

    rec["checks"] = results
    reachable = [c for c in results if c["status"] == "reachable"]
    gated = [c for c in results if c.get("code") == 401 or c.get("code") == 403]

    if reachable:
        rec["status"] = "available"
    elif gated or name in GATED:
        rec["status"] = "gated"
        rec["reason"] = GATED.get(name, "authentication required")
    else:
        rec["status"] = "blocked"
        rec["reason"] = "; ".join(
            f"{c.get('code', c['status'])} {c.get('reason', '')}" for c in results
        )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe model weight availability")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--row", default=cfg.BASELINE_ROW)
    args = ap.parse_args()

    names = args.models
    if args.all or not names:
        names = sorted(set(URLS) | set(cfg.MODELS))

    results = [probe(n) for n in names]

    print("=" * 72)
    print("WEIGHT AVAILABILITY PROBE")
    print("=" * 72)
    width = max(len(r["model"]) for r in results)
    for r in results:
        detail = r.get("reason", "")
        if not detail and r.get("checks"):
            c = r["checks"][0]
            detail = f"{c.get('code', '')} {c.get('size_mb', '')}".strip()
        print(f"  {r['model']:<{width}}  {r['status']:<11} {detail}")

    out_dir = os.path.join(cfg.ARTEFACTS, args.row, "probe")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "weights.json")
    with open(path, "w") as fh:
        json.dump({"results": results}, fh, indent=2)
    print("=" * 72)
    print(f"  wrote {path}")

    blocked = [r["model"] for r in results if r["status"] == "blocked"]
    if blocked:
        print(f"  BLOCKED rows must report as blocked: {', '.join(blocked)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())