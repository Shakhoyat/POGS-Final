"""Download every model's weights, per row, with a verifiable manifest.

Two jobs, and the second matters more than the first.

* **Download.** One folder per model under ``weights/``. Resumable, verified
  against ``Content-Length``, and it stops cleanly on a gated model instead of
  dying halfway through a 4.7 GB shard.

* **Record.** ``weights/manifest.json`` gets the URL, byte count and sha256 of
  every file, so a later run on the GPU host can *prove* the weights it has are
  the weights this machine downloaded. Without that, "the experiment used the
  right checkpoint" is a claim nobody can check.

Gated models (``dinov3``) need ``HF_TOKEN`` in the environment.
They are reported ``gated``, not failed -- the rest of the row still runs.

Usage::

    python download_weights.py --row 00_baseline_detic_dinov2_clip
    python download_weights.py --row 02_rfdetr_dinov3_mobileclip
    python download_weights.py --all
    python download_weights.py --models dino_vits14 siglip2
    python download_weights.py --verify
    python download_weights.py --verify --strict     # non-zero exit if any gap

Sizes are large. Row 01 wants LocateAnything3D (~9.5 GB across two shards), so
``--all`` is roughly 13 GB. Prefer per-row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402

HF = "https://huggingface.co"

#: model -> list of (relative-path-under-weights/model, url, required)
#:
#: ``required`` is False for a file whose absence only reduces capability rather
#: than breaking the row -- a config JSON where the weights are the expensive
#: part, say. Everything required must be present for ``--verify --strict`` to
#: pass.
FILES: dict[str, list[tuple[str, str, bool]]] = {
    "dino_vits14": [(
        "dinov2_vits14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"
        "dinov2_vits14_pretrain.pth",
        True,
    )],
    "dino_vitl14": [(
        "dinov2_vitl14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/"
        "dinov2_vitl14_pretrain.pth",
        True,
    )],
    "detic": [(
        "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        "https://dl.fbaipublicfiles.com/detic/"
        "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        True,
    )],
    "clip_vitb16": [
        ("open_clip_pytorch_model.bin",
         f"{HF}/laion/CLIP-ViT-B-16-laion2B-s34B-b88K/resolve/main/"
         "open_clip_pytorch_model.bin", True),
        ("open_clip_config.json",
         f"{HF}/laion/CLIP-ViT-B-16-laion2B-s34B-b88K/resolve/main/"
         "open_clip_config.json", False),
    ],
    # Row 03 loads this via AutoModel+AutoProcessor from weights/siglip2/ when
    # POGS_WEIGHTS points at it (offline-first). config + preprocessor +
    # tokenizer are what make a bare safetensors loadable as a full model.
    "siglip2": [
        ("model.safetensors",
         f"{HF}/google/siglip2-base-patch16-224/resolve/main/model.safetensors",
         True),
        ("config.json",
         f"{HF}/google/siglip2-base-patch16-224/resolve/main/config.json",
         False),
        ("preprocessor_config.json",
         f"{HF}/google/siglip2-base-patch16-224/resolve/main/"
         "preprocessor_config.json", False),
        ("tokenizer.json",
         f"{HF}/google/siglip2-base-patch16-224/resolve/main/tokenizer.json",
         False),
        ("tokenizer_config.json",
         f"{HF}/google/siglip2-base-patch16-224/resolve/main/"
         "tokenizer_config.json", False),
    ],
    "yolo": [(
        "yolo11x.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
        "yolo11x.pt",
        True,
    )],
    "rfdetr": [
        ("model.safetensors",
         f"{HF}/Roboflow/rf-detr-base/resolve/main/model.safetensors", True),
        ("config.json",
         f"{HF}/Roboflow/rf-detr-base/resolve/main/config.json", False),
    ],
    # timm's OpenCLIP re-export: ships open_clip_config.json, which is what lets
    # it load through POGS's existing open_clip.create_model_and_transforms
    # call with nothing but a different clip_model_type string.
    "mobileclipv2": [
        ("open_clip_pytorch_model.bin",
         f"{HF}/timm/MobileCLIP2-B-OpenCLIP/resolve/main/"
         "open_clip_pytorch_model.bin", True),
        ("open_clip_config.json",
         f"{HF}/timm/MobileCLIP2-B-OpenCLIP/resolve/main/open_clip_config.json",
         False),
    ],
    "locateanything3d": [
        ("model-00001-of-00002.safetensors",
         f"{HF}/nvidia/LocateAnything-3B/resolve/main/"
         "model-00001-of-00002.safetensors", True),
        ("model-00002-of-00002.safetensors",
         f"{HF}/nvidia/LocateAnything-3B/resolve/main/"
         "model-00002-of-00002.safetensors", True),
        ("model.safetensors.index.json",
         f"{HF}/nvidia/LocateAnything-3B/resolve/main/"
         "model.safetensors.index.json", True),
        ("config.json",
         f"{HF}/nvidia/LocateAnything-3B/resolve/main/config.json", True),
        ("preprocessor_config.json",
         f"{HF}/nvidia/LocateAnything-3B/resolve/main/preprocessor_config.json",
         False),
        ("tokenizer_config.json",
         f"{HF}/nvidia/LocateAnything-3B/resolve/main/tokenizer_config.json",
         False),
    ],
    "dinov3": [
        ("model.safetensors",
         f"{HF}/facebook/dinov3-vitb16-pretrain-lvd1689m/resolve/main/"
         "model.safetensors", True),
        # AutoModel.from_pretrained(weights/dinov3/) needs the config next to
        # the snapshot; loading by repo id misses this folder entirely.
        ("config.json",
         f"{HF}/facebook/dinov3-vitb16-pretrain-lvd1689m/resolve/main/"
         "config.json", False),
    ],
    # dino.txt is four public files on Meta's server, not one: the backbone, a
    # vision head, a text encoder and the BPE vocabulary. This is exactly what
    # dinov2/hub/dinotxt.py fetches. They go to the torch-hub checkpoint cache
    # (see HUB_CACHED), so torch.hub.load finds them with no download at run time.
    # No token: an earlier single-file URL here returned 403, which this script
    # misreported as "gated".
    "dino_txt": [
        ("dinov2_vitl14_reg4_pretrain.pth",
         "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/"
         "dinov2_vitl14_reg4_pretrain.pth", True),
        ("dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
         "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/"
         "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth", True),
        ("dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth",
         "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/"
         "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth", True),
        ("bpe_simple_vocab_16e6.txt.gz",
         "https://dl.fbaipublicfiles.com/dinov2/thirdparty/"
         "bpe_simple_vocab_16e6.txt.gz", False),
    ],
}

#: Models whose files must sit in the torch-hub checkpoint cache rather than in
#: ``weights/<model>/``, because the library that loads them looks there.
HUB_CACHED = {"dino_txt"}

# LocateAnything3D is a custom-code model (tag ``custom_code``): transformers
# builds it from Python files shipped in the repo, so weights + config alone
# cannot be loaded. These are the modules modeling_locateanything.py imports,
# plus the processor and tokenizer files. All small; the .py files are required.
_LA = f"{HF}/nvidia/LocateAnything-3B/resolve/main/"
FILES["locateanything3d"] += [
    (f, _LA + f, True) for f in (
        "configuration_locateanything.py", "configuration_qwen2.py",
        "modeling_locateanything.py", "modeling_qwen2.py", "modeling_vit.py",
        "mask_sdpa_utils.py", "mask_magi_utils.py", "generate_utils.py",
        "image_processing_locateanything.py", "processing_locateanything.py")
] + [
    (f, _LA + f, False) for f in (
        "processor_config.json", "generation_config.json", "added_tokens.json",
        "special_tokens_map.json", "merges.txt", "vocab.json", "chat_template.json")
]

#: Needs an HF token, or HTTP 401. Reported, not failed.
GATED = {"dinov3"}

#: No source exists at all. Never attempted.
BLOCKED = {"vfmdetr"}

TIMEOUT = 60
CHUNK = 1 << 20          # 1 MiB
RETRIES = 3


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            # Whole bytes read better than "512.0 B" in a progress line, and a
            # byte count is the one case where the fraction is never useful.
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def manifest_path() -> str:
    return os.path.join(cfg.WEIGHTS, "manifest.json")


def load_manifest() -> dict:
    p = manifest_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return {}


def save_manifest(man: dict) -> None:
    os.makedirs(cfg.WEIGHTS, exist_ok=True)
    with open(manifest_path(), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)


def auth_headers(url: str) -> dict:
    """Bearer token, but only for HuggingFace.

    Sending the token to a non-HF host would leak it to Meta or GitHub. The
    token is read from ``HF_TOKEN`` because several harnesses expose it that way.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and url.startswith(HF):
        return {"Authorization": f"Bearer {token}"}
    return {}


def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD", headers=auth_headers(url))
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n and n.isdigit() else None
    except Exception:  # noqa: BLE001
        return None


def sha256(path: str, progress: bool = False) -> str:
    h = hashlib.sha256()
    done = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
            done += len(block)
            if progress:
                pct = done / max(os.path.getsize(path), 1) * 100
                print(f"\r    hashing {pct:5.1f}%", end="", flush=True)
    if progress:
        print()
    return h.hexdigest()


def settle_size(path: str, tries: int = 5, delay: float = 0.4) -> int:
    """File size, polled until two consecutive reads agree.

    Written because this bit us: a download to the network-mounted ``G:`` drive
    reported 88283115 bytes, matched ``Content-Length`` exactly, and still failed
    the size check on the first attempt -- the stat returned a stale count that
    the very next read corrected. Failing the download over that would have left
    a valid file rejected and a retry to redo 84 MB. Polling costs nothing when
    the size is already right, which is the overwhelmingly common case.
    """
    last = -1
    for _ in range(tries):
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        if size == last and size >= 0:
            return size
        last = size
        time.sleep(delay)
    return last


def fetch(url: str, dest: str, expected: int | None) -> str:
    """Download with resume. Returns a status string.

    Resume via a ``.part`` file and a Range header. Retries from where the last
    attempt stopped rather than restarting, because LocateAnything3D's single
    shard is 4.7 GB and restarting that on every blip is not acceptable.
    """
    part = dest + ".part"
    for attempt in range(1, RETRIES + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if expected is not None and have == expected:
            os.replace(part, dest)
            return "resumed-complete"
        headers = auth_headers(url)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                # A server that ignores Range replies 200 with the whole body,
                # so the partial file must be discarded or the result is corrupt.
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                total = expected or 0
                t0 = time.time()
                with open(part, mode) as fh:
                    while True:
                        block = r.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
                        have += len(block)
                        rate = have / max(time.time() - t0, 1e-6)
                        eta = ((total - have) / rate) if total and rate else 0
                        print(f"\r    {human(have)}/{human(total) if total else '?'}"
                              f"  {human(rate)}/s  eta {eta / 60:.1f}m   ",
                              end="", flush=True)
                    # The size check below reads the file back through the
                    # filesystem, and on a network mount that stat can still be
                    # catching up with the last write. fsync first so the check is
                    # comparing against flushed bytes.
                    fh.flush()
                    os.fsync(fh.fileno())
            print()

            got = settle_size(part)
            if expected is not None and got != expected:
                raise IOError(f"size mismatch: got {got}, expected {expected}")
            os.replace(part, dest)
            return "downloaded"
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return f"gated (HTTP {exc.code})"
            if exc.code == 404:
                return "missing (HTTP 404)"
            print(f"\n    attempt {attempt}/{RETRIES}: HTTP {exc.code} "
                  f"{exc.reason}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n    attempt {attempt}/{RETRIES}: {type(exc).__name__}: {exc}")
        time.sleep(min(2 ** attempt, 10))
    return "failed"


def models_for(row: str | None, all_rows: bool, explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    if all_rows:
        return sorted(set(cfg.MODELS))
    if row:
        meta = cfg.ROWS.get(row)
        if meta is None:
            raise SystemExit(f"unknown row: {row}")
        wanted = [meta.get("detection"), meta.get("features"),
                  meta.get("alignment")]
        if meta.get("available_models"):
            wanted += meta["available_models"]
        return [m for m in dict.fromkeys(wanted) if m]
    return sorted(set(cfg.MODELS))


def main() -> int:
    ap = argparse.ArgumentParser(description="Download and verify model weights")
    ap.add_argument("--row", default=None, help="row whose models to fetch")
    ap.add_argument("--list-scenes", action="store_true",
                    help="print the scene matrix, one per line, and exit. "
                         "Weights are per model rather than per scene, so this "
                         "only exists so run.sh's sweep has one scene list.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--dest", default=None, help="weights root (default: "
                    "model_study/weights)")
    ap.add_argument("--verify", action="store_true",
                    help="check what is present, download nothing")
    ap.add_argument("--strict", action="store_true",
                    help="with --verify: exit non-zero if anything is missing")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip sha256 (fast, but the manifest cannot prove "
                         "which bytes were used)")
    args = ap.parse_args()

    if args.list_scenes:
        for s in cfg.SCENES:
            print(s)
        return 0

    if args.dest:
        cfg.WEIGHTS = os.path.abspath(args.dest)
    os.makedirs(cfg.WEIGHTS, exist_ok=True)

    names = models_for(args.row, args.all, args.models)
    man = load_manifest()

    print("=" * 72)
    print("WEIGHT DOWNLOADER")
    print("=" * 72)
    print(f"  root   : {cfg.WEIGHTS}")
    print(f"  models : {', '.join(names)}")
    if any(n in GATED for n in names):
        print(f"  gated  : {', '.join(n for n in names if n in GATED)}"
              "  (set HF_TOKEN for these)")
    print()

    missing_required: list[str] = []
    blocked: list[str] = []

    for name in names:
        files = FILES.get(name)
        info = cfg.MODELS.get(name, {})

        if name in BLOCKED or not files:
            reason = info.get("note", "no weights source identified")
            print(f"  {name:<20} BLOCKED")
            print(f"      {reason}")
            blocked.append(name)
            man.setdefault(name, {})["status"] = "blocked"
            man[name]["reason"] = reason
            continue

        mdir = (os.path.join(cfg.WEIGHTS, "torch", "hub", "checkpoints")
                if name in HUB_CACHED else os.path.join(cfg.WEIGHTS, name))
        os.makedirs(mdir, exist_ok=True)
        rec = man.setdefault(name, {})
        rec["status"] = "partial"

        present = required = 0
        for fname, url, is_required in files:
            dest = os.path.join(mdir, fname)
            want = remote_size(url)
            have = os.path.getsize(dest) if os.path.exists(dest) else 0

            if want is not None and have == want:
                print(f"  {name:<20} have {fname}  ({human(have)})")
                present += 1
                required += int(is_required)
                rec[fname] = {"url": url, "bytes": want, "status": "present"}
                continue

            if args.verify:
                print(f"  {name:<20} MISS {fname}"
                      + (f"  ({human(have)}/{human(want)})" if have and want else ""))
                rec[fname] = {"url": url, "bytes": want, "status": "missing"}
                if is_required:
                    missing_required.append(f"{name}/{fname}")
                continue

            print(f"  {name:<20} get  {fname}  ({human(want or 0)})")
            status = fetch(url, dest, want)
            if status.startswith("gated"):
                print(f"      gated: needs HF_TOKEN")
                rec[fname] = {"url": url, "bytes": want, "status": "gated"}
                missing_required.append(f"{name}/{fname} (gated)")
            elif status == "failed":
                rec[fname] = {"url": url, "bytes": want, "status": "failed"}
                if is_required:
                    missing_required.append(f"{name}/{fname}")
            else:
                size = os.path.getsize(dest)
                entry = {"url": url, "bytes": size, "status": "present",
                         "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S")}
                if not args.no_hash:
                    entry["sha256"] = sha256(dest, progress=False)
                rec[fname] = entry
                present += 1
                required += int(is_required)
                print(f"      {status}, {human(size)}"
                      + (f", sha256 {entry['sha256'][:16]}..." if "sha256" in entry else ""))

        rec["status"] = "present" if present == len(files) else (
            "partial" if present else "absent")
        print(f"  {'':<20} -> {rec['status']} ({present}/{len(files)} files)")
        print()

    save_manifest(man)
    print(f"  wrote {manifest_path()}")

    if blocked:
        print(f"\n  BLOCKED (no source exists): {', '.join(blocked)}")
    if missing_required:
        print(f"  MISSING or GATED: {len(missing_required)} required file(s)")
        for m in missing_required:
            print(f"    - {m}")
        if args.strict:
            return 1
    elif args.verify:
        print("\n  all required files present")
    return 0


if __name__ == "__main__":
    sys.exit(main())