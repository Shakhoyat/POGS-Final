"""Shared configuration for the POGS model-substitution study.

One place for every path, resolution and model fact the row scripts and the
report both need, so a row cannot drift from what the report claims.

Resolutions are not chosen -- they are what POGS actually runs at:

* DETIC   ``Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml`` -> 896
* DINOv2  ``MAX_DINO_SIZE = 1260`` (pogs/data/utils/dino_dataloader.py),
           patch stride 14, layer 11, facet "key"
* CLIP    input divided by ``clip_downscale_factor = 2``
           (pogs/data/full_images_datamanager.py:78)

Benchmarking at anything else produces a number that does not predict POGS's
runtime, which is the only reason this table exists.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_ROOT = os.path.dirname(HERE)
INPUTS = os.path.join(STUDY_ROOT, "inputs")
DEPS = os.path.join(STUDY_ROOT, "deps")

REAL_14 = os.path.join(INPUTS, "real_14")

# --------------------------------------------------------------------------- #
# the scene matrix
# --------------------------------------------------------------------------- #
#: Pendulum swing angles, in degrees. The independent variable of the sweep: the
#: cylinder's peak displacement grows with the angle, so this is a difficulty
#: ladder rather than nine near-identical scenes. 5 deg is barely moving, 60 deg
#: is a wide arc.
PENDULUM_ANGLES = (5, 10, 15, 20, 25, 30, 40, 50, 60)


def _pendulum(a: int) -> str:
    return os.path.join(INPUTS, f"pendulum_{a:02d}")


PICKPLACE = os.path.join(INPUTS, "pickplace")
"""Primary study input: a mug lifted and carried across a table by a gripper.

This is the scene the model comparison runs on. It is a real COCO/LVIS class, it
fills ~2.3% of frame rather than 0.9%, the frame is a room rather than mostly
black, and the gripper occludes the mug -- so DETIC is being asked something it
is actually trained to do.
"""

PENDULUM = _pendulum(30)
"""One rung of the pendulum ladder, named for backward compatibility.

Its ground truth is exact (centroids match closed form to 3e-8 m, pivot drift
exactly 0) and its seed cloud is byte-identical between runs, so it is a genuinely
deterministic pipeline test. It is a poor *perception* test: an unnamed,
textureless, 0.96%-of-frame cylinder against a dark scene, against which DETIC's
default LVIS vocabulary has nothing to match.
"""

#: Every scene a row trains and tracks on, in report order. The pendulum ladder
#: first because it is the cheap, exact, deterministic end of the range;
#: pickplace last because it is the hard one and the one worth reading.
SCENES_V1: tuple[str, ...] = tuple(
    os.path.basename(_pendulum(a)) for a in PENDULUM_ANGLES
) + ("pickplace",)

#: The v2 matrix: the same ten scenes regenerated with a thicker pendulum (it
#: spans ~4% of the frame rather than ~1%, several DINO patches instead of one),
#: object-local surface texture, a textured background, and camera noise. Built
#: because the v1 pendulum was narrower than one DINOv2 patch on a black
#: background, which left the DINO-only tracker nothing to match. Geometry,
#: motion and ground truth are identical to v1.
SCENES_V2: tuple[str, ...] = tuple(
    f"pendulum_v2_{a:02d}" for a in PENDULUM_ANGLES
) + ("pickplace_v2",)

#: One scene: the pickplace sequence with a small gripper that holds only the mug's
#: handle, so the mug body is never occluded. Same room, mug, trajectory and
#: ground truth as `pickplace`; only the gripper differs.
SCENES_HANDLE: tuple[str, ...] = ("pickplace_handle",)

#: Null control: a pendulum swinging 1 degree (about 2.6 mm). Any tracker drift
#: beyond a few mm here is not following the object, so it measures the harness
#: and the tracker's resting behaviour rather than motion tracking.
SCENES_CONTROL = ("pendulum_v2_01",)

#: Which matrix `sweep`, `report` and `compare` operate on:
#: SCENE_SET=v2 bash run.sh sweep      (or SCENE_SET=handle)
_SCENE_SETS = {"v1": SCENES_V1, "v2": SCENES_V2, "handle": SCENES_HANDLE, "control": SCENES_CONTROL}
SCENE_SET = os.environ.get("SCENE_SET", "v1")
if SCENE_SET not in _SCENE_SETS:
    raise SystemExit(f"SCENE_SET must be one of {', '.join(_SCENE_SETS)}, got {SCENE_SET!r}")
SCENES: tuple[str, ...] = _SCENE_SETS[SCENE_SET]

#: Appended to report file names so a v2 run never overwrites the v1 reports.
RESULTS_SUFFIX = "" if SCENE_SET == "v1" else f"_{SCENE_SET}"

#: Scenes that also carry a tracking sequence. All ten do, but recorded as its
#: own list so a future capture-only scene degrades to a warning rather than a
#: crash inside the tracker.
SCENES_WITH_TRACKING: tuple[str, ...] = SCENES


def primary_scene() -> str:
    """The scene whose figures lead a report, for the active ``SCENE_SET``.

    The pickplace-family scene: it is the hard case, the one the study exists
    for, and the only one at 1280x720, which matters because latency scales with
    the input POGS feeds the models. Each set has exactly one -- ``pickplace``,
    ``pickplace_v2``, ``pickplace_handle`` -- so this is a lookup, not a choice.

    Derived rather than written down because `compare.py` previously hardcoded
    the literal ``"pickplace"``. That name is absent from every set but v1, so
    outside v1 the lead silently fell through to the first pendulum rung, which
    carries no `bench/`, and the v2 and handle reports printed "no row has been
    benchmarked yet" while row 02's latency sat in the artefacts unread.
    """
    for scene in SCENES:
        if scene.startswith("pickplace"):
            return scene
    return SCENES[0]


def scene_resolution(scene_dir: str) -> tuple[int, int]:
    """Capture resolution of a scene, read from its own transforms.json.

    Read rather than hardcoded because the two scene families differ (640x480
    pendulum, 1280x720 pickplace) and POGS feeds DINO and CLIP resized copies of
    these images. A benchmark taken at the wrong input size is a measurement of
    the wrong scene.
    """
    import json

    p = os.path.join(scene_dir, "transforms.json")
    try:
        with open(p) as fh:
            frames = json.load(fh)["frames"]
        if frames:
            return int(frames[0]["w"]), int(frames[0]["h"])
    except (OSError, KeyError, json.JSONDecodeError):
        pass
    return IMAGE_W, IMAGE_H


def scene_path(name: str) -> str:
    """Absolute path to a scene by name, validated against the matrix.

    The check exists because a typo'd scene name otherwise produces a path that
    does not exist and a failure deep inside the renderer, which reads as a POGS
    bug rather than a wrong argument.
    """
    if name not in SCENES_V1 + SCENES_V2 + SCENES_HANDLE + SCENES_CONTROL:
        raise SystemExit(
            f"unknown scene: {name}\n  known scenes: {', '.join(SCENES_V1 + SCENES_V2 + SCENES_HANDLE + SCENES_CONTROL)}"
        )
    return os.path.join(INPUTS, name)


def detector_iou_paths(row_root: str) -> tuple[str, ...]:
    """Candidate locations for a row's real-capture detector IoU, best first.

    Both readers resolve through this one list on purpose. An earlier version
    left the path written out in `compare.py` and `report.py` separately and
    they drifted from the writer: `run.sh validate` calls
    ``detect_check.py --real``, which writes ``real_14/detector_real_iou.json``,
    but both readers looked only at ``validate/`` and ``real_14/validate/``.
    Row 00 therefore reported "not run" for a measured mean IoU of 0.6169.

    The two later entries are the older ``validate_detector.py`` layout, kept so
    an older run stays readable rather than silently reporting nothing.
    """
    return (
        os.path.join(row_root, "real_14", "detector_real_iou.json"),
        os.path.join(row_root, "real_14", "validate", "detector_real_iou.json"),
        os.path.join(row_root, "validate", "detector_real_iou.json"),
    )


def load_detector_iou(row_root: str) -> dict:
    """A row's real-capture detector IoU artefact, or ``{}`` if never measured.

    Prefers a file that actually carries a ``summary``. ``validate/`` holds a
    committed stub (``"status": "not run -- DETIC inference pending"``,
    ``"summary": null``) that parses as perfectly valid JSON, so a plain
    first-match-wins search returns the stub and hides a real result sitting in
    the canonical slot. Falling back to the first file that merely *loads* keeps
    the stub's own ``status`` string reachable for a row that genuinely has no
    measurement.
    """
    import json

    fallback: dict = {}
    for path in detector_iou_paths(row_root):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("summary"):
            return data
        if not fallback:
            fallback = data
    return fallback


#: Every model's weights land here, one subdirectory per model. Kept outside
#: ``artefacts/`` on purpose: artefacts are measurements and get regenerated,
#: weights are 6-10 GB of downloads that should never be deleted by a re-run.
WEIGHTS = os.path.join(STUDY_ROOT, "weights")

# Where row artefacts land. Each row owns a subdirectory so a row can be
# re-run without destroying another row's evidence.
ARTEFACTS = os.path.join(STUDY_ROOT, "artefacts")

BASELINE_ROW = "00_baseline_detic_dinov2_clip"

# --------------------------------------------------------------------------- #
# what POGS actually uses
# --------------------------------------------------------------------------- #
#: nerfstudio's MachineConfig.seed default. POGS seeds nothing of its own --
#: ``dino_feats`` init, ``random_quat_tensor``, random CLIP tile scale and the
#: ``randperm`` calls in the datamanager are all unseeded -- so the only handle
#: on run-to-run variance is nerfstudio's global seed.
POGS_SEED = 42

#: Optimizer iterations per tracking frame. ``track_main_demo.py`` uses
#: ``n_opt_iters = 15``; held fixed across rows because the tracking error is a
#: function of it, and a faster row could otherwise buy error back with compute.
POGS_TRACK_OPT_ITERS = 15

#: Where POGS switches each loss on. Recorded in the report because a row whose
#: dino loss behaves differently may simply be under-trained at 4000 iters.
DETIC_DINO_STEP = 100
DETIC_LERF_STEP = 2500
POGS_TRAIN_ITERS = 4000

#: Camera resolution of the primary scene. Read from its transforms.json rather
#: than hardcoded, because POGS feeds DINO and CLIP resized copies of these
#: images and benchmarking at the wrong input size produces a latency number for
#: a scene nobody is tracking. ``pickplace`` is 1280x720; ``pendulum_30`` is
#: 640x480, and benchmarking the latter while reporting on the former would make
#: every DINO figure wrong by roughly the pixel ratio.
def _capture_resolution() -> tuple[int, int]:
    import json

    for scene in (PICKPLACE, PENDULUM):
        p = os.path.join(scene, "transforms.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as fh:
                frames = json.load(fh)["frames"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        if frames:
            return int(frames[0]["w"]), int(frames[0]["h"])
    return 640, 480  # pendulum fallback, and the original hardcoded value


IMAGE_W, IMAGE_H = _capture_resolution()
CLIP_DOWNSCLALE = 2

DETIC_RES = 896
DINO_MAX_SIZE = 1260
DINO_STRIDE = 14


def dino_resolution(w: int = IMAGE_W, h: int = IMAGE_H,
                    p: int = DINO_STRIDE) -> tuple[int, int]:
    """Resolution ``get_img_resolution`` feeds DINOv2, keeping stride ``p``.

    ``p`` is a parameter because DINOv3 is stride 16, not 14. Using the same
    function for both would compare models at different token counts, which makes
    the latency comparison meaningless rather than merely imprecise.
    """
    if w >= h:
        new_w = DINO_MAX_SIZE
        new_h = (int((h / w) * DINO_MAX_SIZE) // p) * p
    else:
        new_h = DINO_MAX_SIZE
        new_w = (int((w / h) * DINO_MAX_SIZE) // p) * p
    return new_w, new_h


def clip_resolution(w: int = IMAGE_W, h: int = IMAGE_H) -> tuple[int, int]:
    return w // CLIP_DOWNSCLALE, h // CLIP_DOWNSCLALE


# --------------------------------------------------------------------------- #
# model registry
# --------------------------------------------------------------------------- #
# ``stage``    which column of the study table the model fills
# ``pogs_uses`` whether stock POGS runs this exact model
# ``needs``    the distribution that must be importable for it to load
# ``open_vocab`` whether it can accept an arbitrary text vocabulary -- the
#              capability POGS depends on and the one the closed-set
#              replacements cannot provide
MODELS: dict[str, dict] = {
    "dino_vits14": {
        "stage": "features",
        "pogs_uses": True,
        "loader": "torch.hub",
        "spec": "facebookresearch/dinov2:dinov2_vits14",
        "open_vocab": False,
        "note": "ViT-S/14, layer 11, facet 'key' -- stock POGS feature extractor",
    },
    "dino_vitl14": {
        "stage": "features",
        "pogs_uses": False,
        "loader": "torch.hub",
        "spec": "facebookresearch/dinov2:dinov2_vitl14",
        "open_vocab": False,
    },
    "dino_txt": {
        "stage": "features+alignment",
        "pogs_uses": False,
        "loader": "torch.hub",
        "spec": (
            "facebookresearch/dinov2:"
            "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l"
        ),
        "open_vocab": True,
        "note": (
            "single model doing dense features AND text alignment, so it can "
            "displace both DINOv2 and CLIP. ViT-L/14, not POGS's ViT-S/14"
        ),
    },
    "clip_vitb16": {
        "stage": "alignment",
        "pogs_uses": True,
        "loader": "open_clip",
        "spec": "ViT-B-16 / laion2b_s34b_b88k",
        "open_vocab": True,
        "note": "stock POGS relevancy encoder",
    },
    "siglip2": {
        "stage": "alignment",
        # Row 03 substitutes it into POGS as the relevancy encoder via
        # pogs/encoders/siglip2_encoder.py (patch 0001). Previously it was
        # bench-only, so pogs_uses was False.
        "pogs_uses": True,
        "loader": "transformers",
        "spec": "google/siglip2-base-patch16-224",
        "open_vocab": True,
    },
    "yolo": {
        "stage": "detection",
        "pogs_uses": False,
        "loader": "ultralytics",
        "spec": "yolo11x",
        "open_vocab": False,
        "masks": False,
        "note": (
            "boxes only, COCO classes. Cannot serve POGS's instance_masks "
            "contract and has no text interface."
        ),
    },
    "rfdetr": {
        "stage": "detection",
        "pogs_uses": False,
        "loader": "rfdetr",
        "spec": "RFDETRBase",
        "open_vocab": False,
        "masks": True,
        "note": (
            "RF-DETR-Seg gives masks but COCO classes only; POGS's "
            "custom_vocab path needs open-vocabulary."
        ),
    },
    "detic": {
        "stage": "detection",
        "pogs_uses": True,
        "loader": "pogs",
        "spec": "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size",
        "open_vocab": True,
        "masks": True,
        # DETIC needs detectron2 plus the Detic submodule and its CenterNet2
        # third_party path, so bench.py cannot construct it outside the POGS env.
        # Stated in the report rather than left as a silent gap in the table.
        "latency_measurable": False,
        "note": "stock POGS detector; needs detectron2 + the Detic submodule",
    },
    "locateanything3d": {
        "stage": "detection",
        "pogs_uses": False,
        "loader": "transformers",
        "spec": "nvidia/LocateAnything-3B",
        "open_vocab": True,
        "masks": False,
        "params_b": 3.0,
        "note": (
            "Eagle-2.5-based VLM emitting 3D boxes via Chain-of-Sight decoding. "
            "Weights are public (nvidia/LocateAnything-3B, 2 shards, ~4.7 GB "
            "each) but the output is a 3D box, not a per-instance 2D mask, so it "
            "cannot serve POGS's instance_masks contract without an adapter. "
            "Capability gap, documented rather than bridged."
        ),
    },
    "dinov3": {
        "stage": "features",
        "pogs_uses": False,
        "loader": "transformers",
        "spec": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "open_vocab": False,
        "gated": True,
        "stride": 16,
        "note": (
            "HuggingFace licence acceptance required (HTTP 401 anonymously). "
            "Stride 16, so POGS's stride-14 resolution math does not apply and "
            "the feature grid changes shape."
        ),
    },
    "mobileclipv2": {
        "stage": "alignment",
        "pogs_uses": False,
        "loader": "open_clip",
        "spec": "MobileCLIP2-B-OpenCLIP",
        "open_vocab": True,
        "note": (
            "timm/MobileCLIP2-B-OpenCLIP ships open_clip_config.json, so it "
            "loads through the same open_clip.create_model_and_transforms call "
            "POGS already makes for CLIP. Drop-in for the alignment slot."
        ),
    },
    "vfmdetr": {
        "stage": "detection+features",
        "pogs_uses": False,
        "loader": None,
        "spec": None,
        "open_vocab": None,
        "masks": None,
        "blocked": True,
        "note": (
            "No weights located: zero HuggingFace search hits, no repository, "
            "no NeurIPS 2025 page under this name. Nearest matches are "
            "VFM-Det (arXiv 2408.13031, vehicle detection, 2024) and VFMTok "
            "(NeurIPS 2025, a tokenizer). Row 03 reports blocked."
        ),
    },
}

#: Models whose latency can be measured without the POGS environment.
STANDALONE = [k for k, v in MODELS.items() if v["loader"] != "pogs"]

#: Rows of the study table. Filled in as each row is built.
ROWS: dict[str, dict] = {
    BASELINE_ROW: {
        "detection": "detic",
        "features": "dino_vits14",
        "alignment": "clip_vitb16",
        "status": "ready",
        # Row 00 runs stock POGS untouched. Everything else needs at least the
        # dino_model_type swap or a transformers version bump, so its tree is
        # patched and the diff is recorded in patch/.
        "patched": False,
    },
    "01_locateanything_dinotxt": {
        "detection": "locateanything3d",
        "features": "dino_txt",
        "alignment": "dino_txt",
        "status": "patched -- capability gap (3D boxes, not 2D masks)",
        "patched": True,
        "pipeline_note": (
            "In `train` and `track` this row runs the SAME models as row 00 (DETIC, "
            "DINOv2 ViT-S/14, OpenCLIP): LocateAnything3D gives 2D boxes, not the "
            "instance masks POGS consumes, and nothing selects dino.txt in the feature "
            "extractor. The tracking and training numbers below are therefore a rerun of "
            "the baseline pipeline (they show run-to-run variation), NOT results for "
            "LocateAnything3D or dino.txt. Those two models were measured separately: "
            "see `bench/latency.json` (dino.txt) and `bench/locateanything_detect.json` "
            "and `LOCATEANYTHING_DETECT.md` (LocateAnything3D)."
        ),
        "patch_reason": (
            "dino.txt is open_clip-shaped, not DINOv2, so dino_model_type "
            "cannot express it; LocateAnything3D replaces the detector "
            "dataloader entirely. Both hardcoded in stock POGS."
        ),
    },
    "02_rfdetr_dinov3_mobileclip": {
        "detection": "rfdetr",
        "features": "dinov3",
        "alignment": "mobileclipv2",
        "status": "patched -- gated on DINOv3 licence",
        "patched": True,
        "pipeline_note": (
            "In `train` and `track` this row runs the SAME models as row 00 (DETIC, "
            "DINOv2 ViT-S/14, OpenCLIP ViT-B/16): RF-DETR is a closed-set COCO detector "
            "and cannot take the text vocabulary POGS passes to its detector, DINOv3 "
            "(stride 16) and MobileCLIP2 are not selected by anything in the run. The "
            "tracking and training numbers below are therefore a rerun of the baseline "
            "pipeline (they show run-to-run variation), NOT results for RF-DETR, DINOv3 "
            "or MobileCLIP2. Those models are measured separately in `bench/latency.json`."
        ),
        "patch_reason": (
            "DINOv3 needs transformers > 4.44 and POGS pins ==4.44.0; "
            "stride 16 breaks the stride-14 resolution math."
        ),
    },
    "03_vfmdetr_siglip2": {
        # VFM-DETR is blocked (repo found at lch216/VFMDETR, but no released
        # weights anywhere and a closed-set few-shot protocol that cannot feed
        # POGS's open-vocab instance_masks contract). DETIC stays as the
        # detector; SigLIP2 substitutes only the relevancy/alignment encoder.
        "detection": "detic",
        "features": "dino_vits14",
        "alignment": "siglip2",
        "status": "partially unblocked -- DETIC + SigLIP2; VFM-DETR still blocked",
        "patched": True,
        "pipeline_note": (
            "This row really substitutes one component in `train` and `track`: the "
            "relevancy/alignment encoder is SigLIP2-base (768-d) instead of OpenCLIP "
            "ViT-B/16 (512-d). The detector is still DETIC and the features are still "
            "DINOv2 ViT-S/14, as in row 00, because VFM-DETR (which would replace both) "
            "has no released weights. Differences from row 00 therefore reflect the "
            "alignment encoder plus run-to-run variation, not a different detector."
        ),
        "patch_reason": (
            "Stock POGS hardcodes OpenCLIP ViT-B-16 512-d as the relevancy "
            "encoder and hardcodes the GaussianField CLIP head to 512. "
            "SigLIP2-base is 768-d, so both the encoder selection and the "
            "head width are env-gated (POGS_ALIGNMENT_ENCODER=siglip2)."
        ),
        "available_models": ["siglip2", "detic", "dino_vits14"],
    },
    "04_yolo": {
        "detection": "yolo",
        "features": "dino_vits14",
        "alignment": "clip_vitb16",
        "status": "latency only -- closed-set, capability gap recorded",
        "patched": False,
        "available_models": ["yolo", "dino_vits14", "clip_vitb16"],
    },
}


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
# ``run.sh`` needs the scene list from Python without importing torch, numpy or
# anything else heavy. One ``python -c "import study_config"`` per sweep step
# would drag the whole dependency chain in just to print ten names, so this is
# what it calls instead.
def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list-scenes", action="store_true",
                    help="print one scene name per line and exit")
    ap.add_argument("--list-rows", action="store_true")
    args = ap.parse_args()

    if args.list_scenes:
        for s in SCENES:
            print(s)
    if args.list_rows:
        for r in ROWS:
            print(r)
    if not (args.list_scenes or args.list_rows):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())