# Datasets

`synthetic_scenes/pick_place_floor_table/` -- built by
`scene_generator/pick_place_benchmark.py` (`generate_benchmark`, skips what is
already on disk). The module docstring explains every condition.

```
capture/                       Phase 1 scan: 35 hemisphere views
    images.npy (V,H,W,3) uint8 depths.npy (V,H,W) float32 m (stereo-like noise)
    obj_masks.npy (V,H,W) uint8 instance ids   cameras.json (OpenCV c2w, K)
    fused_pointcloud.ply (DBSCAN-filtered seeds, balanced per object)
    seed_obj_ids.npy (evaluation only)         capture_metadata.json
<condition>_s<seed>/           Phase 3: 60 frames at 5 Hz, 889x500 static camera
    rgb/frame_XXXX.png   depth_mm/frame_XXXX.png (uint16 mm, NOISE-FREE)   masks/frame_XXXX.png
    tracking_metadata.json (condition, trajectory parameters, camera, per frame:
                            phase, rotation / yaw / tilt, occlusion fraction per
                            moving object, gt_deltas for every object)
    preview.mp4
<condition>_s<seed>_30hz/      the same trajectory at 30 Hz (355 frames) for the
                               real-time protocol; read lazily (LazyFrame)
```

Both `capture_metadata.json` and every `tracking_metadata.json` record
`scene_version`; `generate_benchmark` re-renders anything from an older scene.

Conditions (12 x 3 seeds = 36 sequences): `base`, `yaw15`,
`yaw90`, `tilt30`, `tilt60`, `combined45`, `occ_none`, `occ_hand`, `occ_full`,
`obj_mustard`, `obj_cereal`, `multi`. Speed (frame stride 1/2/4) and depth noise
(`clean` / `stereo` 3 mm / `noisy` 10 mm + 5% holes) are applied at load time
(`load_sequence(..., speed=, depth=)`), deterministically.

Instance ids: 0 floor, 1 mug, 2 mustard bottle, 3 soup can, 4 apple, 5 cereal box,
9 staging table, 97 hand, 98 second gripper, 99 robot gripper.
