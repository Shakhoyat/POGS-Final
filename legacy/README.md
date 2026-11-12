# Legacy modules (not maintained)

These files come from the original exploratory notebook port. No experiment
imports them, and they no longer import cleanly against the current `common/`
package (e.g. `CaptureConfig` was removed). They are kept for reference only.

| File | What it was |
|---|---|
| `simscene.py` | First analytic scene simulator (superseded by `scene_generator/`) |
| `grasp.py` | Occluded grasp sampling (Sec. IV-E) -- out of scope for this project |
| `servoing.py` | Tool-servoing simulation (Sec. V-B) -- out of scope |
| `datasets.py` | Dataset helpers for `simscene.py` |
| `viz.py`, `generate_visuals.py` | Plotting helpers for the notebook |
| `scene_generator/trajectory_planner.py`, `tracking_generator.py`, `video_annotator.py`, `generate_scene.py` | Old scenario generator built around a "30 deg angle constraint" (respected / threshold / violated). That framing misreads the paper -- 30 deg is the in-gripper tool perturbation of Table II, not a trajectory limit. Replaced by `scene_generator/pick_place_benchmark.py`. |
