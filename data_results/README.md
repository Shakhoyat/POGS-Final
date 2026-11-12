# Results

```
benchmark_runs/<case>/<method>/   shared run cache (every experiment reads/writes here)
    telemetry.json                per frame: timings, loss, per-object errors, pred / GT poses
    part_deltas.npy               (F, K, 7) optimised pose deltas
    row.json                      the summary row of this run (with scene_version)
    schedule.json                 real-time runs: frames tracked, start / finish
                                  times, which estimate each camera frame saw
v1_base_pogs/  v2_kalman/  v3_objects/
    summary.md                    one table per benchmark factor (levels x methods,
                                  mean ± std over seeds) + overall means; always
                                  includes the "Static (no motion)" baseline
    summary.csv                   one row per (case, method)
    videos/                       dashboards for seed 0 (experiments/visuals.py)
v1_base_pogs/ablations/           paper ablations (No Depth, No DINO) on `base`
realtime/                         real-time protocol (summary.md per playback speed)
scene_preview/                    contact sheet of the scene + an example dashboard frame
```

A case is `<condition>_s<seed>__<speed>x_<depth>`; real-time cases are
`<condition>_s<seed>_30hz__rt<speed>x_<depth>` (speed = playback speed).
Results produced before scene v2 (everything currently in this folder except
`scene_preview/`) are from the old scene and are replaced on the next run. A method name is a cache key:
the same name must always mean the same settings.
