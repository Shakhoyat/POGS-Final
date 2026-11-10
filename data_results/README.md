# Results

```
benchmark_runs/<case>/<method>/   shared run cache (every experiment reads/writes here)
    telemetry.json                per frame: timings, loss, per-object errors, pred / GT poses
    part_deltas.npy               (F, K, 7) optimised pose deltas
    row.json                      the summary row of this run
v1_base_pogs/  v2_kalman/  v3_objects/
    summary.md                    one table per benchmark factor (levels x methods,
                                  mean ± std over seeds) + overall means; always
                                  includes the "Static (no motion)" baseline
    summary.csv                   one row per (case, method)
    videos/                       dashboards for seed 0 (experiments/visuals.py)
v1_base_pogs/ablations/           paper ablations (No Depth, No DINO) on `base`
```

A case is `<condition>_s<seed>__<speed>x_<depth>`. A method name is a cache key:
the same name must always mean the same settings.
