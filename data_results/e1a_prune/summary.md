# E1a: pruning over-covering Gaussians from the drill group

Pre-registered in `docs/note/e1a_overcoverage_prune_preregistration.md`. Static (no motion) not applicable. Rules first.

| Rule | Result |
|---|---|
| PR1 pruned abs(argmin s=1) ≤ 3° on ≥ 2/3 | FAIL |
| PR2 control argmin s=1 ≤ −5° on ≥ 2/3 | pass |
| PR3 pruned abs(shift s 0→1) < 2° on ≥ 2/3 | FAIL |

| Seed | group / outside | variant | argmin s=0 | argmin s=1 | mask outside GT px | halo outside GT px | GT missed px |
|---|---|---|---|---|---|---|---|
| 0 | 12340 / 175 | original | -2.5 | -10.5 | 887 | 1948 | 98 |
| 0 | 12340 / 175 | pruned | -2.0 | -7.5 | 611 | 1538 | 102 |
| 0 | 12340 / 175 | control | -2.5 | -8.5 | 873 | 1952 | 100 |
| 1 | 12515 / 158 | original | -1.5 | -10.5 | 1096 | 1688 | 89 |
| 1 | 12515 / 158 | pruned | -1.0 | -10.5 | 758 | 1515 | 94 |
| 1 | 12515 / 158 | control | -1.5 | -10.5 | 1094 | 1683 | 90 |
| 2 | 12262 / 110 | original | +0.5 | -4.5 | 728 | 1591 | 84 |
| 2 | 12262 / 110 | pruned | -2.0 | -7.5 | 506 | 1380 | 90 |
| 2 | 12262 / 110 | control | -0.5 | -4.5 | 728 | 1591 | 86 |
