# Result tables

Small tables behind the numbers in the paper (no raw logs).

| File | Content |
|---|---|
| `calm_legs_E2.csv` | per-leg metrics, normal weather: A (seeds 7-9 = `A7..A9`), straight-cost learner (`L7..L9`), fixed-gain learner (`B7`), turn-aware learner (`LT`) |
| `strong_legs_E2d.csv` | the same plus doubled wind/waves: A (`SA`), frozen L_t (`SL`), resumed L_t+ (`SR`) |
| `wq_probe_variants_E1.csv` | map RMSE for A/BU/CU, 3 seeds x 10 fields, probe variants actual / no motion noise / no lag / ideal |
| `wq_lag_corrected.csv` | map RMSE with the raw 5-s probe and the lag-corrected probe (Tf = 1, 2, 3 s) for E1, calm A vs L_t and the doubled-wind runs |

Column definitions of the leg tables are at the top of `analysis/legs.py`
(`e_settled`, `cte`, `t_settle2`, `overshoot`, `effort`, `sat`, `tau_settled`, `lyap`, `phi`).
