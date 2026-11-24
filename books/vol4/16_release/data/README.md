# Data Provenance (`16_release`)

## `waymo_safety_impact_headline.csv`

- **Source**: Waymo LLC (2026), *Waymo Safety Impact Data Hub* (`@waymo2026safetyimpact`), <https://waymo.com/safety/impact/> (through the `202606` v2 release covering 271.3 million rider-only miles).
- **Epistemic category**: `measured` (operator-reported rider-only crash counts and incidence per million miles across seven cumulative data cuts from `202409` to `202606`, paired with Waymo's location-adjusted 2022 human crash benchmark).
- **Schema**: `data_cut`, `ro_miles_millions`, `outcome`, `waymo_count`, `waymo_ipmm`, `benchmark_ipmm`, `pct_reduction`, `pct_reduction_ci_lo`, `pct_reduction_ci_hi`.
- **Accounting boundary**: Filters `All Locations (mileage blended)` and `All Crashes` against the dynamic (location-adjusted) benchmark (`2022benchmark_v2.csv` for `202606`). Between `202506` and `202509`, Waymo renamed `Airbag Deployment` to `Any Airbag Deployment` when introducing `Ego Vehicle Airbag Deployment`. In `202606`, the row labeled `Any Fatal` (`pct_reduction = -0.7983`, `benchmark_ipmm = 0.03655`) transcribes `Any Fatal (Dynamic)` from `2022benchmark_v2.csv`, whereas Waymo's static non-dynamic `Any Fatal` benchmark is `-0.7305` (`benchmark_ipmm = 0.02736`, reported alongside dynamic outcomes in Volume II's static-plus-dynamic summary table).
