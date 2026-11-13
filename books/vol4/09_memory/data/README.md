# Data Provenance (`09_memory`)

## `kitti_odometry_leaderboard.csv`

- **Source**: Geiger, Lenz, and Urtasun (2012), KITTI Vision Benchmark Suite — Visual Odometry / SLAM Evaluation 2012 (`@geiger2012are`), <https://www.cvlibs.net/datasets/kitti/eval_odometry.php>.
- **Epistemic category**: `measured` (public benchmark evaluation leaderboard over hidden test sequences 11–21; `runtime_s` is self-reported by submitting authors across unstandardized compute hardware).
- **Schema**: `rank`, `method`, `sensor_group`, `loop_closure`, `translation_err_pct`, `rotation_err_deg_per_m`, `runtime_s` across all 127 ranked submissions as of verification.
- **Sensor icon semantics**: `sensor_group` records the upstream KITTI table icons (`stereo` for `icon_st.png`, `laser` for `icon_la.png`, `stereo+laser` when both appear, and `no_icon(...)` when neither icon is set). On the upstream leaderboard, `laser` includes hybrid visual-lidar odometry submissions such as `V-LOAM`, `SDV-LOAM`, `LIMO`, and `DEMO`. Rank 127 (`SLL`, `laser`) is a failure-mode submission at 90.05 percent translational drift.
