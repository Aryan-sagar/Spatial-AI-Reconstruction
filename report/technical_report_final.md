# Technical report (LiDAR tier, single room; honest status)

**Scope.** An offline pipeline from a Stray-Scanner-style LiDAR export (depth, confidence, RGB, odometry, IMU) to a per-room plan with intervals. Photo, video, drift correction, damage, stitching and the head-to-head are NOT implemented (see `docs/compliance_matrix.md`). No pretrained models are used.

**Architecture.** `io/` loads and synchronises the export; `calibration/` chooses pose direction, camera axes and depth scale by multi-view depth consistency (not assumed); `reconstruction/` builds a voxel cloud from motion-selected keyframes; `geometry/` finds up/floor/ceiling (IMU gravity + height peaks), wall lines (Manhattan-axis histogram, snapped), then filters them (tall columns, camera-crossing, high-band support, ray-boundary pruning), trims corners, builds the footprint, detects openings; `output/` validates schema 0.1 and renders; `uncertainty.py` attaches intervals; `evaluation/` scores gates.

**Drift.** Not handled: poses are used as supplied. The scene says so (`drift: not_run`). This is a failed row.

**Error budget (room A, Measure-app ground truth, one room).** Ceiling 3.30 m vs 3.45 m (+/-5 cm for the reference itself); repeat spread 2.2 cm, so repeatable but possibly biased. Two ceiling-height peaks 22 cm apart exist; the stronger lower one is chosen. Horizontal scale looks right (top wall 3.32 m vs 3.33 m). Floor area 13.9 m2 vs 15.67 m2 (-11%), traced to the bottom boundary choice. Openings: all 3 missed after the fix.

**Calibration.** Intervals are tier priors and are uncalibrated; `evaluate --calibrate` can produce conformal quantiles, but one room is not a calibration set. Where a footprint fails its quality checks (camera inside < 50%, outside floor evidence > 50% of area, no enclosure) the area is published as `estimated`, `reliable: false`, with an evidence-bounded wide interval and a visible warning on the plan.

**Fix loop.** See `docs/fix_loop_declaration.md`: furniture-height lines displaced real walls and fitted lines overshot corners; two fixes shipped; before/after regenerable with `scripts/fix_loop.py`.

**Known failure modes.** Furnished rooms where the ray view of a wall is blocked; built-in wardrobes (which surface is "the wall"); walkthroughs of several rooms (single-room assumption; the area is flagged unreliable, segmentation is experimental); two ceiling levels; mirrors, glass, wet-look surfaces and low light (not tested, depth there is unreliable); scans that never look up (no high band, no ceiling).

**Data.** Raw benchmark data in `benchmark/` (room A: two scans, Measure-app readings). Recruiter-provided scans have no ground truth and were used for development only. All gate numbers are from room A and are reproducible from the scripts above.
