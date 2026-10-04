# Compliance matrix (honest status)

| Requirement | Implementation | File | Command | Artifact | Status | Evidence |
|---|---|---|---|---|---|---|
| Dataset inventory + sync check | inventory, synchronizer | io/inventory.py, calibration/synchronization.py | `cli inspect` | outputs/inventory/*.json | DONE | all 3 supplied scans pass (user-run) |
| LiDAR single-room scene | conventions, cloud, floor/ceiling, walls, openings, plan | reconstruction/lidar_run.py | `cli run` | scene.json, plan.png/svg, pointcloud.ply | IMPLEMENTED; synthetic-validated; **real-data accuracy unmeasured** | tests/integration |
| Ceiling / unobserved handling | prominence test, `unobserved` status | geometry/coordinate_system.py | `cli run` | scene.json | DONE (synthetic) | test_no_ceiling_is_unobserved |
| Intervals on every measurement | UncertaintyModel + schema validation | uncertainty.py, output/schema.py | `cli validate` | scene.json | DONE as **uncalibrated priors**; calibration tooling ready | needs ground truth |
| Gates / metrics | metrics + conformal calibration | evaluation/metrics.py | `cli evaluate` | gate_results.json | IMPLEMENTED; not run on real GT | tests/unit/test_metrics.py |
| Drift correction | - | - | - | - | NOT DONE | scene diagnostics say `not_run` |
| Multi-room stitching / room graph | - | - | - | - | NOT DONE | |
| Video tier | - | - | - | - | NOT DONE | |
| Photo tier | - | - | - | - | NOT DONE | |
| Damage / concealed damage / scope items | - | - | - | - | NOT DONE (empty lists in scene) | |
| Head-to-head vs consumer app | - | - | - | - | NOT DONE | needs app data |
| Fix loop | - | - | - | - | NOT DONE | needs a real failing gate |
| Published schema adapter | stub | output/schema.py | - | - | OFFICIAL SCHEMA NOT SUPPLIED | internal schema 0.1 only |
| Clean-machine run | setup/reproduce scripts | scripts/*.ps1 | see README | - | UNTESTED on a clean machine | |
| Wall closure / outline (LiDAR) | Manhattan-frame axis walls, column filter, corner closing, camera-region enclosure | geometry/walls.py, geometry/floorplan.py, geometry/room_layout.py | `cli run` (ablation: `--config configs/ablation_legacy_walls.yaml`) | diagnostics.json (`manhattan`, `footprint_enclosure`) | IMPLEMENTED; synthetic ablation only (`docs/synthetic_wall_ablation.md`); **no real-data result yet** | tests/unit/test_wall_closure.py |
