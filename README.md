# applied-ai-case-study

Offline spatial reconstruction pipeline. **Implemented: LiDAR/RGB-D tier (single room).** Photo, video, damage, drift correction,
multi-room stitching, head-to-head and fix loop are NOT implemented (see `docs/compliance_matrix.md`).

## Run (Windows PowerShell)
    scripts\setup.ps1                                   # venv, install, tests
    .\.venv\Scripts\Activate.ps1
    python -m applied_ai.cli inspect --input benchmark/single_scan_with_ceiling
    python -m applied_ai.cli run --input benchmark/single_scan_with_ceiling --tier lidar --config configs/lidar.yaml
    python -m applied_ai.cli validate --scene outputs/single_scan_with_ceiling/scene.json
    python -m applied_ai.cli evaluate --scene outputs/<scan>/scene.json --ground-truth gt.yaml --calibrate   # see docs/ground_truth_format.md

Outputs go to `outputs/<scan>/`: scene.json (internal schema 0.1, NOT the official one), plan.png/svg, pointcloud.ply, diagnostics.json,
convention_report.json, debug/. Raw `benchmark/` data is read-only and git-ignored.

Pose direction, camera axes and depth scale are selected empirically on every run (docs/assumptions.md); the run fails loudly if the
evidence is poor or ambiguous. Confidence intervals are **uncalibrated priors** until you run `evaluate --calibrate` on real ground truth
and set `uncertainty.calibration_file`.
