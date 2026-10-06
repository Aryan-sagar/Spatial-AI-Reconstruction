# Compliance matrix (requirement -> file path -> artifact -> status)

Status vocabulary: **DONE**, **PARTIAL** (works, limits stated), **EXPERIMENTAL** (runs, validated on synthetic data only), **NOT DONE**. Nothing below is claimed beyond what was run.
Ground truth for the one real benchmark room (room A) comes from the iPhone Measure app, NOT tape or laser (development ground truth only).

| # | Requirement (brief) | File path | Artifact | Status |
|---|---|---|---|---|
| 1 | LiDAR tier: depth + poses + intrinsics -> per-room plan | `src/applied_ai/reconstruction/lidar_run.py`, `geometry/{walls,room_layout,floorplan,openings,rooms}.py` | `outputs/<scan>/scene.json`, `plan.png` | PARTIAL: single room; room A gates measured (see 17) |
| 2 | Photo tier (2-8 stills/room, no depth/poses) | - | - | NOT DONE |
| 3 | Video tier (handheld walkthrough) | - | - | NOT DONE |
| 4 | Confidence interval on every measurement | `src/applied_ai/uncertainty.py`, `output/schema.py` | `scene.json` (every measurement has `confidence_interval`) | PARTIAL: tier priors, UNCALIBRATED (one benchmark room cannot calibrate); unreliable footprints get an evidence-bounded wide interval |
| 5 | One command per capture | `src/applied_ai/cli.py` (`run`) | `python -m applied_ai.cli run --input <scan> --tier lidar` | PARTIAL: LiDAR only |
| 6 | JSON to the published schema | `src/applied_ai/output/schema.py` | internal schema 0.1 | NOT DONE: the published schema was never supplied; internal schema validated by `cli validate` |
| 7 | Rendered plan | `src/applied_ai/output/renderer.py` | `plan.png/svg`, `debug/topdown.png`, `debug/rooms.png` | DONE (LiDAR) |
| 8 | Stitched multi-room plan with adjacency | `geometry/room_segmentation.py`, `geometry/visibility.py` | `diagnostics.multiroom`, `debug/rooms.png` | EXPERIMENTAL: free-space rooms, doorways, adjacency; synthetic layouts only; not in `scene.rooms` |
| 9 | Per-surface damage regions, concealed-damage flags, scope items | - | - | NOT DONE |
| 10 | Drift accountability (loop closure / pose graph + on/off ablation) | - | scene reports `drift: not_run` | NOT DONE: poses are used as supplied (an automatic fail on this row, stated plainly) |
| 11 | Photo-tier whole-property stitch | - | - | NOT DONE |
| 12 | Capture route: one-page stock-capture protocol | `docs/capture_protocol.md` | - | PARTIAL: app named as Stray Scanner by inference from the export layout, unconfirmed |
| 13 | Device matrix | `docs/device_matrix.md` | - | PARTIAL |
| 14 | Benchmark set (3+ rooms + connector, furnished room with 2 damage classes, all three tiers, repeat capture, tape/laser GT) | `benchmark/` (raw data, git-ignored) | room A: 2 LiDAR scans + Measure-app GT | PARTIAL: one room, LiDAR tier only, no damage, GT not tape |
| 15 | Gates: openings, ceiling, repeatability | `src/applied_ai/evaluation/metrics.py`, `cli evaluate` | `gate_results.json` | PARTIAL: measured on room A only |
| 16 | Head-to-head vs a consumer scanning app | - | - | NOT DONE |
| 17 | Fix loop: declaration, shipped fix, regenerable before/after, diff | `docs/fix_loop_declaration.md`, `scripts/fix_loop.py`, `configs/fix_before.yaml`, `configs/fix_after.yaml` | `outputs/fix_loop/{before,after,comparison.json,report.md}` | PARTIAL: two fixes shipped (high-band perimeter support; corner trimming); after-numbers in the declaration |
| 18 | Repo + README, runs on a fresh capture < 15 min, clean machine | `README.md`, `scripts/setup.ps1`, `scripts/reproduce.ps1` | - | PARTIAL: Windows PowerShell only; no clean-machine test run |
| 19 | Reproduction bundle | `scripts/reproduce.ps1`, `configs/` | - | PARTIAL |
| 20 | Technical report (<= 6 pages) | `report/technical_report_final.md` | - | DONE (honest about gaps) |
| 21 | Mirrors, glass, wet-look surfaces, low light covered | - | listed under known failure modes in the report | NOT DONE: no test capture, no handling |
| 22 | Pretrained models disclosed | `docs/model_disclosures.md` | - | DONE: none used |
| 23 | Process evidence (real commit history) | `git log` | - | PARTIAL: real history, one early large commit |
