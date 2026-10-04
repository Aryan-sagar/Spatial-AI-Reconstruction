# HANDOFF — Applied AI Case Study (read this first, then the repo)

Paste this file plus the original assessment PDF text and the original build spec into the new LLM. The repo is on the USER'S machine
(Windows, PowerShell, `C:\Users\aryan\OneDrive\Desktop\applied-ai-case-study`, venv `.venv`). The previous assistant's sandbox is gone.
Ask the user to upload a fresh zip of the repo if you need to see code.

## 1. The task (assessment: "Applied AI Engineer Case Study", Aug 2026)
Build an offline, reproducible pipeline: iPhone captures -> dimensioned per-room plan (walls, ceiling height, floor area, openings), stitched
multi-room plan with adjacency, per-surface damage regions + class + metric extent, concealed-damage flags with the rule that fired, scope line
items keyed to surfaces, **a confidence interval on every measurement**, one command per capture, JSON "to the published schema" (NOT supplied -
we use our own internal schema 0.1 + adapter stub), rendered plan.
Three tiers ALL mandatory: photos (2-8 stills/room, per-room folders, must stitch whole property), video (handheld walkthrough), LiDAR (depth+poses+intrinsics).
Scoring: walk-in cold run on an unseen space vs laser measurements 30%; fix loop 25% (worst gate -> hypothesis -> shipped fix -> before/after regenerable);
verified benchmark accuracy all tiers 15%; compliance matrix 10%; head-to-head vs a consumer scanning app on 2 rooms (beat/tie >=70% dims) 10%;
capture route quality (named stock app + one-page protocol, or own iOS app) 5%; process evidence (real commit history) 5%.
Gates: openings <=2 cm on >=85% (missed AND phantom openings count as misses); ceiling <=1.5 cm per room, repeat spread <=1 cm; repeatability <=1 cm or 0.5%/wall;
drift: "poses used as-is" is an automatic fail -> need loop closure/pose graph + footprint ablation on/off; photo stitch footprint +/-8% correct adjacency no overlaps;
photo walls +/-8%, video walls +/-3% with calibrated intervals; calibration scored at every tier; confident garbage on thin input caps the score.
Benchmark must be self-built: multi-room (3+ rooms + connector), furnished room with staged damage (2 classes), same rooms at all 3 tiers, one room repeated, tape/laser ground truth.
Constraints: no cloud inference, pretrained models allowed WITH disclosure, weights by script, cover mirrors/glass/low light, <=6 page report, README to run in <15 min.
Engineering rules from the spec: never invent formats/schema fields, never hardcode dims, every assumption documented/configurable/logged/testable, never fabricate measurements
(ceiling not seen -> status "unobserved"), raw benchmark data untouched, incremental git commits, fail loudly.

## 2. Data (real, on user's machine, git-ignored in `benchmark/`)
Three scans: `single_room` (1715 frames, 37 s), `single_scan_floor_only` (5251, 115 s), `single_scan_with_ceiling` (9745, 215 s). Layout (Stray-Scanner-like export, app unconfirmed):
`depth/NNNNNN.png` (256x192 uint16, raw ~240-5400 => looks like mm), `confidence/`, `rgb.mp4` (1920x1440, ~1.3 GB), `odometry.csv` (header has leading spaces; cols timestamp,frame,x,y,z,qx,qy,qz,qw,fx,fy,cx,cy,distortion_center_x/y),
`camera_matrix.csv` (3x3), `imu.csv` (timestamp,a_x..a_z,alpha_x..z). distortion_center_* are EMPTY (NaN) in every row (harmless, optional).
All 3 scans pass inventory + sync checks (counts equal, ids contiguous, odometry span == frames/fps within ~4 ms). Odometry dt median 0.01667 s, max 0.05 s (60 Hz grid with drops; video ~45 fps).
Per-frame fx varies ~2-3% within a scan (e.g. 1581-1615); fixed camera_matrix fx sits inside that range. Per-frame vs fixed intrinsics is UNTESTED (potential ~1% length error).
NO ground truth has been collected yet. No other captures (multi-room, photo, video, damage) exist yet.

## 3. What is implemented (repo: ~30 commits, 80 unit/integration tests pass on user's machine; pytest run via `python -m pytest -q`)
`src/applied_ai/`: config.py (YAML layered on configs/default.yaml, hashed), logging_utils.py (stage timer), cli.py (commands: inspect, run, validate, evaluate),
io/ (inventory, odometry, depth, rgb, imu, dataset=ScanDataset/FrameRecord), calibration/ (synchronization, intrinsics [depth-grid scaling modes], conventions [auto-select]),
domain/models.py, reconstruction/ (poses, projection, pointcloud [numpy PLY + voxel], keyframes, pipeline, lidar_run = end-to-end), geometry/ (planes RANSAC, coordinate_system [up/floor/ceiling],
walls [2-D RANSAC lines, Manhattan snap keeping raw, corner snap, dedupe, ray-cast pruning], floorplan, openings, rooms), output/ (schema+validate, provenance, renderer), uncertainty.py, evaluation/metrics.py (gates, Hungarian matching, conformal calibration).
Tests use a synthetic ray-traced box room with exact ground truth (`tests/synth.py`) - validates LOGIC only, not real accuracy.
Docs: docs/compliance_matrix.md (honest, mostly NOT DONE), assumptions.md, device_matrix.md, capture_protocol.md (generic), model_disclosures.md (no ML models used), ground_truth_format.md; report/technical_report.md (draft); scripts/setup.ps1, reproduce.ps1, diagnose_run.py, diagnose_odometry.py.
Commands: `python -m applied_ai.cli inspect --input benchmark/<scan>` ; `run --input benchmark/<scan> --tier lidar --config configs/lidar.yaml` ; `validate --scene outputs/<scan>/scene.json` ; `evaluate --scene ... --ground-truth gt.yaml [--repeat-scene ...] [--calibrate]`.
Run outputs: outputs/<scan>/{scene.json, plan.png/svg, pointcloud.ply, diagnostics.json, convention_report.json, run.log, debug/{topdown.png, floor_plane.ply, ceiling_plane.ply, walls.ply}, up_axis_failure.json if up estimation fails}.

## 4. Key design decisions (be ready to defend)
- Conventions are NOT assumed: `calibration/conventions.py` scores {camera_to_world, world_to_camera} x {opencv, arkit axes} x {scale 1e-4, 1e-3, 1e-2} by multi-view depth consistency (back-project frame i, project into frame j, compare depth). Fails loudly if best error > 8% or margin < 1.5x. Scale only marked "empirically_calibrated" if the score is sensitive to +/-10% scale.
- REAL RESULT: camera_to_world + **opencv** axes + scale 0.001 (error 1.4%, 24x better than runner-up) on single_room. Depth is millimetres (consistent with poses in metres).
- Up direction: trajectory PCA (least-variance axis of camera positions) as hint; SIGN from IMU gravity (assumes iOS convention: accel reading points toward gravity; fixed iPhone landscape-right device->camera mapping `planes.imu_device_to_camera`; falls back to a 24-mapping search; flags mirror ambiguity as unreliable). Camera-image-up average was wrong (98 deg off) because the sensor image is rotated vs a portrait-held phone. Real result: IMU prior resultant 0.997, IMU-vs-trajectory 1.2 deg, camera 1.40 m above floor.
- Floor = lowest significant height peak below cameras; ceiling = strongest prominent peak above cameras; ceiling not seen -> `unobserved` (value null, no CI). single_room: ceiling unobserved (expected for that scan).
- Walls in floor-plane coords from occupied cells in a height band; raw and regularized geometry both kept; boundary pruning by rays from camera path.
- Intervals = tier prior (abs + rel*value) + z*fit sigma, inflated at low support; **currently UNCALIBRATED PRIORS** (flagged in scene). `evaluate --calibrate` makes conformal quantiles -> set `uncertainty.calibration_file`.
- Raw benchmark never modified; outputs separate; provenance (input fingerprint, config hash, git commit) in scene.json.

## 5. Current state of the real run (single_room), as of last user output
After the up-axis fix: floor area 11.13 m2, ceiling unobserved, 4 walls (3.61, 6.07, 2.80, 3.71 m), 3 openings (door 0.56, door 1.07, unknown 0.51). **The plan.png is BAD**: walls disconnected, ragged floor polygon, one wall (y~-3.9) outside floor evidence, area probably underestimated. Likely causes: patchy floor evidence, furniture/door-frame/other-room lines, single-view pruning (since improved to multi-origin ray pruning - UNTESTED on real data). Wall 3.8 m raw -> 3.61 m after corner snap (check vs tape).
Last user error: Windows `OSError [Errno 22]` writing outputs\single_room\plan.png (file locked by viewer / OneDrive); fixed with timestamped-file fallback in renderer (in latest update.zip, not yet confirmed by user). Advise moving repo out of OneDrive.
History of bugs found on REAL data (all fixed): optional distortion columns all-NaN dropped every row; sync report crashed on None; up-axis from camera-up was wrong (ceiling 7.8 m); plan file lock.

## 6. NOT DONE (be honest in the report/compliance matrix)
Photo tier, video tier, damage detection/concealed damage/scope items, drift correction (+ on/off ablation; scene says drift "not_run"), multi-room stitching / room graph / adjacency, head-to-head vs consumer app, fix loop (configs/fix_before.yaml, fix_after.yaml not created), benchmark runner command `benchmark`, `compare`, `reproduce`, `fix-loop` CLI commands, published-schema adapter (schema never supplied - ask organisers or document), clean-machine test, device matrix with real device, named capture app + one-page protocol, handling of mirrors/glass/low light section, ground truth + real gate numbers, report (<=6 pages).

## 7. Recommended next steps (priority order)
1. User re-runs single_room with latest code; send `debug/topdown.png` + `plan.png` + real room dims (width, length, #doors/windows, whether other rooms were scanned). Fix wall closure/outline (ideas: free-space carving from camera rays; rectilinear polygon from Manhattan-aligned occupancy; extend walls to perpendicular neighbours; restrict to height band excluding door-frame heights).
2. Run `single_scan_with_ceiling`; tape-measure ceiling/walls/doors for the rooms scanned; write gt.yaml (docs/ground_truth_format.md); run `evaluate --calibrate`; record real gate numbers.
3. Pick the worst real gate -> do the FIX LOOP (25%): create configs/fix_before.yaml/fix_after.yaml, `fix-loop` command, outputs/fix_loop/{before,after,comparison.json,report.md}, regenerable from final repo, with a stated prediction.
4. Name the capture app (probably the Stray-Scanner-like app the data came from - CONFIRM with user) and write the one-page protocol; fill device matrix with the real iPhone model.
5. Drift: pose-graph/loop-closure (Open3D or numpy) + footprint ablation on a multi-room capture; coarse video and photo paths that report WIDE honest intervals (never confident garbage); stitching with adjacency graph.
6. Head-to-head: run a free consumer app (Polycam/magicplan) on 2 rooms, tabulate per-dimension errors.
7. Keep committing in small real commits (history is scored). Update docs/compliance_matrix.md truthfully; trim report to <=6 pages.

## 8. Practical notes
- Sandbox lacked pytest/network; tests were run with a homemade shim - user's `pytest` shows 80 passed. Open3D was unavailable, so the code is numpy/scipy/OpenCV/matplotlib only (PLY writer is custom).
- Updates were delivered to the user as `update.zip` overlays (src, configs, tests, scripts, docs) - the user's local git may have uncommitted overlay changes; ask them to `git add -A && git commit` in meaningful chunks.
- Config knobs live in configs/default.yaml (copied to lidar.yaml): auto_calibration, reconstruction (confidence_min=1 is an ASSUMED ARKit 0/1/2 semantic), keyframes, planes (up_*, imu_*), walls, floorplan, openings, uncertainty, debug.
- User style: short on time, wants working results; be honest about what is unverified; do not claim gates pass.

## 9. UPDATE - wall closure / outline rework (code only; NOT yet run on the real scans)
Why: the real `single_room` top-down showed 5 candidates on one wall, a wall off the dominant axes left in the plan, a wall hidden behind furniture cut short,
patchy floor evidence driving a ragged outline, and area swinging 11.1 -> 11.8 m2 with pruning changes.
What changed (all config-switchable; `configs/ablation_legacy_walls.yaml` reproduces the old behaviour on any scan):
- `geometry/walls.py`: `dominant_axes` (frame first, with a `contrast` reliability score), `detect_walls_axes` (peaks of offset histograms on the two axes, non-maximum
  suppression, free-angle raw fit kept, off-axis lines REJECTED AND LOGGED), `close_corners` (open ends extended <= 1.0 m, recorded as `inferred_ext_m`), tall-column filter
  in `wall_band_cells` (`min_column_extent_m` 0.4), bridged extents with density test and sparse-tail trimming.
- `geometry/floorplan.py`: area/polygon from the wall-enclosed region that HOLDS THE CAMERA PATH (not the union, not raw floor evidence); reports `floor_coverage_frac`,
  floor evidence outside the region, camera fraction inside. `geometry/room_layout.py`: the wall stage as one function used by `lidar_run` and the tests.
Evidence (SYNTHETIC ONLY, `docs/synthetic_wall_ablation.md`, `scripts/synthetic_wall_ablation.py`): furnished rotated room, 12 scenes. Before: -11.6% area. Axis walls without the
column filter: -46.8% (table/bed blobs become rows of walls). Legacy + filter: -11.6%. Both together: 0.0% (worst 0.1%). 92 tests pass.
KNOWN RISKS to check first on the real scan: (1) `min_column_extent_m` may shorten walls seen only low; compare with `configs/ablation_no_column_filter.yaml`; (2) `min_axis_contrast`
1.5 is an assumption - read the real value in `diagnostics.json` -> quality.manhattan.axes.contrast; (3) tall furniture fronts become boundaries, so area is observed-surface-bounded,
not tape wall-to-wall: record the convention in gt.yaml; (4) area interval is still the uncalibrated prior; (5) multiple enclosed regions containing cameras (= several rooms) are NOT
merged and NOT yet split into rooms - only the camera-majority region is reported.
Sandbox note: pytest is unavailable offline; the previous run used a small shim (approx/fixture/raises/parametrize) - the user's `python -m pytest -q` is authoritative.
