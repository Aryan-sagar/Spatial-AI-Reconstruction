# Spatial AI Reconstruction

An offline pipeline that turns a handheld LiDAR / RGB-D capture into a dimensioned floor plan with walls, ceiling height, floor area, openings and a confidence interval on every measurement.

**Scope in one line:** the **LiDAR tier, single room** is implemented and has been run against a real benchmark room. Photo tier, video tier, drift correction, damage outputs and the stitched whole-property plan are **not implemented**. Everything that is missing, unverified or unreliable is stated below and in `docs/compliance_matrix.md`; the pipeline also reports it at run time instead of hiding it behind a confident-looking number.

---

## Status

| Capability | Status |
|---|---|
| Dataset inventory, synchronisation checks | Done |
| Pose direction / camera axes / depth scale chosen from evidence, not assumed | Done |
| Point cloud, floor, ceiling, up-axis (IMU gravity + height peaks) | Done |
| Walls (Manhattan axes), corner closing, corner-to-corner trimming | Done, first real-room numbers below |
| Furniture handling (tall-column filter, camera-crossing filter, high-band perimeter support) | Done, validated synthetically; one real room |
| Openings (doors / windows / unknown) | Implemented, **weak on the real room** |
| Floor area with a reliability verdict | Done |
| Confidence intervals on every measurement | Done, **uncalibrated priors** |
| Scene JSON validation, provenance, rendered plan | Done |
| Evaluation gates, repeatability, conformal calibration tooling | Done |
| Fix-loop tooling (`scripts/fix_loop.py`, before/after configs) | Done |
| Multi-room segmentation (free space, doorways, adjacency) | **Experimental**: synthetic layouts only; not part of `scene.rooms` |
| Real-data accuracy | **One room**, Measure-app reference (not tape/laser) |
| Photo tier / video tier | **Not implemented** |
| Drift correction (loop closure / pose graph + ablation) | **Not implemented** (poses are used as supplied) |
| Damage, concealed damage, scope items | **Not implemented** |
| Stitched multi-room plan in the scene | **Not implemented** |
| Consumer-app head-to-head | **Not implemented** |
| Published output schema | Not supplied; internal schema 0.1 only |
| Mirrors, glass, wet-look surfaces, low light | **Not tested** (see failure modes) |
| Clean-machine run | Not verified |

No pretrained models are used (`docs/model_disclosures.md`).

---

## Quick start (Windows, PowerShell, Python 3.10+)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

Manual install if the script fails:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Keep the repository outside OneDrive or any synced folder: file locks break the renderer's output writes (it falls back to a timestamped file and warns), and a copied virtual environment keeps pointing at the old location.

### Capture

Follow `docs/capture_protocol.md` (one page; Stray Scanner is the assumed capture app, inferred from the export layout and not confirmed). A capture folder must contain `depth/`, `confidence/`, `rgb.mp4`, `odometry.csv`, `camera_matrix.csv`, `imu.csv`. Put it under `benchmark/<name>/` and never edit it: raw data stays untouched and `benchmark/` is git-ignored.

### One command per capture

```powershell
python -m applied_ai.cli inspect --input benchmark\<name>
python -m applied_ai.cli run --input benchmark\<name> --tier lidar --config configs\lidar.yaml
python -m applied_ai.cli validate --scene outputs\<name>\scene.json
```

`run` accepts `--output <dir>` and `--frame <id>` (single-frame reconstruction for debugging).

### Evaluate against ground truth

```powershell
python -m applied_ai.cli evaluate --scene outputs\<name>\scene.json --ground-truth benchmark\<name>_gt.yaml --repeat-scene outputs\<name2>\scene.json --calibrate
```

Ground-truth format: `docs/ground_truth_format.md`. Gates reported: ceiling height, wall lengths, opening widths, ceiling repeat spread.

### Regenerate the fix-loop before / after

```powershell
python scripts\fix_loop.py --input benchmark\room_a_scan1 --ground-truth benchmark\room_a_gt.yaml --repeat-input benchmark\room_a_scan2 `
  --before-config configs\fix_before.yaml --after-config configs\fix_after.yaml
```

Writes `outputs/fix_loop/{before,after}/`, `comparison.json` and `report.md` (gate-by-gate table). The declaration is `docs/fix_loop_declaration.md`.

---

## Outputs

Per run, under `outputs/<name>/`:

```text
scene.json              internal schema 0.1: rooms, walls, openings, intervals, diagnostics, provenance
plan.png / plan.svg     rendered plan (unreliable areas drawn dashed red with the reasons)
diagnostics.json        conventions, floor/ceiling peaks, wall filtering, footprint reliability, multi-room result
convention_report.json  evidence for the chosen pose direction / axes / depth scale
pointcloud.ply, run.log
debug/topdown.png       floor evidence, wall-band cells, kept and pruned wall lines, camera path
debug/rooms.png         experimental multi-room segmentation
```

Every measurement carries `value`, `unit`, `status` (`observed`, `estimated`, `unobserved`), `method` and a `confidence_interval`. A ceiling that was not seen is `unobserved` with no value; nothing is fabricated.

---

## How it works

```text
capture -> inventory + sync checks -> convention selection (pose direction, camera axes, depth scale)
        -> keyframes -> point cloud -> up axis, floor, ceiling
        -> wall candidates (Manhattan axes)
             -> tall-column filter           furniture is low; walls are tall
             -> high-band support filter     needs evidence above 2.1 m when the ceiling is observed
             -> camera-crossing filter       a camera cannot walk through wall material
             -> ray-boundary pruning         keep what bounds the scanned space
        -> corner snapping / closing / corner-to-corner trimming
        -> footprint + reliability verdict -> openings -> intervals -> scene.json + plan
```

Key choices (details in `docs/assumptions.md`; thresholds there are marked ASSUMED where they are not validated):

- **Conventions are measured, not assumed.** Candidate pose directions, camera axes and depth scales are scored by multi-view depth consistency; the run fails loudly if no candidate clearly wins.
- **Up direction** from IMU gravity (sign) with trajectory PCA as a hint; floor and ceiling from height-histogram peaks. If the IMU mapping is unreliable the run says so.
- **Furniture vs walls.** Ray pruning alone prefers the nearest supported line, which in a furnished room is often a wardrobe face. The high-band filter and the camera-crossing filter exist for that reason; each can be switched off in `configs/` to regenerate a "before".
- **Corner-to-corner lengths.** Fitted lines that run past the wall they meet are cut at the corner (`walls.trim_overshoot`); a wall is never extended.
- **Floor-area reliability.** The area is flagged `reliable: false` when too few camera positions lie inside the enclosed region, when floor evidence outside it exceeds half of its area, or when no region is enclosed. It is then published as `estimated` with an interval bounded by evidence (observed floor area up to the convex hull of the detected walls), and the plan says so.
- **Multi-room segmentation (experimental).** Free space is carved from every keyframe's depth (visibility carving), split into rooms by distance-transform basins, with doorways accepted only if wall evidence stands at both ends. Areas are obstacle-shrunk free space, not wall-to-wall. It is reported in `diagnostics.multiroom` and `debug/rooms.png`, not in the scene's rooms.

### Uncertainty

Intervals are tier priors plus fit error, widened at low support, and are **uncalibrated** unless `uncertainty.calibration_file` is set from `evaluate --calibrate`. One benchmark room cannot calibrate them; do not read them as verified guarantees.

---

## Benchmark status

Room A (LiDAR, two scans of the same room, `benchmark/room_a_scan1`, `room_a_scan2`). **Ground truth is from the iPhone Measure app, not tape or laser**, so it is development ground truth only.

First evaluation (before the wall fixes): all four gates failed.

| Gate | Result | Target |
|---|---|---|
| Ceiling height error | 15.01 cm | <= 1.5 cm |
| Wall lengths, max error | 33.44% (partly a ground-truth entry error: four ~3.33 m walls listed against a 15.67 m2 area) | <= 0.5% |
| Opening widths | 0/3 within 2 cm, 3 extra detections | >= 85% |
| Ceiling repeat spread | 2.23 cm | <= 1 cm |

After the two wall fixes the rendered plan shows matching opposite walls (3.32 m and 4.19 m) and a floor area of 13.9 m2 against 15.67 m2 measured (-11%), with no phantom openings but also no detected openings (3 in the reference). The ceiling is unchanged: two ceiling-height peaks 22 cm apart exist and the lower, stronger one is chosen. The current gate table is regenerated by `scripts/fix_loop.py` rather than copied here.

The three recruiter-provided scans (`single_room`, `single_scan_floor_only`, `single_scan_with_ceiling`) have no ground truth and are development data only. The two `single_scan_*` captures are multi-room walkthroughs; on them the pipeline correctly refuses to certify an area (wide range, `reliable: false`).

---

## Known limitations and failure modes

1. Single-room assumption in the scene output; multi-room walkthroughs are flagged unreliable.
2. No drift correction: poses are used as supplied. Long or looping captures can show doubled walls.
3. Photo and video tiers do not exist.
4. Furnished rooms: a floor-to-ceiling built-in along a wall is indistinguishable from the wall by height; which surface counts as "the wall" depends on the measuring convention.
5. Two ceiling levels (bulkhead, tray ceiling) make the ceiling choice ambiguous.
6. Openings: detection relies on gaps in wall density plus see-through evidence; it misses openings with no see-through points.
7. Mirrors, glass, wet-look surfaces and low light are untested; depth there is unreliable.
8. A scan that never looks up has no observed ceiling and no high band.
9. Intervals are uncalibrated; reference measurements are not tape.
10. The official published schema was not supplied.

---

## Repository layout

```text
configs/        default.yaml, lidar.yaml, fix_before.yaml, fix_after.yaml, ablation_*.yaml
docs/           assumptions.md, compliance_matrix.md, capture_protocol.md, device_matrix.md,
                fix_loop_declaration.md, ground_truth_format.md, model_disclosures.md, synthetic_wall_ablation.md
report/         technical report
scripts/        setup.ps1, reproduce.ps1, fix_loop.py, synthetic_wall_ablation.py, diagnostics helpers
src/applied_ai/
  io/ calibration/ reconstruction/     loading, conventions, point cloud
  geometry/                            planes, coordinate_system, walls, room_layout, floorplan, openings, rooms,
                                       room_segmentation, visibility
  output/                              schema + validation, renderer
  evaluation/ uncertainty.py cli.py
tests/          unit/ and integration/ (synthetic ray-traced rooms with exact ground truth)
benchmark/      raw captures (git-ignored, never modified)
outputs/        run artifacts (generated)
```

The synthetic tests validate logic and mechanisms, **not** real-world accuracy.

---

## AI tooling disclosure

This project was built with AI coding assistance. Design decisions, thresholds and failure analysis are documented in `docs/assumptions.md` and the fix-loop declaration so that they can be defended and reproduced.

## License

Created as part of an applied AI engineering case study.
