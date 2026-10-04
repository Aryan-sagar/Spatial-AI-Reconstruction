# Assumptions

| Assumption | Status | Evidence | Validation |
|---|---|---|---|
| Pose direction (camera_to_world vs world_to_camera) | **empirically selected per run** | multi-view depth-consistency score over both options (`convention_report.json`); run fails if best candidate is poor/ambiguous | `python -m applied_ai.cli run ...` (auto_calibration.enabled) ; `tests/integration/test_conventions_synth.py` |
| Camera axes of the pose frame (arkit vs opencv) | **empirically selected per run** | same score; ARKit-style axes are the default prior only | same |
| Depth scale (0.001 m/unit) | **provisional unless identified** | candidates 0.0001/0.001/0.01 are scored; status becomes `empirically_calibrated` only if the score is sensitive to +/-10% scale (`scale_identified`). Consistency is relative to pose translation units (assumed metres). Must still be checked against tape ground truth | `evaluate` residuals; `scale_sensitivity` in scene diagnostics |
| RGB/depth alignment (frame i <-> frame i) | provisional | counts, ids and durations agree on all 3 supplied scans (`inspect`); per-frame content alignment unverified | depth-consistency score implicitly uses depth+pose only; RGB is unused in the LiDAR geometry path |
| Intrinsics mapping to 256x192 depth | provisional | `scaled_rgb_intrinsics` (RGB intrinsics x depth/RGB size ratio), per-frame values from odometry. Per-frame fx varies ~2-3% within a scan | other modes are stubs; residual scale error shows in ground-truth comparison |
| Confidence map semantics (0/1/2, keep >=1) | provisional | assumed ARKit-style; not verified | set `reconstruction.confidence_min: 0` to disable; compare results |
| IMU axes / clock | unused in geometry | IMU is loaded and windowed only | none |
| Up direction | derived | pose-derived camera-up is a weak hint; floor plane + height histogram decide | `up_hint_vs_plane_deg` in diagnostics |
| Metric scale for photo/video | **not implemented** | tiers not built | - |
| Interval widths | **uncalibrated priors** | `uncertainty.priors` in config; replaced by empirical quantiles via `evaluate --calibrate` + `uncertainty.calibration_file` | needs real ground truth |
| Wall frame (Manhattan axes) | **assumed rectilinear; checked per run** | the frame angle is estimated from the wall-band cells (offset-histogram sharpness over 0-90 deg); `walls.min_axis_contrast` (1.5, ASSUMED) decides if it is trustworthy. If not, the run **falls back to legacy RANSAC and logs a warning** (`method_used` in `diagnostics.quality.manhattan`). Round / strongly non-rectilinear rooms are therefore not handled by the axis path | `test_dominant_axes_*`, `test_unreliable_axis_frame_falls_back_*`; real-data contrast value in `diagnostics.json` |
| Off-axis walls | **dropped and logged** | a line whose free-angle fit deviates > `snap_tolerance_deg` from its axis is `non_conforming` (listed with deviation and length), never kept silently. A genuinely non-orthogonal wall is therefore lost from the plan (but listed) | `test_non_conforming_wall_is_dropped_*` |
| Tall-column filter (`min_column_extent_m` = 0.4) | **ASSUMED; the single most data-dependent knob** | a 0.1 m column must span >= 0.4 m vertically to count as wall evidence, so table tops / beds (horizontal blobs) do not become rows of walls. Without it the synthetic furnished room gives -47% area. **Risk:** a wall seen only low (camera pitched down, 0.6 m was enough to shorten a wall in a synthetic test) loses its low-seen stretch. A warning is logged if > 70% of band points are removed | `scripts/synthetic_wall_ablation.py`; real-data ablation: `configs/ablation_no_column_filter.yaml` |
| Tall furniture | **not separable by height** | a wardrobe/fridge reaches the ceiling band like a wall. Its front face becomes a boundary of the observed free space; the wall behind it is bridged (below) if it is seen on both sides | synthetic scene only |
| Wall hidden behind furniture | **bridged, flagged as unobserved by the opening logic** | runs >= `bridge_min_run_m` (0.4) and >= `bridge_min_occupancy` (50% of 10 cm bins) are joined across gaps > 1.5 m; sparse tails are trimmed. The bridged stretch has no wall evidence and is reported as an unobserved gap, not as a measured wall | `test_hidden_wall_stretch_is_bridged_*` |
| Corner closing | **INFERRED, recorded** | an open wall end is extended to a perpendicular wall within `corner_extend_m` (1.0, ASSUMED). Each extension is stored as `inferred_ext_m` and appears in `diagnostics.quality.inferred_corner_extension_m`. Gaps larger than that are NOT closed (the outline then stays open and the run warns) | `test_close_corners_*` |
| Floor area semantics | **observed-surface-bounded, not wall-to-wall** | area = region enclosed by observed wall/furniture-front surfaces that holds the camera path (centre-line semantics), NOT the observed floor evidence. A tall furniture front can therefore shorten the area relative to a tape wall-to-wall measurement. **State the ground-truth convention in gt.yaml** (`docs/ground_truth_format.md`). `floor_coverage_frac`, outside-evidence area and camera-fraction-inside are reported so a weakly supported area is visible | `diagnostics.quality.footprint_enclosure` |
| Interval on floor area | **still the uncalibrated prior (rel 1.2%)** | the area interval does NOT yet reflect outline uncertainty (inferred corners, coverage); it will only be honest after `evaluate --calibrate` on real ground truth | needs ground truth |

## Footprint reliability (floorplan.min_camera_inside_frac, min_floor_coverage, max_outside_evidence_frac) - ASSUMED thresholds
A wall-enclosed floor area is published as a tight `observed` measurement unless (a) fewer than 50% of camera positions lie inside the enclosed region, or
(c) floor evidence outside the region exceeds 50% of its area, or no region is enclosed. Floor evidence backing under 50% of the region (b) is listed as a reason
only together with (a)/(c); on its own it is a warning (a correct, fully enclosed room that was mostly looked at horizontally must not be downgraded). Otherwise the area is published as
`estimated`, `reliable: false`, with the failed checks as `unreliable_reasons`, and its interval is [observed floor evidence area, convex hull of the
detected walls] instead of a tier prior; the plan draws it dashed red with the reasons. Thresholds are unvalidated: on the synthetic furnished rooms
all four pass (corridor seen through the door stays at 3-11% outside evidence); on `single_room` all three fail. Revisit once real ground truth exists.

## Multi-room segmentation (room_segmentation.*) - EXPERIMENTAL, ASSUMED parameters
Whole-property scans (the recruiter's `single_scan_*` captures are walk-throughs of several rooms) cannot be handled as one room. `geometry/room_segmentation.py`
splits the observed free space (floor evidence + camera trail, minus dilated wall-band cells) into rooms: persistence-filtered peaks of the distance transform
(`h_maxima_m`) as seeds, a priority flood to label the core, margins re-attached by nearest label. Narrow passages between basins are doorways
(`passage_width_m` = contact length, minus the obstacle dilation); basins under `min_room_area_m2` are dropped and listed, never merged silently.
Areas are obstacle-shrunk free-space areas, NOT tape wall-to-wall. Validated on synthetic layouts only (3 rooms + 3 doors, single room, solid wall, tiny pocket);
on the real scans it has only been eyeballed on masks recovered from debug images. Currently additive: `diagnostics.multiroom` + `debug/rooms.png`; the scene's
`rooms` is still the single-room result.
