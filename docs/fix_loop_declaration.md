# Fix declaration (one page)

**Benchmark:** room A, LiDAR tier, two scans (`benchmark/room_a_scan1`, `room_a_scan2`). Ground truth: iPhone Measure app readings, NOT tape/laser. Regenerate everything with
`python scripts/fix_loop.py --input benchmark/room_a_scan1 --ground-truth benchmark/room_a_gt.yaml --repeat-input benchmark/room_a_scan2 --before-config configs/fix_before.yaml --after-config configs/fix_after.yaml`.

## 1. Worst-performing gate (before)
`wall_lengths_lidar`: max error **33.44%** (target <= 0.5% per wall). Other gates also failed: ceiling height error 15.01 cm (<= 1.5), opening widths 0/3 within 2 cm with 3 extra detections, ceiling repeat spread 2.23 cm (<= 1 cm).
**Caveat found while diagnosing:** the first ground-truth file listed four walls of ~3.33 m but a floor area of 15.67 m2 (implies ~4.7 m long walls), so part of the 33.44% is a ground-truth entry error (a 3.33 m entry was matched to a 4.44 m wall). Corrected ground truth: **[FILL after re-measuring the two long walls]**.

## 2. Root-cause hypothesis and evidence
(a) Furniture-height lines displace the real walls. Evidence from `diagnostics.json`: every outermost parallel line had boundary-ray fraction 0.0 (blocked), the kept left wall sat ~0.45 m inside the end of the top wall, interior lines (`wall_15`, ray fraction 0.27) cut the room so the footprint was 8.1 m2 vs 15.67 m2 measured, and 6 openings (GT: 3) were detected on interior furniture lines.
(b) Fitted wall lines overshoot corners, so opposite walls of one room reported lengths 0.3-0.65 m apart (4.84 vs 4.19 m, 3.32 vs 3.65 m).

## 3. Fix shipped and prediction
Fix 1 (`walls.high_band`, commit **[FILL hash]**): when the ceiling is observed, a candidate wall needs cell support above 2.1 m; lines that exist only at furniture height are dropped.
Fix 2 (`walls.trim_overshoot`, commit **[FILL hash]**): wall ends are cut at the perpendicular wall they meet (corner to corner).
**Prediction (written when the fixes were shipped):** opposite walls agree within 1 cm; max wall error against corrected ground truth <= 5% (the 0.5% gate will NOT be reached); floor area within 15% of 15.67 m2; ceiling gate unchanged (not addressed by this fix).

## 4. After (measured) and post-mortem
Observed in `plan.png` after the fixes: top/bottom 3.32 m and left/right 4.19 m (opposite walls identical), floor area 13.9 m2 (-11% vs 15.67 m2), no phantom openings (0 detected; GT 3, so all 3 are now misses), ceiling 3.30 m (unchanged).
Gate table from `outputs/fix_loop/report.md`: **[PASTE]**.
Post-mortem: the kept bottom line (`wall_05`) sits ~0.5 m inside a pruned line (`wall_03`) that had the strongest ceiling-height support (0.97). 4.19 + 0.5 = 4.7 m, the length the measured area implies. Both have high-band evidence, so the inner one may be a floor-to-ceiling built-in; which one is "the wall" depends on the measuring convention, which we did not settle. The ceiling gate (two ceiling-height peaks 22 cm apart: 3.30 m chosen, 3.50 m would match the 3.45 m measured) and the opening misses are NOT fixed.
