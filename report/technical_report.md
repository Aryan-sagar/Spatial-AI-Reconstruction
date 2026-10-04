# Technical report (DRAFT - covers only what is implemented)

**Architecture.** ScanDataset (read-only loaders, sync check) -> empirical convention selection -> keyframes -> voxel cloud -> floor/ceiling
(pose-derived up hint, height histogram, constrained RANSAC) -> walls (2-D RANSAC on floor-plane cells, Manhattan snap with raw geometry kept,
corner snap, ray-cast boundary pruning) -> footprint -> openings (mid-height density gaps validated by see-through points) -> Scene JSON 0.1 -> plan.

**Uncertainty.** Half-width = tier prior (abs + rel*value) + z*fit sigma, doubled when support is low; weaker tiers have wider priors.
Calibrated alternative: conformal quantiles from ground-truth residuals (`evaluate --calibrate`). Currently uncalibrated.

**Validation so far.** Synthetic box room with known dimensions (logic only): ceiling, wall lengths, area, door width recovered within a few cm.
**Real-data accuracy against tape/laser ground truth: not measured yet.**

**Not implemented:** photo, video, damage/concealed/scope, drift correction, stitching, head-to-head, fix loop.
**Known limitations:** confidence semantics assumed; per-frame intrinsics vary ~2-3% and the fixed-vs-per-frame choice is untested; openings need
see-through evidence; furnished-room clutter handling is basic.
