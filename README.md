# Spatial AI Reconstruction

An offline spatial-reconstruction pipeline for turning handheld RGB-D / LiDAR-style captures into a structured indoor scene representation, floor plan, measurements, and uncertainty-aware outputs.

The current implementation focuses on the **LiDAR / RGB-D single-room tier**. The repository is deliberately explicit about what is implemented, what is provisional, and what remains incomplete.

---

## Current Status

### Implemented

- Dataset inventory and capture validation
- RGB/depth/odometry consistency checks
- Frame synchronization diagnostics
- Camera intrinsics handling and depth projection
- Pose/convention selection with diagnostic evidence
- RGB-D / depth point-cloud reconstruction
- Floor and ceiling estimation
- Gravity / up-direction reasoning
- Manhattan wall extraction and regularization
- Structural wall segments and corner handling
- Opening detection
- Floor-plan generation
- Measurement confidence intervals
- Scene JSON validation
- Provenance and run diagnostics
- Deterministic rendered plan output
- Evaluation and gate-metric tooling
- Synthetic geometry validation and ablation tests

### Not Yet Implemented

- Photo-only reconstruction tier
- Video-only reconstruction tier
- Multi-room stitching and adjacency graph
- Damage classification / segmentation
- Concealed-damage rules
- Scope line-item generation
- Consumer-app head-to-head evaluation
- Production drift correction / loop closure
- Calibrated uncertainty from real benchmark ground truth
- Official published-schema adapter

These limitations are intentionally exposed in the generated scene diagnostics and compliance documentation rather than being hidden behind a confident-looking result.

---

## Pipeline

```text
Capture
   │
   ├── depth / RGB / odometry / IMU
   │
   ▼
Dataset Inventory
   │
   ▼
Synchronization Checks
   │
   ▼
Convention Selection
   │
   ├── pose direction
   ├── camera axes
   └── depth scale
   │
   ▼
RGB-D Reconstruction
   │
   ▼
Point Cloud
   │
   ├── floor / ceiling
   ├── up-axis estimation
   ├── wall-band filtering
   ├── Manhattan wall extraction
   └── openings
   │
   ▼
Floor Plan
   │
   ├── walls
   ├── corners
   ├── openings
   └── measurements
   │
   ▼
Uncertainty + Provenance
   │
   ▼
scene.json
plan.png
plan.svg
diagnostics.json
```

The current end-to-end LiDAR runner wires these stages together and validates the final scene before publishing the output artifacts. :chatgpt-content-reference{index="1"}

---

## Repository Structure

```text
Spatial-AI-Reconstruction/
│
├── benchmark/
│   └── supplied scan data
│
├── configs/
│   ├── lidar.yaml
│   └── ablation_*.yaml
│
├── docs/
│   ├── assumptions.md
│   ├── compliance_matrix.md
│   ├── ground_truth_format.md
│   └── synthetic_wall_ablation.md
│
├── reproduction/
│   └── reproducibility helpers
│
├── report/
│   └── technical / benchmark documentation
│
├── scripts/
│   ├── setup.ps1
│   └── diagnostic utilities
│
├── src/
│   └── applied_ai/
│       ├── calibration/
│       ├── domain/
│       ├── evaluation/
│       ├── geometry/
│       ├── io/
│       ├── output/
│       ├── reconstruction/
│       ├── uncertainty.py
│       └── cli.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# Requirements

- Windows 10/11
- PowerShell
- Python 3.10+
- A local copy of the supplied benchmark capture

Python dependencies are declared in `pyproject.toml`; the project uses a `src/` package layout and pytest configuration. :chatgpt-content-reference{index="2"}

---

# Setup

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or install directly:

```powershell
python -m pip install -e .
python -m pip install pytest
```

---

# Test Suite

Run the complete test suite:

```powershell
python -m pytest -q
```

Or:

```powershell
make test
```

The repository also contains integration and synthetic-geometry tests covering scene construction, wall extraction, conventions, uncertainty validation, and related geometry behavior.

---

# CLI

The main entry point is:

```powershell
python -m applied_ai.cli
```

Available commands:

```text
inspect
run
validate
evaluate
```

---

## 1. Inspect a Capture

Run inventory and synchronization checks:

```powershell
python -m applied_ai.cli inspect `
  --input benchmark/single_scan_with_ceiling
```

This produces inventory and synchronization diagnostics under:

```text
outputs/
```

The current repository has successfully used this flow on the supplied scan folders. :chatgpt-content-reference{index="3"}

---

## 2. Run Reconstruction

Run the current LiDAR / RGB-D pipeline:

```powershell
python -m applied_ai.cli run `
  --input benchmark/single_scan_with_ceiling `
  --tier lidar `
  --config configs/lidar.yaml
```

The pipeline performs convention selection, reconstruction, floor/ceiling analysis, wall extraction, floor-plan construction, opening detection, uncertainty assignment, scene validation, and rendering. :chatgpt-content-reference{index="4"}

Outputs are written under:

```text
outputs/single_scan_with_ceiling/
```

Typical artifacts:

```text
scene.json
plan.png
plan.svg
pointcloud.ply
pointcloud_stats.json
diagnostics.json
convention_report.json
run.log
debug/
```

---

## 3. Reconstruct One Frame

For debugging a specific frame:

```powershell
python -m applied_ai.cli run `
  --input benchmark/single_scan_with_ceiling `
  --tier lidar `
  --config configs/lidar.yaml `
  --frame 1000
```

This is useful for isolating projection, pose, and local geometry behavior without fusing the full trajectory.

---

## 4. Validate a Scene

Validate the generated internal scene representation:

```powershell
python -m applied_ai.cli validate `
  --scene outputs/single_scan_with_ceiling/scene.json
```

The validator checks required scene fields and ensures that every observed/estimated measurement has a valid confidence interval. Explicitly unobserved measurements are allowed without a numeric value. :chatgpt-content-reference{index="5"}

---

## 5. Evaluate Against Ground Truth

Ground-truth evaluation:

```powershell
python -m applied_ai.cli evaluate `
  --scene outputs/single_scan_with_ceiling/scene.json `
  --ground-truth gt.yaml
```

With uncertainty calibration:

```powershell
python -m applied_ai.cli evaluate `
  --scene outputs/single_scan_with_ceiling/scene.json `
  --ground-truth gt.yaml `
  --calibrate
```

This produces:

```text
gate_results.json
benchmark_results.json
calibration.json
```

The evaluation system contains the required gate/metric machinery, but **real-data benchmark values are only valid once measured against actual ground truth**. :chatgpt-content-reference{index="6"}

---

# Scene Output

The pipeline currently writes an **internal scene schema v0.1**.

It is intentionally separated from the future published schema through an adapter layer.

```text
schema_version
capture_id
tier
rooms
adjacency
measurements
damages
concealed_damage_flags
scope_line_items
diagnostics
provenance
```

The official published schema was not supplied with the case study, so the repository does **not** pretend that the internal schema is the official contract. :chatgpt-content-reference{index="7"}

---

# Measurements and Uncertainty

Measurements are represented with:

```json
{
  "value": 3.42,
  "unit": "m",
  "status": "estimated",
  "method": "..."
  "confidence_interval": {
    "lower": 3.30,
    "upper": 3.54,
    "confidence_level": 0.95
  }
}
```

The current uncertainty intervals are **uncalibrated priors** unless empirical calibration has been performed using real ground-truth measurements.

The intended calibration workflow is:

```text
scene.json
    +
real ground truth
    ↓
cli evaluate --calibrate
    ↓
calibration.json
    ↓
uncertainty calibration file
```

This distinction is important because an uncertainty interval without empirical calibration should not be presented as a verified accuracy guarantee. :chatgpt-content-reference{index="8"}

---

# Coordinate and Calibration Assumptions

The reconstruction does not silently assume that every capture uses the correct convention.

The pipeline records and diagnoses:

- pose direction
- camera coordinate convention
- depth scale
- intrinsics mapping
- RGB/depth alignment assumptions
- confidence-map semantics
- up-direction estimation
- wall-frame assumptions

The current system can select conventions empirically and records the resulting evidence in:

```text
convention_report.json
```

A run can fail when the evidence is poor or ambiguous.

Current depth scale remains provisional until validated against real ground truth. :chatgpt-content-reference{index="9"}

---

# Structural Geometry

The current LiDAR/RGB-D path uses a geometry-first approach:

```text
point cloud
    ↓
floor / ceiling
    ↓
gravity / up-axis reasoning
    ↓
wall-band extraction
    ↓
Manhattan wall estimation
    ↓
wall regularization
    ↓
corners / enclosure
    ↓
opening detection
    ↓
floor plan
```

The wall model explicitly distinguishes observed geometry from inferred geometry.

For example, corner extensions are retained as inferred quantities rather than silently being presented as directly measured wall evidence. :chatgpt-content-reference{index="10"}

---

# Openings

The current LiDAR/RGB-D implementation includes a geometry-first opening detector.

Detected openings are represented separately from walls and carry their own measurement metadata.

Observed openings can include:

```text
door
window
unknown
```

Opening measurements should still be treated as provisional until benchmarked against tape/laser ground truth.

---

# Floor Area Semantics

The current floor-area implementation is intentionally conservative.

The published area is based on the detected structural enclosure / observed free-space semantics rather than automatically assuming that every inferred wall-to-wall region has been fully observed.

The diagnostics expose:

```text
floor coverage
outside evidence
camera-inside fraction
inferred corner extensions
footprint reliability
```

An unreliable footprint is flagged instead of being silently converted into a high-confidence room measurement. :chatgpt-content-reference{index="11"}

---

# Benchmarking

The repository contains tooling for:

- measurement comparison
- gate evaluation
- repeatability checks
- confidence calibration
- wall/footprint diagnostics
- synthetic geometry tests
- ablation experiments

However, the current compliance status is:

| Capability | Status |
|---|---|
| Dataset inventory / synchronization | ✅ Done |
| LiDAR single-room pipeline | ✅ Implemented |
| Floor / ceiling estimation | ✅ Implemented |
| Structural walls | ✅ Implemented |
| Openings | ✅ Implemented |
| Measurement intervals | ✅ Implemented, uncalibrated |
| Evaluation metrics | ✅ Implemented |
| Synthetic validation | ✅ Implemented |
| Real-GT accuracy | ⚠️ Not yet measured |
| Drift correction | ❌ Not implemented |
| Multi-room stitching | ❌ Not implemented |
| Video tier | ❌ Not implemented |
| Photo tier | ❌ Not implemented |
| Damage detection | ❌ Not implemented |
| Concealed damage | ❌ Not implemented |
| Scope line items | ❌ Not implemented |
| Consumer head-to-head | ❌ Not implemented |
| Fix loop | ❌ Not implemented |
| Official published schema | ⚠️ Adapter stub only |
| Clean-machine reproduction | ⚠️ Not yet verified |

This status intentionally matches the detailed compliance matrix in `docs/compliance_matrix.md`. :chatgpt-content-reference{index="12"}

---

# Reproducibility

The intended workflow is:

```text
install
  ↓
inspect
  ↓
run
  ↓
validate
  ↓
evaluate
```

Example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

.\.venv\Scripts\Activate.ps1

python -m applied_ai.cli inspect `
  --input benchmark/single_scan_with_ceiling

python -m applied_ai.cli run `
  --input benchmark/single_scan_with_ceiling `
  --tier lidar `
  --config configs/lidar.yaml

python -m applied_ai.cli validate `
  --scene outputs/single_scan_with_ceiling/scene.json
```

The remaining clean-machine reproduction step is still a verification item rather than something claimed as complete. :chatgpt-content-reference{index="13"}

---

# Known Limitations

The current release should **not** be interpreted as a complete three-tier production floor-plan system.

Most importantly:

1. The current LiDAR/RGB-D pipeline is a single-room implementation.
2. Global drift correction / loop closure has not been implemented.
3. Photo and video reconstruction are not implemented.
4. Multi-room adjacency stitching is not implemented.
5. Damage, concealed damage, and scope-line generation are not implemented.
6. Real benchmark accuracy has not yet been established against laser/tape ground truth.
7. Uncertainty intervals remain uncalibrated until real ground-truth calibration is performed.
8. The official published output schema was not supplied, so scene output currently uses internal schema v0.1. :chatgpt-content-reference{index="14"}

These limitations are part of the engineering result and are intentionally surfaced in diagnostics and documentation.

---

# Design Principles

### Evidence over assumptions

Every important geometric assumption should have either:

- supporting evidence,
- a diagnostic,
- a fallback,
- or an explicit failure state.

### Geometry before semantics

The current system prioritizes reliable spatial structure:

```text
depth
→ geometry
→ structural surfaces
→ room layout
→ measurements
```

rather than producing semantic labels unsupported by the captured evidence.

### Uncertainty is part of the output

A measurement without an uncertainty model is incomplete for this problem.

### Failure should be visible

When the geometry is unreliable, the system should say so instead of publishing confident garbage.

---

# Project Status

This repository is an **engineering prototype / case-study implementation**.

The strongest currently supported path is:

```text
LiDAR / RGB-D
      ↓
single room
      ↓
3D reconstruction
      ↓
structural geometry
      ↓
floor plan
      ↓
measurements + uncertainty
      ↓
validated scene output
```

Future work is focused on expanding the same output contract to photo/video captures, improving global consistency, calibrating measurement uncertainty against real ground truth, and completing multi-room and damage workflows.

---

## License

This repository was created as part of an applied AI engineering case study.