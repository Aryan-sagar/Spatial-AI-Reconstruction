# Assumptions (all unresolved unless stated)

| Assumption | Status | Evidence | Validation |
|---|---|---|---|
| Depth unit (0.001 m per raw unit) | provisional | Observed raw range ~1132-5375 only; unit not proven | DepthScaleValidator vs ground truth (Phase 2/3+) |
| Pose convention (camera_to_world) | provisional | Not proven | Reproject/overlap test, Phase 3 |
| RGB/depth alignment (frame i <-> frame i) | provisional | Counts equal and durations agree (checked by sync report); per-frame alignment unproven | `inspect` sync report; visual cross-check later |
| Intrinsics mapping (RGB intrinsics / 7.5 for 256x192) | provisional | Matches 1920x1440 -> 256x192 ratio only | Calibration diagnostics, Phase 2 |
| IMU coordinate convention | provisional | Unexamined | Gravity vs floor-normal check later |
| Metric scale source (photo, video) | provisional | None yet | Phases 11-12 |
| Odometry timestamps non-uniform | observed (user-reported) | median dt ~0.01667 s vs mean ~0.0221 s | `inspect` reports dt median/mean/max |
