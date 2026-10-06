# Device matrix

| Tier | Hardware | Runs? | Honest accuracy (measured) |
|---|---|---|---|
| LiDAR | iPhone Pro-class with LiDAR, model: **[FILL: Settings > General > About]**, export from Stray Scanner (assumed) | Yes (single room) | Room A, 2 scans, Measure-app ground truth (not tape): ceiling 3.30 m vs 3.45 m measured (15 cm), repeat spread 2.2 cm; floor area 13.9 m2 vs 15.67 m2 measured (-11%); wall lengths and openings: see `docs/fix_loop_declaration.md`. Intervals are uncalibrated priors. |
| Video | iPhone 15 or newer | **No** | Not implemented. |
| Photo | iPhone 15 or newer | **No** | Not implemented. |

Recruiter-provided scans (`single_room`, `single_scan_floor_only`, `single_scan_with_ceiling`) have no ground truth; they are development data only. On the two walkthrough scans the pipeline refuses to certify an area (marked unreliable with a wide range).
