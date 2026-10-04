# Capture protocol (one page)

**Who:** anyone with an iPhone. **Time:** about 5 minutes per room. **No technical knowledge needed.**

## Before you start
- **Phone:** iPhone 15 Pro or newer for the LiDAR scan. Any iPhone 15 or newer for video and photos.
- **LiDAR app:** **Stray Scanner** (free, App Store). *Working assumption: it matches the exported file layout; confirm before submission.*
- **Reference length:** put a tape measure on the floor, fully extended to 1 m, against a wall in each room, visible in the frame. *(Used by the photo and video tiers when present.)*
- Turn on all lights. Open every door between rooms. Move people and pets out.

## A. LiDAR scan (Stray Scanner)
1. Stand in the doorway of the first room. Hold the phone in landscape at chest height. Press record.
2. Walk slowly, about one step per second. For every wall, tilt the phone up to the ceiling edge, then down to the floor edge, then move on.
3. Pass through each door and scan the next room the same way. End at the starting doorway (a loop).
4. Stop. Scan time should be 3 to 5 minutes for a whole property.
5. **Avoid:** fast turns, walking backwards, pointing at mirrors or windows for long, covering the camera with fingers.
6. **Export:** share the scan folder to your computer (AirDrop or Files). The folder must contain exactly `depth/`, `confidence/`, `rgb.mp4`, `odometry.csv`, `camera_matrix.csv`, `imu.csv`. Copy it unchanged into `benchmark/<name>/`.

## B. Video walkthrough (native Camera app)
1. Video mode, 1x lens, landscape, flash off.
2. 60 to 90 seconds. Walk slowly through every room in the same order as the LiDAR scan. Pan along each wall, including the floor and ceiling edges.
3. Keep the tape reference in view in every room.

## C. Photos (native Camera app)
1. One folder per room, named after the room: `photos/<room>/`.
2. Take **4 to 8** photos per room, 1x lens, no zoom, no filters, flash off.
3. Stand in a corner and photograph the opposite corner. Repeat for the other corners. Include every door and window, and the wall-floor and wall-ceiling edges.

## D. Hand off
Run one command per capture: `python -m applied_ai.cli run --input benchmark/<name> --tier lidar`
*(Only the LiDAR tier is implemented so far. Video and photo are documented but not yet runnable.)*

## Tape ground truth (benchmark only)
Before scanning each room, measure with tape or laser and write it down: length and width (two readings each), both diagonals, each door and window clear width, ceiling height.
