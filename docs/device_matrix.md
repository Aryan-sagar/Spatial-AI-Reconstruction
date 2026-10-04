# Device matrix (only what was actually observed)

| tier | device | capture method | required sensors | supported resolution | expected accuracy | known limitations |
|---|---|---|---|---|---|---|
| lidar | not recorded in the dataset (files: rgb.mp4 1920x1440, 256x192 uint16 depth, confidence, odometry, imu) | handheld scan | depth + pose + intrinsics | 256x192 depth | **not yet measured on real ground truth** | tested only on 3 supplied scans (inspection) and a synthetic room (logic) |
| video | - | - | - | - | - | not implemented |
| photo | - | - | - | - | - | not implemented |
