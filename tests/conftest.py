"""Tiny synthetic scans for tests. Built in tmp dirs; never touches benchmark/."""
import cv2
import numpy as np
import pytest

from applied_ai.config import load_config

ODO_HEADER = " timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"


def make_scan(root, n=6, depth_hw=(192, 256), rgb_wh=(192, 144), fps=10.0, dt=0.1, drop=(), skip_files=(), odo_rows=None, quat=(0, 0, 0, 1)):
    root.mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(exist_ok=True)
    (root / "confidence").mkdir(exist_ok=True)
    for i in range(n):
        if i in drop:
            continue
        a = (np.arange(depth_hw[0] * depth_hw[1], dtype=np.uint16).reshape(depth_hw) % 4000) + 1000
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), a)
        cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), np.full(depth_hw, 2, np.uint8))
    rows = odo_rows if odo_rows is not None else list(range(n))
    lines = [ODO_HEADER]
    for k, f in enumerate(rows):
        lines.append(f" {100 + k * dt:.4f}, {f}, 0.0, 0.0, {0.01 * k}, {quat[0]}, {quat[1]}, {quat[2]}, {quat[3]}, 100.0, 100.0, 50.0, 40.0, 50.0, 40.0")
    (root / "odometry.csv").write_text("\n".join(lines) + "\n")
    (root / "camera_matrix.csv").write_text("100.0, 0.0, 50.0\n0.0, 100.0, 40.0\n0.0, 0.0, 1.0\n")
    (root / "imu.csv").write_text("timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n100.0, 0, 0, 9.8, 0, 0, 0\n100.1, 0, 0, 9.8, 0, 0, 0\n")
    vw = cv2.VideoWriter(str(root / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, rgb_wh)
    for i in range(n):
        vw.write(np.full((rgb_wh[1], rgb_wh[0], 3), (i * 20) % 255, np.uint8))
    vw.release()
    for s in skip_files:
        p = root / s
        (p.unlink() if p.is_file() else __import__("shutil").rmtree(p))
    return root


@pytest.fixture
def cfg():
    c = load_config()
    c["dataset"]["rgb"].update(width=192, height=144)  # synthetic RGB size
    return c


@pytest.fixture
def scan(tmp_path):
    return make_scan(tmp_path / "scan_ok")
