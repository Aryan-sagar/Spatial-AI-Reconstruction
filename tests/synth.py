"""Synthetic room scans with exactly known geometry (for end-to-end tests only; never used by the pipeline)."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

DEPTH_WH = (256, 192)


def render_room_depth(origin, R_world_cam_cv, K, W, H, D, door=None):
    """Depth (z-forward, metres) of a box room [0,W]x[0,H]x[0,D] (y up). door=(zlo,zhi,ytop) cuts an opening in wall x=0
    that reveals a far plane at x=-2."""
    w, h = DEPTH_WH
    v, u = np.mgrid[0:h, 0:w]
    d_cv = np.stack([(u - K[2]) / K[0], (v - K[3]) / K[1], np.ones_like(u, float)], -1).reshape(-1, 3)
    d = d_cv @ R_world_cam_cv.T
    o = np.asarray(origin, float)
    lims = [W, H, D]
    best = np.full(len(d), np.inf)
    which = np.full(len(d), -1)
    for ax in range(3):
        for sgn, c in ((0, 0.0), (1, lims[ax])):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (c - o[ax]) / d[:, ax]
            ok = (t > 1e-6) & (t < best)
            best[ok] = t[ok]
            which[ok] = ax * 2 + sgn
    if door is not None:
        hit = o + best[:, None] * d
        in_door = (which == 0) & (hit[:, 2] > door[0]) & (hit[:, 2] < door[1]) & (hit[:, 1] < door[2])
        with np.errstate(divide="ignore", invalid="ignore"):
            t2 = (-2.0 - o[0]) / d[:, 0]
        best[in_door] = t2[in_door]
    return best.reshape(h, w)


PITCH_BIAS_DEG = 0.0


def trajectory(n, W, H, D, cam_h=1.4, radius=0.5):
    cx, cz = W / 2, D / 2
    out = []
    for i in range(n):
        a = 2 * np.pi * i / n
        pos = np.array([cx + radius * np.cos(a), cam_h, cz + radius * np.sin(a)])
        yaw = 2 * np.pi * i / n * 2.0  # two full turns
        pitch = np.radians(28) * np.sin(2 * np.pi * i / n * 3.0) + np.radians(PITCH_BIAS_DEG)  # look up and down
        R = Rotation.from_euler("y", yaw).as_matrix() @ Rotation.from_euler("x", pitch).as_matrix()  # ARKit-style cam->world
        out.append((pos, R))
    return out


def make_room_scan(root, W=4.0, H=2.6, D=3.0, n=90, door=(1.0, 2.0, 2.0), depth_scale=0.001, rgb_wh=(192, 144), noise_m=0.0, seed=0):
    root.mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(exist_ok=True)
    (root / "confidence").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    fxd = 190.0  # depth-grid focal length
    sx = rgb_wh[0] / DEPTH_WH[0]
    K = (fxd, fxd, DEPTH_WH[0] / 2, DEPTH_WH[1] / 2)
    A = np.diag([1.0, -1.0, -1.0])  # cv -> arkit camera axes
    lines = [" timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"]
    fps = 30.0
    vw = cv2.VideoWriter(str(root / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, rgb_wh)
    traj = trajectory(n, W, H, D)
    for i, (pos, R) in enumerate(traj):
        depth = render_room_depth(pos, R @ A, K, W, H, D, door)
        if noise_m:
            depth = depth + rng.normal(0, noise_m, depth.shape)
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), np.round(depth / depth_scale).astype(np.uint16))
        cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), np.full(DEPTH_WH[::-1], 2, np.uint8))
        q = Rotation.from_matrix(R).as_quat()
        lines.append(f" {1000 + i / fps:.6f}, {i}, {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}, {q[0]:.8f}, {q[1]:.8f}, {q[2]:.8f}, {q[3]:.8f}, "
                     f"{fxd * sx:.5f}, {fxd * sx:.5f}, {K[2] * sx:.5f}, {K[3] * sx:.5f},,")
        vw.write(np.zeros((rgb_wh[1], rgb_wh[0], 3), np.uint8))
    vw.release()
    (root / "odometry.csv").write_text("\n".join(lines) + "\n")
    (root / "camera_matrix.csv").write_text(f"{fxd * sx}, 0.0, {K[2] * sx}\n0.0, {fxd * sx}, {K[3] * sx}\n0.0, 0.0, 1.0\n")
    rows = ["timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z"]
    for k in range(int(n / fps * 100)):  # 100 Hz; reading points TOWARD gravity (iOS), device frame == ARKit camera frame
        t = k / 100.0
        R = traj[min(int(t * fps), n - 1)][1]
        a = R.T @ np.array([0.0, -1.0, 0.0])
        rows.append(f"{1000 + t:.6f}, {a[0]:.6f}, {a[1]:.6f}, {a[2]:.6f}, 0, 0, 0")
    (root / "imu.csv").write_text("\n".join(rows) + "\n")
    return {"W": W, "H": H, "D": D, "door_width": door[1] - door[0] if door else None, "area": W * D}
