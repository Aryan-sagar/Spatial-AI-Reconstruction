import numpy as np
import pytest
from applied_ai.geometry.planes import fit_plane_svd, plane_distance, ransac_plane
from applied_ai.geometry.walls import detect_walls, manhattan_regularize, snap_corners, wall_length
from applied_ai.uncertainty import UncertaintyModel, conformal_quantile


def test_plane_distance_and_svd_fit():
    rng = np.random.default_rng(0)
    P = np.column_stack((rng.uniform(-1, 1, 500), rng.uniform(-1, 1, 500), 0.5 + rng.normal(0, 1e-3, 500)))
    n, d = fit_plane_svd(P)
    assert abs(abs(n[2]) - 1) < 1e-3 and abs(abs(d) - 0.5) < 2e-3
    assert plane_distance(np.array([[0, 0, 2.0]]), np.array([0, 0, 1.0]), -0.5)[0] == pytest.approx(1.5)


def test_ransac_finds_plane_among_outliers():
    rng = np.random.default_rng(1)
    plane = np.column_stack((rng.uniform(-2, 2, 2000), rng.uniform(-2, 2, 2000), rng.normal(0, 0.003, 2000)))
    junk = rng.uniform(-2, 2, (1000, 3))
    n, d, m = ransac_plane(np.vstack((plane, junk)), 0.02, 300, rng, 5000)
    assert abs(abs(n[2]) - 1) < 0.01 and abs(d) < 0.01 and m.sum() > 1900


def test_ransac_normal_hint_restricts_orientation():
    rng = np.random.default_rng(2)
    wall = np.column_stack((rng.normal(0, 0.003, 3000), rng.uniform(-2, 2, 3000), rng.uniform(-2, 2, 3000)))
    floor = np.column_stack((rng.uniform(-2, 2, 800), rng.uniform(-2, 2, 800), rng.normal(0, 0.003, 800)))
    n, d, m = ransac_plane(np.vstack((wall, floor)), 0.02, 400, rng, 5000, np.array([0, 0, 1.0]), 20.0)
    assert abs(n[2]) > 0.98  # picks the floor although the wall has more support


def test_wall_lines_manhattan_and_corner_snap_make_exact_rectangle():
    c = 0.02
    xs = np.arange(0, 4.0 + c, c); ys = np.arange(0, 3.0 + c, c)
    cells = np.vstack([np.column_stack((xs, np.zeros_like(xs))), np.column_stack((xs, np.full_like(xs, 3.0))),
                       np.column_stack((np.zeros_like(ys), ys)), np.column_stack((np.full_like(ys, 4.0), ys))])
    # rotate the whole room by 17 degrees to test manhattan alignment
    a = np.radians(17); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    cells = cells @ R.T
    cfg = {"walls": {"cell_m": c, "line_inlier_m": 0.03, "min_wall_length_m": 0.6, "max_walls": 8, "endpoint_trim_frac": 0.0,
                     "corner_snap_m": 0.3, "extent_gap_split_m": 1.5, "manhattan": {"enabled": True, "snap_tolerance_deg": 6.0}}, "planes": {"seed": 0}}
    walls = detect_walls(cells, cfg)
    assert len(walls) == 4
    reg, info = manhattan_regularize(walls, cfg)
    reg = snap_corners(reg, cfg)
    assert sorted(round(wall_length(w), 2) for w in reg) == [3.0, 3.0, 4.0, 4.0]
    assert info["dominant_angle_deg"] == pytest.approx(17.0, abs=0.5)
    assert "_pts" in walls[0] and not walls[0].get("regularized")  # raw geometry retained


def test_measurement_confidence_interval_widens_with_weaker_tier_and_support():
    cfg = {"uncertainty": {"confidence_level": 0.95, "z": 1.96, "calibration_file": None, "low_support_inflation": 2.0,
                           "priors": {"lidar": {"rel": 0.01, "abs_m": 0.01}, "video": {"rel": 0.04, "abs_m": 0.05}, "photo": {"rel": 0.08, "abs_m": 0.1}}}}
    w = {t: UncertaintyModel(cfg, t).interval("wall_length", 4.0)["half_width"] for t in ("lidar", "video", "photo")}
    assert w["lidar"] < w["video"] < w["photo"]
    m = UncertaintyModel(cfg, "lidar")
    assert m.interval("wall_length", 4.0, n_support=10)["half_width"] > m.interval("wall_length", 4.0, n_support=5000)["half_width"]
    meas = m.measurement("wall_length", 4.0, "m", "x")
    ci = meas["confidence_interval"]
    assert ci["lower"] < 4.0 < ci["upper"] and ci["confidence_level"] == 0.95 and meas["uncertainty_basis"] == "uncalibrated_prior"
    assert UncertaintyModel.unobserved("m", "x", "no ceiling")["confidence_interval"] is None


def test_conformal_quantile():
    assert conformal_quantile([0.01] * 19 + [0.5], 0.1) == 0.01 or conformal_quantile([0.01] * 19 + [0.5], 0.1) == 0.5
    assert conformal_quantile(np.arange(1, 101) / 100, 0.05) == pytest.approx(0.96, abs=0.011)


def test_prune_non_boundary_drops_wall_seen_only_through_opening():
    from applied_ai.geometry.walls import prune_non_boundary

    def wall(i, a, b):
        a, b = np.array(a, float), np.array(b, float)
        t = (b - a) / np.linalg.norm(b - a)
        return {"id": f"w{i}", "centroid": (a + b) / 2, "direction": t, "s_min": -np.linalg.norm(b - a) / 2, "s_max": np.linalg.norm(b - a) / 2}
    ws = [wall(0, (0, 0), (4, 0)), wall(1, (4, 0), (4, 3)), wall(2, (4, 3), (0, 3)), wall(3, (0, 3), (0, 0)), wall(4, (-2, -1), (-2, 4))]  # w4 outside
    keep, info = prune_non_boundary(ws, np.array([[2.0, 1.5], [1.0, 1.0], [3.0, 2.0]]), {"walls": {"min_boundary_ray_frac": 0.02}})
    assert [w["id"] for w in keep] == ["w0", "w1", "w2", "w3"] and [p["id"] for p in info["pruned"]] == ["w4"]
    assert abs(sum(info["ray_fractions"].values()) - 1.0) < 0.05  # every ray hits some wall in a closed room (w4 outside takes ~0)


def test_trajectory_up_hint_finds_normal_of_walking_plane():
    import pandas as pd
    from applied_ai.geometry.coordinate_system import trajectory_up_hint
    rng = np.random.default_rng(0)
    n = 80
    xz = rng.uniform(-2, 2, (n, 2))
    P = np.column_stack((xz[:, 0], 0.05 * rng.standard_normal(n) + 1.4, xz[:, 1]))  # walk in the x-z plane, y is height
    df = pd.DataFrame({"frame": np.arange(n), "x": P[:, 0], "y": P[:, 1], "z": P[:, 2], "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0})
    cfg = {"pose": {"convention": "camera_to_world", "camera_axes": "arkit"}}  # arkit, identity rotation: camera image-up = +y world
    up, info = trajectory_up_hint(df, list(range(n)), cfg)
    assert abs(up[1]) > 0.999 and up[1] > 0 and info["trajectory_std_m"][2] < 0.1


def test_gravity_up_from_imu_recovers_up_and_mapping():
    import pandas as pd
    from scipy.spatial.transform import Rotation
    from applied_ai.geometry.coordinate_system import gravity_up_from_imu
    rng = np.random.default_rng(3)
    n = 400
    Rs = Rotation.from_euler("yxz", np.column_stack((rng.uniform(0, 2 * np.pi, n), rng.uniform(-0.6, 0.6, n), rng.uniform(-0.3, 0.3, n)))).as_matrix()
    M = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # device -> opencv camera (a proper rotation)
    g_read = np.array([0.0, -1.0, 0.0])  # world: y up; reading points toward gravity
    a = np.einsum("jk,nk->nj", np.linalg.inv(M), np.einsum("nij,i->nj", Rs, g_read))  # M^-1 R^T g
    t = np.arange(n) * 0.01
    q = Rotation.from_matrix(Rs).as_quat()
    odo = pd.DataFrame({"timestamp": t, "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3]})
    imu = pd.DataFrame({"timestamp": t, "a_x": a[:, 0], "a_y": a[:, 1], "a_z": a[:, 2]})
    cfg = {"pose": {"convention": "camera_to_world", "camera_axes": "opencv"},
           "planes": {"imu_reading_direction": "toward_gravity", "imu_min_resultant": 0.85, "imu_min_margin": 0.1, "imu_device_to_camera": np.eye(3).tolist()}}
    up, d = gravity_up_from_imu(imu, odo, cfg)  # wrong prior -> falls back to the search and still finds the right mapping
    assert d["reliable"] and "searched" in d["mapping_source"] and np.allclose(up, [0, 1, 0], atol=1e-6) and np.allclose(d["device_to_camera_mapping"], M)
    cfg["planes"]["imu_device_to_camera"] = M.tolist()
    up_p, d_p = gravity_up_from_imu(imu, odo, cfg)
    assert "prior" in d_p["mapping_source"] and np.allclose(up_p, [0, 1, 0], atol=1e-6)
    cfg["planes"]["imu_reading_direction"] = "away_from_gravity"
    assert np.allclose(gravity_up_from_imu(imu, odo, cfg)[0], [0, -1, 0], atol=1e-6)
    imu2 = imu.copy(); imu2[["a_x", "a_y", "a_z"]] = rng.normal(size=(n, 3))  # garbage IMU -> must be rejected, not trusted
    up2, d2 = gravity_up_from_imu(imu2, odo, cfg)
    assert up2 is None and d2["reliable"] is False


def test_imu_mirror_ambiguity_is_reported_not_guessed():
    """If the accel directions are coplanar (pitch-only motion) up vs down cannot be decided by searching: must be flagged unreliable."""
    import pandas as pd
    from scipy.spatial.transform import Rotation
    from applied_ai.geometry.coordinate_system import gravity_up_from_imu
    n = 300
    yaw = np.linspace(0, 6, n); pitch = 0.5 * np.sin(np.linspace(0, 12, n))
    Rs = Rotation.from_euler("yx", np.column_stack((yaw, pitch))).as_matrix()
    a = np.einsum("nij,i->nj", Rs, np.array([0.0, -1.0, 0.0]))
    t = np.arange(n) * 0.01; q = Rotation.from_matrix(Rs).as_quat()
    odo = pd.DataFrame({"timestamp": t, "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3]})
    imu = pd.DataFrame({"timestamp": t, "a_x": a[:, 0], "a_y": a[:, 1], "a_z": a[:, 2]})
    cfg = {"pose": {"convention": "camera_to_world", "camera_axes": "opencv"},
           "planes": {"imu_reading_direction": "toward_gravity", "imu_min_resultant": 0.85, "imu_min_margin": 0.1, "imu_device_to_camera": [[0, 0, 1], [1, 0, 0], [0, 1, 0]]}}
    up, d = gravity_up_from_imu(imu, odo, cfg)
    assert up is None and d["reliable"] is False


def test_renderer_falls_back_when_target_unwritable(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from applied_ai.output.renderer import _save
    (tmp_path / "plan.png").mkdir()  # a directory with the target name: opening it for writing raises OSError
    fig = plt.figure()
    out = _save(fig, tmp_path / "plan.png")
    assert out != tmp_path / "plan.png" and out.exists() and out.suffix == ".png"
