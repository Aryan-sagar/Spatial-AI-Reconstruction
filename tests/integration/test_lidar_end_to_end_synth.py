"""End-to-end on a synthetic room with exactly known geometry. Validates the PIPELINE LOGIC only - not real-data accuracy."""
import json
import numpy as np
from applied_ai.output.schema import validate_scene
from applied_ai.reconstruction.lidar_run import run_lidar
from synth import make_room_scan


def _scene(tmp_path, cfg, **kw):
    cfg["synchronization"]["duration_tolerance_frac"] = 0.1
    cfg["auto_calibration"]["pair_translation_m"] = [0.05, 1.0]
    cfg["planes"]["imu_device_to_camera"] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]  # synthetic device frame == ARKit camera frame
    gt = make_room_scan(tmp_path / "room", n=90, noise_m=0.01, **kw)
    return gt, run_lidar(tmp_path / "room", cfg, tmp_path / "out")


def test_room_measurements_match_ground_truth(tmp_path, cfg):
    gt, scene = _scene(tmp_path, cfg)
    r = scene["rooms"][0]
    assert validate_scene(scene) == []
    assert abs(r["ceiling_height"]["value"] - gt["H"]) < 0.015
    assert abs(r["floor_area"]["value"] - gt["area"]) / gt["area"] < 0.02
    assert sorted(round(w["length"]["value"]) for w in r["walls"]) == [3, 3, 4, 4]
    for w in r["walls"]:
        assert abs(w["length"]["value"] - round(w["length"]["value"])) < 0.04
    assert len(r["openings"]) == 1 and abs(r["openings"][0]["width"]["value"] - gt["door_width"]) < 0.03
    # every measurement carries an interval that contains the (known) truth here
    ci = r["ceiling_height"]["confidence_interval"]
    assert ci["lower"] <= gt["H"] <= ci["upper"]
    for name in ("scene.json", "plan.png", "plan.svg", "pointcloud.ply", "diagnostics.json", "convention_report.json"):
        assert (tmp_path / "out" / name).exists()


def test_no_ceiling_is_unobserved_not_invented(tmp_path, cfg):
    """Camera never looks up -> ceiling must be reported unobserved with no value."""
    import synth
    orig = synth.trajectory
    synth.trajectory = lambda n, W, H, D, cam_h=1.4, radius=0.5: [(p, R) for p, R in orig(n, W, H, D, cam_h, radius)]
    cfg["synchronization"]["duration_tolerance_frac"] = 0.1
    cfg["auto_calibration"]["pair_translation_m"] = [0.05, 1.0]
    # remove ceiling by making the room very tall relative to the pitch range: ceiling at 9 m is beyond max_depth -> never observed
    gt = make_room_scan(tmp_path / "room", H=9.0, n=90, noise_m=0.01)
    scene = run_lidar(tmp_path / "room", cfg, tmp_path / "out")
    ch = scene["rooms"][0]["ceiling_height"]
    assert ch["status"] == "unobserved" and ch["value"] is None and ch["confidence_interval"] is None
    synth.trajectory = orig


def test_biased_camera_up_hint_does_not_break_floor_detection(tmp_path, cfg):
    """Camera looks ~40 deg downward all the time (strongly pitched capture). Pipeline must still find floor/walls."""
    import synth
    synth.PITCH_BIAS_DEG = -40.0
    try:
        gt, scene = _scene(tmp_path, cfg)
    finally:
        synth.PITCH_BIAS_DEG = 0.0
    r = scene["rooms"][0]
    assert scene["diagnostics"]["quality"]["up_hint"]["source"] == "trajectory_pca"
    assert abs(r["floor_area"]["value"] - gt["area"]) / gt["area"] < 0.05
    assert sorted(round(w["length"]["value"]) for w in r["walls"]) == [3, 3, 4, 4]
