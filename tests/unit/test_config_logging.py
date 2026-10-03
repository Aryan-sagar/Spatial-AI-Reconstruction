import pytest
from applied_ai.config import config_hash, load_config
from applied_ai.logging_utils import stage


def test_default_loads_and_depth_scale_is_provisional():
    c = load_config()
    assert c["calibration"]["depth_scale"]["status"] == "provisional"
    assert c["pose"]["convention"] in ("camera_to_world", "world_to_camera")


def test_overlay_merges_and_hash_changes(tmp_path):
    p = tmp_path / "o.yaml"
    p.write_text("synchronization:\n  strict: false\n")
    c = load_config(p)
    assert c["synchronization"]["strict"] is False
    assert c["synchronization"]["fps_tolerance_frac"] == 0.02  # untouched sibling key
    assert config_hash(c) != config_hash(load_config())


def test_missing_config_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_stage_records_duration():
    rec = []
    with stage("x", rec, input_points=3):
        pass
    assert rec[0]["stage"] == "x" and rec[0]["duration_sec"] >= 0 and rec[0]["input_points"] == 3
