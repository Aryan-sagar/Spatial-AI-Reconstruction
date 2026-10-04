import copy
import pytest
from applied_ai.calibration.conventions import ConventionError, apply_selection, select_conventions
from applied_ai.io.dataset import ScanDataset
from synth import make_room_scan


def test_selects_true_conventions_on_synthetic_room(tmp_path, cfg):
    cfg["synchronization"]["duration_tolerance_frac"] = 0.1
    cfg["auto_calibration"]["pair_translation_m"] = [0.05, 1.0]
    make_room_scan(tmp_path / "r", n=90)
    with ScanDataset(tmp_path / "r", cfg) as ds:
        rep = select_conventions(ds, cfg)
        assert rep["status"] == "ok", rep["notes"]
        assert rep["selected"] == {"direction": "camera_to_world", "camera_axes": "arkit", "depth_scale": 0.001}
        assert rep["candidates"][0]["median_rel_error"] < 0.01
        c2 = apply_selection(copy.deepcopy(cfg), rep)
        assert c2["calibration"]["depth_scale"]["status"] in ("empirically_calibrated", "provisional") and rep.get("scale_identified") is True


def test_weakly_identified_scale_stays_provisional(tmp_path, cfg):
    """Data stored at 0.5 mm/unit is NOT among the candidates; multi-view consistency barely notices a 2x scale error here,
    so the pipeline must report the scale as not identified and keep its status 'provisional' (never 'calibrated')."""
    cfg["synchronization"]["duration_tolerance_frac"] = 0.1
    cfg["auto_calibration"]["pair_translation_m"] = [0.05, 1.0]
    make_room_scan(tmp_path / "r", n=90, depth_scale=0.0005)
    with ScanDataset(tmp_path / "r", cfg) as ds:
        rep = select_conventions(ds, cfg)
    assert rep["scale_identified"] is False and rep["scale_sensitivity"] < 1.3
    c2 = apply_selection(copy.deepcopy(cfg), rep)
    assert c2["calibration"]["depth_scale"]["status"] == "provisional"
