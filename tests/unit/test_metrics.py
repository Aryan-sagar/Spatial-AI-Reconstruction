import numpy as np
import pytest
from applied_ai.evaluation.metrics import abs_err_cm, calibration_from_residuals, evaluate_scene, match, pct_error, spread


def _scene(tier="lidar", ceil=2.60, widths=(1.0, 0.9), walls=(4.0, 3.0)):
    mk = lambda v: {"value": v, "confidence_interval": {"lower": v - 0.05, "upper": v + 0.05, "confidence_level": 0.95}}
    return {"capture_id": "x", "tier": tier, "rooms": [{"ceiling_height": mk(ceil) if ceil else {"value": None}, "walls": [{"length": mk(w)} for w in walls],
                                                       "openings": [{"width": mk(w)} for w in widths], "floor_area": mk(12.0)}]}


def test_basic_metrics():
    assert pct_error(4.1, 4.0) == pytest.approx(2.5) and abs_err_cm(2.61, 2.60) == pytest.approx(1.0)
    assert spread(2.60, 2.61)[0] == pytest.approx(0.01)


def test_gate_calculation_opening_and_ceiling():
    res = evaluate_scene(_scene(widths=(1.01, 0.93)), {"ceiling_height_m": 2.60, "opening_widths_m": [1.0, 0.9]})
    g = {x["gate_name"]: x for x in res["gates"]}
    assert g["ceiling_height"]["pass"] and not g["opening_width"]["pass"]  # 1.0 of 2 within 2 cm = 50%
    assert "1/2" in g["opening_width"]["actual"]


def test_missed_opening_counts_as_failure_not_exclusion():
    res = evaluate_scene(_scene(widths=(1.0,)), {"opening_widths_m": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]})
    g = res["gates"][0]
    assert not g["pass"] and "1/7" in g["actual"]


def test_unobserved_ceiling_fails_gate_rather_than_vanishing():
    res = evaluate_scene(_scene(ceil=None), {"ceiling_height_m": 2.6})
    assert res["gates"][0]["pass"] is False and res["gates"][0]["actual"] == "unobserved"


def test_matching_is_optimal_and_reports_unmatched():
    pairs, up, ug = match([4.0, 3.0, 9.0], [3.05, 4.02])
    assert sorted(pairs) == [(0, 1), (1, 0)] and up == [2] and ug == []


def test_repeatability_gate():
    a, b = _scene(ceil=2.600), _scene(ceil=2.612)
    g = {x["gate_name"]: x for x in evaluate_scene(a, {"ceiling_height_m": 2.6}, b)["gates"]}
    assert g["repeated_ceiling_spread"]["pass"] is False  # 1.2 cm > 1 cm


def test_calibration_quantiles_are_stratified_by_tier():
    r1 = evaluate_scene(_scene("lidar"), {"ceiling_height_m": 2.61})
    r2 = evaluate_scene(_scene("photo", ceil=2.70), {"ceiling_height_m": 2.61})
    cal = calibration_from_residuals([r1, r2])
    assert set(cal) == {"lidar", "photo"} and cal["photo"]["ceiling_height"]["quantile"] > cal["lidar"]["ceiling_height"]["quantile"]
