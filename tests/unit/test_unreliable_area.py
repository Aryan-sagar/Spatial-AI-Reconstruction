"""An area that fails its quality checks must not be published as a tight measurement. SYNTHETIC: validates the mechanism."""
import numpy as np
from applied_ai.config import load_config
from applied_ai.geometry.floorplan import floor_footprint
from applied_ai.geometry.room_layout import extract_walls
from applied_ai.geometry.rooms import build_room
from applied_ai.output.renderer import render_plan
from applied_ai.output.schema import validate_scene
from applied_ai.uncertainty import UncertaintyModel
from synth_cloud import furnished_room

FC = {"floor": {"normal": np.array([0.0, 0.0, 1.0]), "d": 0.0, "support_count": 5000, "residual_std": 0.01}, "ceiling": None}


def _scene(cam_for_footprint=None):
    cfg = load_config()
    L, cam, gt = furnished_room(seed=0, rot_deg=18.0)
    wl = extract_walls(L, cam, 2.6, cfg)
    fp = floor_footprint(L, wl["final"], cfg, cam if cam_for_footprint is None else cam_for_footprint)
    room = build_room("room_01", wl["raw_walls"], wl["final"], [], fp, FC, 18.0, UncertaintyModel(cfg, "lidar"), [])
    scene = {"schema_version": "0.1", "capture_id": "t", "tier": "lidar", "rooms": [room], "adjacency": [], "measurements": [], "damages": [],
             "concealed_damage_flags": [], "scope_line_items": [], "diagnostics": {}, "provenance": {}}
    return scene, fp, gt


def test_reliable_footprint_keeps_observed_status_and_tight_interval():
    scene, fp, gt = _scene()
    a = scene["rooms"][0]["floor_area"]
    assert fp["area_reliable"] and a["status"] == "observed" and a.get("reliable", True) and scene["rooms"][0]["layout_reliable"]
    assert (a["confidence_interval"]["upper"] - a["confidence_interval"]["lower"]) / 2 < 0.1 * a["value"]
    assert validate_scene(scene) == []


def test_unreliable_footprint_gets_evidence_bounded_interval_and_flag():
    scene, fp, gt = _scene(cam_for_footprint=np.array([[40.0, 40.0], [41.0, 40.0]]))   # camera nowhere near the walls -> 0% inside
    assert not fp["area_reliable"] and any("camera positions" in r for r in fp["unreliable_reasons"])
    room = scene["rooms"][0]
    a = room["floor_area"]
    assert a["status"] == "estimated" and a["reliable"] is False and a["unreliable_reasons"] == fp["unreliable_reasons"]
    ci = a["confidence_interval"]
    assert ci["lower"] <= fp["floor_evidence_area_m2"] and ci["upper"] >= fp["wall_hull_area_m2"] and ci["lower"] <= a["value"] <= ci["upper"]
    assert ci["upper"] - ci["lower"] > 0.5            # visibly wider than any prior-based interval for this room
    assert room["layout_reliable"] is False and room["layout_unreliable_reasons"]
    assert validate_scene(scene) == []


def test_unreliable_plan_renders_with_visible_warning(tmp_path):
    scene, _, _ = _scene(cam_for_footprint=np.array([[40.0, 40.0], [41.0, 40.0]]))
    render_plan(scene, tmp_path / "plan.png", tmp_path / "plan.svg")
    assert (tmp_path / "plan.png").stat().st_size > 1000 and "AREA UNRELIABLE" in (tmp_path / "plan.svg").read_text()
