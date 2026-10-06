import numpy as np
from applied_ai.config import load_config
from applied_ai.geometry.walls import endpoints, trim_overshoot, wall_length


def _wall(i, p, q):
    p, q = np.array(p, float), np.array(q, float)
    d = (q - p) / np.linalg.norm(q - p)
    n = np.array([-d[1], d[0]])
    c = (p + q) / 2
    return {"id": f"w{i}", "normal": n, "d": float(-n @ p), "direction": d, "centroid": c, "s_min": -np.linalg.norm(q - p) / 2, "s_max": np.linalg.norm(q - p) / 2}


def test_overshoot_past_a_perpendicular_wall_is_cut_to_the_corner():
    cfg = load_config()
    top = _wall(0, (0, 4.0), (3.3, 4.0))
    right = _wall(1, (3.3, -0.2), (3.3, 4.65))       # runs 0.65 m past the top wall
    bottom = _wall(2, (0, 0), (3.6, 0))              # runs 0.3 m past the right wall
    left = _wall(3, (0, 0), (0, 4.0))
    out, log = trim_overshoot([top, right, bottom, left], cfg, np.array([[1.6, 2.0]]))
    L = {w["id"]: wall_length(w) for w in out}
    assert abs(L["w1"] - 4.0) < 1e-6 and abs(L["w2"] - 3.3) < 1e-6 and abs(L["w0"] - 3.3) < 1e-6 and abs(L["w3"] - 4.0) < 1e-6
    assert {e["id"] for e in log} == {"w1", "w2"}


def test_a_long_run_past_a_partition_is_left_alone():
    cfg = load_config()
    through = _wall(0, (0, 0), (8.0, 0))
    partition = _wall(1, (4.0, -0.1), (4.0, 3.0))     # crosses the through wall in the middle: 4 m on either side, far above max_m
    out, log = trim_overshoot([through, partition], cfg)
    assert abs(wall_length(out[0]) - 8.0) < 1e-6 and log == []


def test_disabled_is_a_no_op_and_nothing_is_ever_extended():
    cfg = load_config()
    cfg["walls"]["trim_overshoot"]["enabled"] = False
    a, b = _wall(0, (0, 4.0), (3.3, 4.0)), _wall(1, (3.3, -0.2), (3.3, 4.65))
    out, log = trim_overshoot([a, b], cfg)
    assert out == [a, b] and log == []
    cfg["walls"]["trim_overshoot"]["enabled"] = True
    short = _wall(2, (0, 4.0), (3.0, 4.0))            # stops 0.3 m short of the right wall: must NOT be extended
    out, _ = trim_overshoot([short, b], cfg)
    assert abs(wall_length(out[0]) - 3.0) < 1e-6


def test_a_corridor_wall_touching_the_line_from_outside_is_not_a_corner():
    cfg = load_config()
    south = _wall(0, (0, 0), (5.9, 0))
    corridor = _wall(1, (1.0, 0), (1.0, -3.0))        # starts at the south wall's line and goes AWAY from the room (cameras are north of it)
    west = _wall(2, (0, 0), (0, 3.7))
    out, log = trim_overshoot([south, corridor, west], cfg, np.array([[3.0, 1.8]]))
    assert abs(wall_length(out[0]) - 5.9) < 1e-6 and all(e["id"] != "w0" for e in log)


def test_a_stub_that_meets_a_wall_at_its_far_end_is_not_cut_to_nothing():
    """Seen on room A: a 0.97 m stub ending at the bottom wall was trimmed to 0.00 m because the crossing was 0.97 m from its OTHER end."""
    cfg = load_config()
    bottom = _wall(0, (0, 0), (3.3, 0))
    stub = _wall(1, (0.4, 0.97), (0.4, 0.0))          # 0.97 m, ends on the bottom wall
    left = _wall(2, (0, 0), (0, 4.0))
    out, log = trim_overshoot([bottom, stub, left], cfg, np.array([[1.6, 2.0]]))
    assert abs(wall_length(out[1]) - 0.97) < 1e-6 and all(e["id"] != "w1" for e in log)
