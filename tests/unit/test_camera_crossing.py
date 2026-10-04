"""Camera-crossing wall filter. SYNTHETIC: validates the mechanism (a camera cannot pass through wall material; gaps/doorways are allowed),
not accuracy on real scans."""
import copy

import numpy as np
from applied_ai.config import load_config
from applied_ai.geometry.room_layout import extract_walls
from applied_ai.geometry.walls import drop_camera_crossed, wall_length
from synth_cloud import furnished_room


def _line(p, q, c=0.02):
    p, q = np.array(p, float), np.array(q, float)
    n = int(np.linalg.norm(q - p) / c) + 1
    return p + (q - p) * np.linspace(0, 1, n)[:, None]


def _cloud(cells, W, D, seed=0):
    rng = np.random.default_rng(seed)
    z = np.linspace(0.45, 1.55, 12)
    wall_pts = np.column_stack((np.repeat(cells, len(z), 0), np.tile(z, len(cells))))
    floor = np.column_stack((rng.uniform(0, W, 3000), rng.uniform(0, D, 3000), np.zeros(3000)))
    return np.vstack([wall_pts, floor])


def test_interior_lines_crossed_by_camera_are_dropped_perimeter_kept():
    W, D = 6.0, 3.7
    perim = [_line((0, 0), (W, 0)), _line((0, D), (W, D)), _line((0, 0), (0, D)), _line((W, 0), (W, D))]
    inner = [_line((1.5, .2), (1.5, 3.5)), _line((2.6, .2), (2.6, 3.5)), _line((4.0, .2), (4.0, 3.5)), _line((.3, 1.2), (5.7, 1.2)), _line((.3, 2.4), (5.7, 2.4))]
    L = _cloud(np.vstack(perim + inner), W, D)
    t = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    cam = np.vstack([np.column_stack((3.0 + 2.4 * np.cos(t), 1.85 + 1.4 * np.sin(t))), [[.8, .5], [5.2, 3.2], [1.0, 3.0], [5.0, .6]]])
    off, on = load_config(), load_config()
    off["walls"]["camera_crossing"]["enabled"] = False
    assert len(extract_walls(L, cam, None, off)["final"]) == 9          # 4 perimeter + 5 ghost lines
    wl = extract_walls(L, cam, None, on)
    assert sorted(round(wall_length(w), 1) for w in wl["final"]) == [3.7, 3.7, 6.0, 6.0]
    assert len(wl["prune_info"]["camera_crossing"]["dropped"]) == 5


def test_walking_through_a_doorway_does_not_delete_the_wall():
    L, _, gt = furnished_room(seed=0, rot_deg=0.0, shift=(0, 0), wardrobe=False, bed=False, table=False, screen=False, corridor=True)
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    loop = np.column_stack((2.95 + 1.9 * np.cos(th), 1.85 + 1.1 * np.sin(th)))
    out = np.array([[1.45, 1.0], [1.45, .2], [1.45, -.5], [1.45, -2.0], [1.45, -.5], [1.45, .5], [1.45, 1.0]])   # through the door gap and back
    wl = extract_walls(L, np.vstack([loop, out]), 2.6, load_config())
    lens = sorted(round(wall_length(w), 1) for w in wl["final"])
    assert lens.count(5.9) == 2 and lens.count(3.7) == 2                  # both long walls (incl. the one with the door) and both short walls survive
    assert wl["prune_info"]["camera_crossing"]["dropped"] == []


def test_disabled_filter_is_a_no_op():
    cfg = load_config()
    cfg["walls"]["camera_crossing"]["enabled"] = False
    walls = [{"id": "w", "normal": np.array([0., 1.]), "d": 0.0, "direction": np.array([1., 0.]), "centroid": np.zeros(2), "s_min": 0.0, "s_max": 3.0}]
    kept, info = drop_camera_crossed(walls, np.array([[1., -1.], [1., 1.]]), np.zeros((0, 2)), cfg)
    assert kept == walls and info["dropped"] == []
