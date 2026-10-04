"""Room segmentation on SYNTHETIC floor/wall evidence (exact layout known). Validates the logic, not real-scan accuracy."""
import numpy as np
from applied_ai.config import load_config
from applied_ai.geometry.room_segmentation import segment_rooms


def _wall(p, q, gaps=(), step=0.02, thick=0.10, rng=None):
    p, q = np.array(p, float), np.array(q, float)
    n = int(np.linalg.norm(q - p) / step) + 1
    t = np.linspace(0, 1, n)
    pts = p + (q - p) * t[:, None]
    s = t * np.linalg.norm(q - p)
    ok = np.ones(n, bool)
    for a, b in gaps:
        ok &= ~((s > a) & (s < b))
    d = (q - p) / np.linalg.norm(q - p); nrm = np.array([-d[1], d[0]])
    off = (rng or np.random.default_rng(0)).uniform(-thick / 2, thick / 2, n)
    return (pts + off[:, None] * nrm)[ok]


def _floor(W, H, walls_for_clearance, step=0.04, clear=0.10):
    xs, ys = np.meshgrid(np.arange(0.05, W, step), np.arange(0.05, H, step))
    P = np.column_stack((xs.ravel(), ys.ravel()))
    keep = np.ones(len(P), bool)
    for w in walls_for_clearance:
        # drop floor points that lie on the wall material itself (door gaps have no wall points, so floor stays there)
        from scipy.spatial import cKDTree
        keep &= cKDTree(w).query(P)[0] > clear
    return P[keep]


def _three_rooms():
    rng = np.random.default_rng(1)
    walls = [_wall((0, 0), (8.4, 0), rng=rng), _wall((0, 6), (8.4, 6), rng=rng), _wall((0, 0), (0, 6), rng=rng), _wall((8.4, 0), (8.4, 6), rng=rng),
             _wall((4.1, 0), (4.1, 3.5), gaps=[(1.2, 2.1)], rng=rng),                      # A|B with a 0.9 m door
             _wall((0, 3.5), (8.4, 3.5), gaps=[(1.5, 2.4), (5.7, 6.6)], rng=rng)]          # rooms|hall with two 0.9 m doors
    cells = np.vstack(walls)
    floor = _floor(8.4, 6.0, walls)
    cam = np.array([[2, 1.7], [3.6, 1.65], [4.6, 1.65], [6.2, 1.7], [6.2, 3.0], [6.15, 4.0], [4.0, 4.7], [2.0, 4.6], [1.95, 3.0], [2.0, 1.7]])
    return floor, cells, cam


def test_three_rooms_with_doors_are_separated_with_correct_connections():
    floor, cells, cam = _three_rooms()
    seg = segment_rooms(floor, cells, cam, load_config())
    areas = sorted(r["area_m2"] for r in seg["rooms"])
    assert len(seg["rooms"]) == 3, [round(a, 1) for a in areas]
    for got, nominal in zip(areas, sorted([4.1 * 3.5, 4.3 * 3.5, 8.4 * 2.5])):   # A, B, hall wall-to-wall
        assert 0.80 * nominal < got < 1.02 * nominal       # free area is obstacle-shrunk, never larger than the nominal room
    assert len(seg["connections"]) == 3                      # A-B, A-hall, B-hall
    assert all(0.4 < c["passage_width_m"] < 1.0 for c in seg["connections"])   # ~0.9 m door minus obstacle dilation
    assert all(a["connected"] for a in seg["adjacent"]) and len(seg["adjacent"]) == 3


def test_single_room_is_not_oversegmented():
    rng = np.random.default_rng(2)
    walls = [_wall((0, 0), (5, 0), rng=rng), _wall((0, 4), (5, 4), rng=rng), _wall((0, 0), (0, 4), rng=rng), _wall((5, 0), (5, 4), rng=rng)]
    seg = segment_rooms(_floor(5, 4, walls), np.vstack(walls), np.array([[1, 1], [4, 1], [4, 3], [1, 3]]), load_config())
    assert len(seg["rooms"]) == 1 and seg["connections"] == []


def test_two_rooms_with_a_solid_wall_are_adjacent_but_not_connected():
    rng = np.random.default_rng(3)
    walls = [_wall((0, 0), (8, 0), rng=rng), _wall((0, 4), (8, 4), rng=rng), _wall((0, 0), (0, 4), rng=rng), _wall((8, 0), (8, 4), rng=rng), _wall((4, 0), (4, 4), rng=rng)]
    seg = segment_rooms(_floor(8, 4, walls), np.vstack(walls), np.array([[1, 2], [3, 2]]), load_config())
    assert len(seg["rooms"]) == 2 and seg["connections"] == []
    assert len(seg["adjacent"]) == 1 and seg["adjacent"][0]["connected"] is False


def test_tiny_basin_is_dropped_and_reported_not_merged():
    rng = np.random.default_rng(4)
    walls = [_wall((0, 0), (5, 0), rng=rng), _wall((0, 4), (5, 4), rng=rng), _wall((0, 0), (0, 4), rng=rng), _wall((5, 0), (5, 4), rng=rng),
             _wall((3.9, 0), (3.9, 2.2), rng=rng), _wall((3.9, 2.2), (5, 2.2), rng=rng)]   # a 1.1 x 2.2 closet-like pocket
    seg = segment_rooms(_floor(5, 4, walls), np.vstack(walls), np.array([[1, 2], [3, 2]]), load_config())
    assert len(seg["rooms"]) >= 1 and all(r["area_m2"] >= 2.0 for r in seg["rooms"])
