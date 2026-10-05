"""Visibility carving on a SYNTHETIC floor plan with exact walls: depth rays are cast in 2-D against known wall segments (with door gaps)."""
import numpy as np
from applied_ai.config import load_config
from applied_ai.geometry.room_segmentation import segment_rooms
from applied_ai.geometry.visibility import frame_free_polygons

W, H = 8.4, 6.0
# (p, q) wall segments; doors are gaps between segments
SEGS = [((0, 0), (W, 0)), ((0, H), (W, H)), ((0, 0), (0, H)), ((W, 0), (W, H)),
        ((4.1, 0), (4.1, 1.2)), ((4.1, 2.1), (4.1, 3.5)),
        ((0, 3.5), (1.5, 3.5)), ((2.4, 3.5), (5.7, 3.5)), ((6.6, 3.5), (W, 3.5))]


def _cast(origin, bearing, max_r=6.0):
    o = np.array(origin, float); d = np.array([np.cos(bearing), np.sin(bearing)])
    best = None
    for p, q in SEGS:
        p, q = np.array(p, float), np.array(q, float); e = q - p
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-12:
            continue
        t = ((p[0] - o[0]) * e[1] - (p[1] - o[1]) * e[0]) / den
        u = ((p[0] - o[0]) * d[1] - (p[1] - o[1]) * d[0]) / den
        if t > 0.05 and 0 <= u <= 1 and (best is None or t < best):
            best = t
    return best if best is not None and best <= max_r else None


def _frame(origin, facing_deg, fov_deg=70, step_deg=0.4, rng=None):
    rng = rng or np.random.default_rng(0)
    pts = []
    for b in np.radians(np.arange(facing_deg - fov_deg / 2, facing_deg + fov_deg / 2, step_deg)):
        t = _cast(origin, b)
        if t is None:
            continue
        for h in rng.uniform(0.3, 1.7, 3):   # three wall-band points per ray
            pts.append((origin[0] + (t + rng.normal(0, 0.01)) * np.cos(b), origin[1] + (t + rng.normal(0, 0.01)) * np.sin(b), h))
    return np.array(pts)


def _walk():
    """Camera positions and the directions it looks while turning on the spot: it never needs to see the floor."""
    stops = [(2.0, 1.7), (3.3, 1.7), (6.2, 1.7), (2.0, 4.7), (4.2, 4.7), (6.3, 4.7), (1.95, 3.5), (6.15, 3.5), (4.1, 1.65)]
    return [(s, f) for s in stops for f in range(0, 360, 40)]


def test_single_frame_polygon_stops_before_the_wall():
    cfg = load_config()
    cam = (4.5, 1.7)   # looking +y at the SOLID part of the wall y=3.5 (x 2.4-5.7), 1.8 m away; a narrow view so no door gap is in sight
    polys = frame_free_polygons(np.array(cam), _frame(cam, 90, fov_deg=40), cfg)
    assert len(polys) == 1
    far = polys[0][1:]
    assert far[:, 1].max() < 3.5 - 0.05                              # carved short of the surface by the margin
    assert far[:, 1].max() > 3.5 - 0.35


def test_carved_free_space_gives_three_rooms_without_any_floor_evidence():
    cfg = load_config()
    polys = []
    for cam, face in _walk():
        polys += frame_free_polygons(np.array(cam), _frame(cam, face), cfg)
    # wall evidence only (what the depth points in the wall band show), NO floor points at all
    rng = np.random.default_rng(5); cells = []
    for p, q in SEGS:
        p, q = np.array(p), np.array(q); n = int(np.linalg.norm(q - p) / 0.02) + 1
        e = (q - p) / np.linalg.norm(q - p); nrm = np.array([-e[1], e[0]])
        t = np.linspace(0, 1, n); cells.append(p + (q - p) * t[:, None] + rng.normal(0, 0.01, (n, 1)) * nrm)   # scatter across the wall thickness
    seg = segment_rooms(np.zeros((0, 2)) + np.array([[2.0, 1.7]]), np.vstack(cells), np.array([c for c, _ in _walk()]), cfg, free_polys=polys)
    areas = sorted(r["area_m2"] for r in seg["rooms"])
    assert len(seg["rooms"]) == 3, [round(a, 1) for a in areas]
    for got, nominal in zip(areas, sorted([4.1 * 3.5, 4.3 * 3.5, 8.4 * 2.5])):
        assert 0.7 * nominal < got < 1.02 * nominal
    assert len(seg["connections"]) >= 2          # the doors seen through are open passages
