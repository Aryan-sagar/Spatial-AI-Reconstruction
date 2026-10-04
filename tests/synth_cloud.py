"""Furnished-room point cloud in floor-local coordinates (z up, floor at 0) with exactly known geometry, for testing the WALL/FOOTPRINT
stage on the failure modes seen in the first real scan: a room rotated against the grid, a corridor seen through a door, a tall wardrobe
hiding a stretch of wall, a table (horizontal blob), a low bed, a thin free-standing off-axis screen, patchy floor evidence.
Validates logic only; it is not a model of real sensor noise or of any real room. Never used by the pipeline."""
from __future__ import annotations

import numpy as np

DENS = 1 / 0.03 ** 2  # points per m2 (about a 3 cm voxel cloud)


def _rect(rng, o, a, b, dens=DENS):
    """Random points on the parallelogram o + s*a + t*b."""
    area = np.linalg.norm(np.cross(a, b)) if len(a) == 3 else abs(a[0] * b[1] - a[1] * b[0])
    n = max(int(area * dens), 4)
    return np.asarray(o, float) + rng.random((n, 1)) * np.asarray(a, float) + rng.random((n, 1)) * np.asarray(b, float)


def furnished_room(seed=0, W=5.9, D=3.7, H=2.6, rot_deg=18.0, noise=0.008, wardrobe=True, bed=True, table=True, screen=True, corridor=True,
                   shift=(-1.0, -1.5)):
    rng = np.random.default_rng(seed)
    P = []

    def vwall(p0, p1, z0=0.0, z1=H, cut=None, dens=DENS):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        d = p1 - p0
        pts = _rect(rng, [p0[0], p0[1], z0], [d[0], d[1], 0], [0, 0, z1 - z0], dens)
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        pts[:, :2] += rng.normal(0, noise, (len(pts), 1)) * nrm
        if cut is not None:
            pts = pts[~cut(pts)]
        return pts

    door = lambda p: (p[:, 0] > 1.0) & (p[:, 0] < 1.9) & (p[:, 2] < 2.0)        # door in the south wall (y=0)
    hidden = lambda p: (p[:, 0] > 2.0) & (p[:, 0] < 3.8)                          # wall behind the wardrobe (north wall) is not observed
    P += [vwall((0, 0), (W, 0), cut=door if corridor else None), vwall((0, D), (W, D), cut=hidden if wardrobe else None),
          vwall((0, 0), (0, D)), vwall((W, 0), (W, D))]
    floor_parts = [(_rect(rng, [0, 0, 0], [W, 0, 0], [0, D, 0], DENS / 2))]
    occl = []
    if wardrobe:
        P += [vwall((2.0, D - 0.6), (3.8, D - 0.6), 0, 2.0), vwall((2.0, D - 0.6), (2.0, D), 0, 2.0), vwall((3.8, D - 0.6), (3.8, D), 0, 2.0)]
        occl.append((2.0, 3.8, D - 0.6, D))
    if bed:  # 0.55 m high: front face + top surface (a horizontal blob in the top-down cell map)
        P += [vwall((W - 0.9, 0.8), (W - 0.9, 2.8), 0, 0.55), vwall((W - 0.9, 0.8), (W, 0.8), 0, 0.55), vwall((W - 0.9, 2.8), (W, 2.8), 0, 0.55),
              _rect(rng, [W - 0.9, 0.8, 0.55], [0.9, 0, 0], [0, 2.0, 0])]
        occl.append((W - 0.9, W, 0.8, 2.8))
    if table:  # 0.75 m: top surface blob + 4 thin legs
        P += [_rect(rng, [2.2, 1.2, 0.75], [1.2, 0, 0], [0, 0.8, 0])]
        for (lx, ly) in ((2.25, 1.25), (3.35, 1.25), (2.25, 1.95), (3.35, 1.95)):
            P += [np.column_stack((lx + rng.normal(0, 0.01, 40), ly + rng.normal(0, 0.01, 40), rng.random(40) * 0.75))]
        occl.append((2.2, 3.4, 1.2, 2.0))
    if screen:  # thin tall panel, 1.5 m long, 35 deg off the room axes (a folding screen / ladder: passes any height filter)
        c, a = np.array([1.7, 2.6]), np.radians(-35.0)
        t = np.array([np.cos(a), np.sin(a)])
        P += [vwall(c - 0.75 * t, c + 0.75 * t, 0, 1.8)]
    if corridor:  # space beyond the door, seen through it
        P += [vwall((1.0, 0), (1.0, -3.0)), vwall((1.9, 0), (1.9, -3.0)), vwall((1.0, -3.0), (1.9, -3.0))]
        floor_parts.append(_rect(rng, [1.0, -3.0, 0], [0.9, 0, 0], [0, 3.0, 0], DENS / 2))
    F = np.vstack(floor_parts)
    keep = np.ones(len(F), bool)
    for (x0, x1, y0, y1) in occl:  # floor under furniture is not observed
        keep &= ~((F[:, 0] > x0) & (F[:, 0] < x1) & (F[:, 1] > y0) & (F[:, 1] < y1))
    F = F[keep]
    F[:, 2] = rng.normal(0, 0.005, len(F))
    P.append(F)
    L = np.vstack(P)
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    L[:, :2] = L[:, :2] @ R.T + np.asarray(shift)
    # camera path: loop around the middle of the room, clear of the furniture
    th = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    cam = np.column_stack((2.95 + 1.9 * np.cos(th), 1.85 + 1.1 * np.sin(th))) @ R.T + np.asarray(shift)
    gt = {"W": W, "D": D, "H": H, "area": W * D, "rot_deg": rot_deg % 90.0, "wall_lengths": sorted([W, W, D, D]),
          "n_true_walls": 4}
    return L, cam, gt
