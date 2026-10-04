"""Multi-room segmentation of a whole-property scan, from observed free space and wall evidence (no assumption of a single room).

free space   = floor evidence (+ the camera path, which is always on the floor), closed over small holes
obstacles    = wall-band cells (tall vertical evidence), slightly dilated; they are removed from the free space, so a doorway is the
               narrow passage that remains between two basins
rooms        = h-maxima of the distance transform of the free space (one marker per basin at least `h_maxima_m` deeper than the saddle
               to its neighbour) grown by watershed; basins smaller than `min_room_area_m2` are dropped (reported, not silently merged)
connections  = pairs of rooms whose labels touch directly through free space (a doorway / open plan); `passage_width_m` = contact length
adjacency    = pairs separated only by a wall (touch after dilating by `adjacent_gap_m`); `connected` says whether a free passage exists
Areas are free-space areas inside the room (observed, obstacle-shrunk), NOT tape wall-to-wall; they are an estimate and are reported as such.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def _persistent_peaks(dist: np.ndarray, h: float, mask: np.ndarray) -> list[int]:
    """Flat indices of the peaks of `dist` (within `mask`) whose dynamic is >= h: a peak survives if, flooding from the top, the level at which it
    merges into a higher peak is at least h below it. The highest peak of each connected component always survives. Exact union-find version of
    h-maxima (no external dependency)."""
    H, W = dist.shape
    flat = dist.ravel()
    idx = np.flatnonzero(mask.ravel())
    order = idx[np.argsort(-flat[idx], kind="stable")]
    parent: dict[int, int] = {}
    peak: dict[int, tuple[float, int]] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out: list[int] = []
    for p in order.tolist():
        parent[p], peak[p] = p, (float(flat[p]), p)
        r, c = divmod(p, W)
        nbr = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < H and 0 <= cc < W and rr * W + cc in parent:
                    nbr.add(find(rr * W + cc))
        for q in nbr:
            a, b = find(p), q
            if a == b:
                continue
            win, lose = (a, b) if peak[a][0] >= peak[b][0] else (b, a)
            if peak[lose][0] - float(flat[p]) >= h:
                out.append(peak[lose][1])
            parent[lose] = win
    out.extend(peak[r][1] for r in {find(x) for x in parent})
    return out


def _flood_labels(dist: np.ndarray, core: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Marker-based watershed as a priority flood: the frontier pixel with the LARGEST distance-to-obstacle is claimed next, by the basin that
    pushed it. Basins therefore grow outward from their deep interiors along ridges before either reaches shallow pixels, so a room's margin
    follows the room itself rather than a neighbour that reaches it first through a doorway."""
    import heapq
    H, W = dist.shape
    lab = markers.astype(np.int32).copy()
    heap: list[tuple[float, int, int, int]] = []
    n = 0
    offs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    def push_neighbours(r: int, c: int, label: int) -> None:
        nonlocal n
        for dr, dc in offs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and core[rr, cc] and lab[rr, cc] == 0:
                n += 1
                heapq.heappush(heap, (-float(dist[rr, cc]), n, rr * W + cc, label))

    for r, c in zip(*np.nonzero(markers)):
        push_neighbours(int(r), int(c), int(markers[r, c]))
    while heap:
        _, _, p, label = heapq.heappop(heap)
        r, c = divmod(p, W)
        if lab[r, c]:
            continue
        lab[r, c] = label
        push_neighbours(r, c, label)
    return lab


def segment_rooms(floor_uv: np.ndarray, wall_cells: np.ndarray, cam_uv: np.ndarray | None, cfg: dict) -> dict:
    sc = cfg["room_segmentation"]
    res = sc["cell_m"]
    floor_uv, wall_cells = np.asarray(floor_uv, float).reshape(-1, 2), np.asarray(wall_cells, float).reshape(-1, 2)
    cam = np.asarray(cam_uv, float).reshape(-1, 2) if cam_uv is not None and len(cam_uv) else np.zeros((0, 2))
    if len(floor_uv) == 0:
        return {"rooms": [], "connections": [], "adjacent": [], "dropped_basins": [], "warnings": ["no floor evidence"]}
    allp = np.vstack([floor_uv, wall_cells, cam]) if len(wall_cells) else np.vstack([floor_uv, cam]) if len(cam) else floor_uv
    lo = allp.min(0) - 0.5
    shape = (np.ceil((allp.max(0) + 0.5 - lo) / res).astype(int))[::-1]

    def to_ij(p):
        return np.floor((np.asarray(p) - lo) / res).astype(int)

    def stamp(p):
        g = np.zeros(shape, np.uint8)
        ij = to_ij(p)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0])
        g[ij[ok, 1], ij[ok, 0]] = 1
        return g

    k = lambda m: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(round(m / res)) + 1,) * 2)
    free = stamp(floor_uv)
    if len(cam) > 1:   # the camera walked here, so it is floor (drawn as a ~0.4 m wide trail)
        trail = np.zeros(shape, np.uint8)
        pts = to_ij(cam).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(trail, [pts], False, 1, thickness=max(int(round(0.4 / res)), 1))
        free |= trail
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, k(sc["closing_m"]))
    obst = cv2.dilate(stamp(wall_cells), k(sc["obstacle_dilate_m"])) if len(wall_cells) else np.zeros(shape, np.uint8)
    free = ((free > 0) & (obst == 0))
    free = ndi.binary_fill_holes(free) & (obst == 0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] * res * res >= sc["min_free_component_m2"]]
    free = np.isin(lab, keep)
    if not free.any():
        return {"rooms": [], "connections": [], "adjacent": [], "dropped_basins": [], "warnings": ["no free space left after removing obstacles"]}
    dist = ndi.gaussian_filter(cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * res, sc["smooth_sigma_cells"])
    # Work on the CORE (free space at least core_min_dist_m from any obstacle): thin slivers hugging the walls would otherwise connect every room
    # to its neighbours around the perimeter and let one basin flood another's margin. The margin is given back afterwards by nearest label.
    core = free & (dist >= sc["core_min_dist_m"])
    markers = np.zeros(shape, np.int32)
    nm = 0
    for pk in _persistent_peaks(dist, sc["h_maxima_m"], core):
        nm += 1
        markers[divmod(pk, shape[1])] = nm
    ws = _flood_labels(dist, core, markers)
    ws[~core] = 0
    near, (ri, ci) = ndi.distance_transform_edt(ws == 0, return_indices=True)
    fill = free & (ws == 0) & (near <= sc["margin_fill_m"] / res)
    ws[fill] = ws[ri[fill], ci[fill]]
    # basins below the minimum area are dropped and reported
    rooms, dropped, relabel = [], [], np.zeros(nm + 2, int)
    warnings: list[str] = []
    for i in range(1, nm + 1):
        m = ws == i
        a = float(m.sum() * res * res)
        if a < sc["min_room_area_m2"]:
            if a > 0:
                dropped.append({"area_m2": a, "centroid": (np.argwhere(m).mean(0)[::-1] * res + lo + res / 2).tolist()})
            continue
        relabel[i] = len(rooms) + 1
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(cnt, sc["polygon_epsilon_m"] / res, True).reshape(-1, 2).astype(float) * res + lo + res / 2
        ij = to_ij(cam) if len(cam) else np.zeros((0, 2), int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0]) if len(cam) else np.zeros(0, bool)
        inside = float(m[ij[ok, 1], ij[ok, 0]].mean()) if ok.any() else 0.0
        rooms.append({"id": f"room_{len(rooms) + 1:02d}", "area_m2": a, "polygon": poly.tolist(), "centroid": (np.argwhere(m).mean(0)[::-1] * res + lo + res / 2).tolist(),
                      "camera_fraction_inside": inside, "peak_inscribed_radius_m": float(dist[m].max())})
    lab2 = relabel[ws]
    nr = len(rooms)
    # direct contact through free space (4-neighbour label changes) = open passage
    contact: dict[tuple[int, int], int] = {}
    for a, b in ((lab2[:, :-1], lab2[:, 1:]), (lab2[:-1, :], lab2[1:, :])):
        sel = (a != b) & (a > 0) & (b > 0)
        for x, y in zip(a[sel], b[sel]):
            key = (min(x, y), max(x, y))
            contact[key] = contact.get(key, 0) + 1
    connections = [{"a": rooms[i - 1]["id"], "b": rooms[j - 1]["id"], "passage_width_m": float(c * res)} for (i, j), c in sorted(contact.items())
                   if c * res >= sc["min_passage_m"]]
    cset = {(c["a"], c["b"]) for c in connections}
    adjacent = []
    r = k(sc["adjacent_gap_m"])
    for i in range(1, nr + 1):
        di = cv2.dilate((lab2 == i).astype(np.uint8), r)
        for j in range(i + 1, nr + 1):
            ov = float((di.astype(bool) & (lab2 == j)).sum() * res * res / max(sc["adjacent_gap_m"], 1e-9))
            if ov >= sc["min_shared_boundary_m"]:
                a_id, b_id = rooms[i - 1]["id"], rooms[j - 1]["id"]
                adjacent.append({"a": a_id, "b": b_id, "shared_boundary_m": ov, "connected": (a_id, b_id) in cset})
    if not rooms:
        warnings.append(f"no basin reached min_room_area_m2={sc['min_room_area_m2']}: free space is derived from floor evidence and camera trail, "
                        f"which are too sparse here ({float(free.sum() * res * res):.1f} m2 of free space)")
    return {"rooms": rooms, "connections": connections, "adjacent": adjacent, "dropped_basins": dropped, "warnings": warnings,
            "_labels": lab2, "_markers": markers, "_free": free, "_dist": dist, "_grid": {"origin": lo.tolist(), "res": res}, "_obstacles": obst > 0}
