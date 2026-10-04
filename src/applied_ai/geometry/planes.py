"""Vectorized RANSAC plane fitting (numpy). Planes: n . p + d = 0 with unit n."""
from __future__ import annotations

import numpy as np


def fit_plane_svd(P: np.ndarray):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    return n, float(-n @ c)


def plane_distance(P: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """Signed distance of points to plane."""
    return P @ n + d


def ransac_plane(P: np.ndarray, thresh: float, iters: int, rng: np.random.Generator, subsample: int = 40000,
                 normal_hint: np.ndarray | None = None, max_angle_deg: float | None = None, min_inliers: int = 50):
    """Returns (n, d, inlier_mask) or None. Hypotheses can be restricted to normals within max_angle_deg of normal_hint."""
    if len(P) < max(min_inliers, 3):
        return None
    sub = P if len(P) <= subsample else P[rng.choice(len(P), subsample, replace=False)]
    sub = sub.astype(np.float64)
    cos_lim = np.cos(np.radians(max_angle_deg)) if (normal_hint is not None and max_angle_deg is not None) else None
    best_n, best_d, best_cnt = None, 0.0, -1
    done, chunk = 0, 200
    while done < iters:
        c = min(chunk, iters - done)
        idx = rng.integers(0, len(sub), (c, 3))
        a, b, cc = sub[idx[:, 0]], sub[idx[:, 1]], sub[idx[:, 2]]
        n = np.cross(b - a, cc - a)
        nn = np.linalg.norm(n, axis=1)
        ok = nn > 1e-9
        n[ok] /= nn[ok, None]
        if cos_lim is not None:
            ok &= np.abs(n @ normal_hint) >= cos_lim
        if ok.any():
            d = -np.einsum("ij,ij->i", n, a)
            dist = np.abs(sub @ n.T + d)  # N x c
            cnt = (dist < thresh).sum(0)
            cnt[~ok] = -1
            k = int(np.argmax(cnt))
            if cnt[k] > best_cnt:
                best_cnt, best_n, best_d = int(cnt[k]), n[k].copy(), float(d[k])
        done += c
    if best_n is None:
        return None
    mask = np.abs(P @ best_n + best_d) < thresh
    if mask.sum() < min_inliers:
        return None
    for _ in range(2):  # least-squares refinement
        n, d = fit_plane_svd(P[mask].astype(np.float64))
        if normal_hint is not None and n @ best_n < 0:
            n, d = -n, -d
        mask = np.abs(P @ n + d) < thresh
        best_n, best_d = n, d
    return best_n, best_d, mask
