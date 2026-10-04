"""Gravity/up estimation (from poses, verified by plane geometry) and floor-aligned coordinates."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..reconstruction.poses import AXES, T_world_cvcamera
from .planes import ransac_plane


class UpAxisError(RuntimeError):
    """Floor/up estimation failed; `.diag` carries evidence so the failure is debuggable."""

    def __init__(self, msg, diag=None):
        super().__init__(msg)
        self.diag = diag or {}


@dataclass
class FloorCoordinateSystem:
    origin: np.ndarray
    normal: np.ndarray  # points up, away from floor
    axis_u: np.ndarray
    axis_v: np.ndarray

    @classmethod
    def from_plane(cls, normal, point_on_plane):
        n = normal / np.linalg.norm(normal)
        ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(n, ref); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        return cls(np.asarray(point_on_plane, float), n, u, v)

    def to_local(self, P: np.ndarray) -> np.ndarray:
        """(u, v, h) with h = height above the floor plane."""
        Q = np.asarray(P, np.float64) - self.origin
        return np.column_stack((Q @ self.axis_u, Q @ self.axis_v, Q @ self.normal))

    def to_world(self, L: np.ndarray) -> np.ndarray:
        return self.origin + L[:, 0:1] * self.axis_u + L[:, 1:2] * self.axis_v + L[:, 2:3] * self.normal


def camera_up_hint(df, frame_ids, cfg) -> np.ndarray:
    """Mean world direction of the camera's image-up axis over the given frames (weak prior only)."""
    ups = []
    sub = df.set_index("frame").loc[frame_ids]
    for _, r in sub.iterrows():
        T = T_world_cvcamera((r.x, r.y, r.z), (r.qx, r.qy, r.qz, r.qw), cfg["pose"]["convention"], cfg["pose"]["camera_axes"])
        ups.append(T[:3, :3] @ np.array([0.0, -1.0, 0.0]))
    m = np.mean(ups, axis=0)
    if np.linalg.norm(m) < 1e-6:
        raise RuntimeError("camera up-vector average is degenerate; cannot form an up hint")
    return m / np.linalg.norm(m)


def _proper_signed_permutations() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, sg) in enumerate(zip(perm, signs)):
                M[i, p] = sg
            if np.linalg.det(M) > 0:
                mats.append(M)
    return mats  # the 24 proper rotations


def gravity_up_from_imu(imu_df, odo_df, cfg: dict, max_samples: int = 3000) -> tuple[np.ndarray | None, dict]:
    """World 'up' from the IMU. The device->camera axis mapping is unknown, so all 24 axis permutations are tried; the right one
    makes (pose rotation x mapping x accel) constant over time. Returns (up or None if unreliable, diagnostics).
    ASSUMES the accelerometer reading points toward gravity (iOS CoreMotion convention); set planes.imu_reading_direction otherwise."""
    pc = cfg["planes"]
    t_o, t_i = odo_df["timestamp"].to_numpy(float), imu_df["timestamp"].to_numpy(float)
    ok = np.flatnonzero((t_i >= t_o[0]) & (t_i <= t_o[-1]))
    if len(ok) < 50:
        return None, {"reliable": False, "reason": f"only {len(ok)} IMU samples overlap the odometry time range"}
    ok = ok[np.linspace(0, len(ok) - 1, min(max_samples, len(ok))).astype(int)]
    a = imu_df[["a_x", "a_y", "a_z"]].to_numpy(float)[ok]
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    j = np.clip(np.searchsorted(t_o, t_i[ok]), 1, len(t_o) - 1)
    j = np.where(np.abs(t_o[j] - t_i[ok]) < np.abs(t_o[j - 1] - t_i[ok]), j, j - 1)
    R = Rotation.from_quat(odo_df[["qx", "qy", "qz", "qw"]].to_numpy(float)[j]).as_matrix()
    if cfg["pose"]["convention"] == "world_to_camera":
        R = R.transpose(0, 2, 1)
    R = R @ AXES[cfg["pose"]["camera_axes"]]
    def resultant(M):
        m = np.einsum("nij,jk,nk->ni", R, M, a).mean(0)
        return float(np.linalg.norm(m)), m / max(np.linalg.norm(m), 1e-12)

    prior = np.asarray(pc["imu_device_to_camera"], float)
    r_p, m_p = resultant(prior)
    diag_p = {"prior_resultant": r_p, "samples": int(len(ok)), "reading_convention": pc["imu_reading_direction"]}
    if r_p >= pc["imu_min_resultant"]:
        diag_p.update(reliable=True, mapping_source="prior (iPhone sensor is landscape-right; documented in docs/assumptions.md)",
                      device_to_camera_mapping=prior.tolist(), reading_direction_world=m_p.tolist())
        return (-m_p if pc["imu_reading_direction"] == "toward_gravity" else m_p), diag_p
    res = []
    for M in _proper_signed_permutations():
        w = np.einsum("nij,jk,nk->ni", R, M, a)
        m = w.mean(0)
        res.append((float(np.linalg.norm(m)), m / max(np.linalg.norm(m), 1e-12), M))
    res.sort(key=lambda r: -r[0])
    best, second = res[0], res[1]
    diag = {**diag_p, "mapping_source": "searched (prior mapping did not fit)", "resultant_best": best[0], "resultant_second": second[0],
            "device_to_camera_mapping": best[2].tolist(), "reading_direction_world": best[1].tolist()}
    reliable = best[0] >= pc["imu_min_resultant"] and best[0] - second[0] >= pc["imu_min_margin"]
    diag["reliable"] = bool(reliable)
    if reliable and abs(res[1][0] - best[0]) < 0.02:
        reliable = diag["reliable"] = False  # mirror ambiguity: a different mapping fits equally well, up vs down undecidable
    if not reliable:
        diag["reason"] = "no device->camera mapping makes the world-frame gravity direction constant (IMU/pose clocks, units or conventions differ)"
        return None, diag
    up = -best[1] if pc["imu_reading_direction"] == "toward_gravity" else best[1]
    return up, diag


def trajectory_up_hint(df, frame_ids, cfg, sign_ref: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Up direction from the camera trajectory itself: a handheld walk is nearly planar, so the direction of least positional
    variance is ~vertical. Sign comes from the (weak) camera-up average. Independent of camera-axes assumptions except for the sign."""
    sub = df.set_index("frame").loc[frame_ids]
    P = sub[["x", "y", "z"]].to_numpy(float)
    C = np.cov((P - P.mean(0)).T)
    w, v = np.linalg.eigh(C)  # ascending
    n = v[:, 0]
    src = "imu_gravity"
    if sign_ref is None:
        sign_ref, src = camera_up_hint(df, frame_ids, cfg), "camera_up_average (WEAK: sensor image-up need not be world-up)"
    if n @ sign_ref < 0:
        n = -n
    return n, {"trajectory_std_m": np.sqrt(np.maximum(w, 0))[::-1].tolist(), "sign_source": src,
               "angle_to_sign_reference_deg": float(np.degrees(np.arccos(np.clip(n @ sign_ref, -1, 1))))}


def _peaks(h, bin_m, min_frac):
    lo, hi = float(h.min()), float(h.max())
    nb = max(int(np.ceil((hi - lo) / bin_m)), 1)
    cnt, edges = np.histogram(h, bins=nb, range=(lo, lo + nb * bin_m))
    sm = np.convolve(cnt, np.ones(3) / 3, mode="same")
    mx = sm.max()
    idx = [i for i in range(len(sm)) if sm[i] >= min_frac * mx and sm[i] >= (sm[i - 1] if i else -1) and sm[i] >= (sm[i + 1] if i + 1 < len(sm) else -1)]
    # merge plateau duplicates closer than 5 bins, keep the larger
    merged = []
    for i in idx:
        if merged and i - merged[-1] < 5:
            if sm[i] > sm[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    out = []
    for i in merged:
        k = int(round(0.16 / bin_m)), int(round(0.40 / bin_m))
        side = [sm[max(i - k[1], 0):max(i - k[0], 0)], sm[min(i + k[0], len(sm)):min(i + k[1], len(sm))]]
        side = [x for x in side if len(x)]
        nb = float(np.median(np.concatenate(side))) if side else 0.0
        out.append((float(edges[i] + bin_m / 2), float(sm[i]), float(sm[i] / max(nb, 1.0))))
    return out, float(mx)


def find_floor_ceiling(P: np.ndarray, up_hint: np.ndarray, cfg: dict, cam_positions: np.ndarray | None = None) -> dict:
    pc = cfg["planes"]
    rng = np.random.default_rng(pc["seed"])
    thr = pc["distance_threshold_m"]
    diag: dict = {"warnings": []}
    diag['up_hint'] = up_hint.tolist()
    r0 = ransac_plane(P, thr, pc["ransac_iterations"], rng, pc["ransac_subsample"], up_hint, pc["up_hint_max_deg"])
    if r0 is None:
        raise RuntimeError("no near-horizontal plane found within up_hint_max_deg of the pose-derived up direction")
    n0, d0, m0 = r0
    up = n0 if n0 @ up_hint > 0 else -n0
    diag["up_hint_vs_plane_deg"] = float(np.degrees(np.arccos(np.clip(abs(n0 @ up_hint), 0, 1))))
    for _ in range(2):  # re-fit the dominant horizontal plane around the refined up, tightening the cone
        r1 = ransac_plane(P, thr, pc["ransac_iterations"], rng, pc["ransac_subsample"], up, 8.0)
        if r1 is None:
            break
        up = r1[0] if r1[0] @ up > 0 else -r1[0]
    diag["up_refined_from_hint_deg"] = float(np.degrees(np.arccos(np.clip(up @ up_hint, -1, 1))))
    h = P @ up
    peaks, mx = _peaks(h, pc["histogram_bin_m"], pc["peak_min_frac"])
    diag["height_peaks"] = [{"height": a, "count": b, "prominence": c} for a, b, c in peaks]
    if not peaks:
        raise UpAxisError("no height-histogram peaks", diag)
    if cam_positions is not None:
        hcam = float(np.median(cam_positions @ up))
        diag["camera_height_along_up"] = hcam
        diag["height_peaks_relative_to_camera"] = [{"rel_height": a - hcam, "count": b, "prominence": c} for a, b, c in peaks]
        below = [p for p in peaks if p[0] <= hcam - pc["min_camera_floor_gap_m"]]
        if not below:
            raise UpAxisError("no horizontal surface found below the cameras along the estimated up direction (floor not observed, or up is flipped); "
                              "see up_axis_failure.json", diag)
        floor_h = below[0][0]  # lowest significant surface under the cameras
        ceil = [p for p in peaks if p[0] >= hcam + 0.15 and p[0] - floor_h >= pc["min_ceiling_height_m"] and p[2] >= pc["ceiling_min_prominence"]]
        ceil_h = max(ceil, key=lambda p: p[1])[0] if ceil else None  # strongest surface above the cameras
    else:
        floor_h = peaks[0][0]
        ceil = [p for p in peaks if p[0] - floor_h >= pc["min_ceiling_height_m"] and p[2] >= pc["ceiling_min_prominence"]]
        ceil_h = ceil[-1][0] if ceil else None

    def refine(h0):
        sel = np.abs(h - h0) < 0.06
        r = ransac_plane(P[sel], thr, pc["ransac_iterations"], rng, pc["ransac_subsample"], up, 10.0)
        if r is None:
            return None
        n, d, m = r
        n, d = (n, d) if n @ up > 0 else (-n, -d)
        pts = P[sel][m]
        res = pts @ n + d
        return {"normal": n, "d": d, "support_count": int(m.sum()), "centroid": pts.mean(0), "residual_std": float(res.std()),
                "points": pts}
    fl = refine(floor_h)
    if fl is None:
        raise UpAxisError("could not fit a floor plane at the chosen height peak", diag)
    ce = refine(ceil_h) if ceil_h is not None else None
    if ceil_h is not None and ce is None:
        diag["warnings"].append("ceiling height peak found but plane fit failed; treating ceiling as unobserved")
    out = {"up": up, "floor": fl, "ceiling": ce, "diagnostics": diag}
    if cam_positions is not None:
        hc = cam_positions @ fl["normal"] + fl["d"]  # camera heights above the floor plane
        diag["camera_height_above_floor_m"] = {"min": float(hc.min()), "median": float(np.median(hc)), "max": float(hc.max())}
        if np.median(hc) < pc["min_camera_floor_gap_m"] or np.median(hc) > 2.6 or np.ptp(hc) > 1.2:
            raise UpAxisError(f"up-axis sanity check failed: cameras sit {np.median(hc):.2f} m above the detected floor (spread {np.ptp(hc):.2f} m); "
                              "the up direction or floor plane is wrong", diag)
        if ce is not None and np.median(cam_positions @ ce["normal"] + ce["d"]) > 0:
            diag["warnings"].append("cameras are above the detected ceiling plane; ceiling may be wrong")
    if ce is not None:
        out["ceiling_height"] = float(abs(fl["normal"] @ ce["centroid"] + fl["d"]))
        out["ceiling_tilt_deg"] = float(np.degrees(np.arccos(np.clip(abs(fl["normal"] @ ce["normal"]), 0, 1))))
    return out
