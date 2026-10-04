"""Empirical selection of pose direction, camera axes and depth scale.

Metric: multi-view depth consistency. Back-project frame i with candidate conventions, move to frame j, and compare the
predicted depth with frame j's measured depth at the projected pixel. Wrong conventions give large relative error.
Candidates are scored on identical frame pairs; the run fails loudly if the winner is poor or ambiguous.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np

from ..reconstruction.keyframes import quat_angle_deg
from ..reconstruction.poses import AXES, DIRECTIONS, T_world_cvcamera, invert_transform, transform_points
from ..reconstruction.projection import depth_to_camera_points
from .intrinsics import DepthProjectionCalibration

log = logging.getLogger(__name__)


class ConventionError(RuntimeError):
    pass


def _pairs(df, n_base: int, tr_rng, max_rot, max_offset: int = 400):
    pos = df[["x", "y", "z"]].to_numpy()
    q = df[["qx", "qy", "qz", "qw"]].to_numpy()
    frames = df["frame"].to_numpy()
    out = []
    for i in np.unique(np.linspace(0, len(df) - 2, n_base).round().astype(int)):
        j = np.arange(i + 1, min(len(df), i + 1 + max_offset))
        d = np.linalg.norm(pos[j] - pos[i], axis=1)
        a = quat_angle_deg(q[j], q[i])
        ok = np.flatnonzero((d >= tr_rng[0]) & (d <= tr_rng[1]) & (a <= max_rot))
        if len(ok):
            out.append((int(frames[i]), int(frames[j[ok[np.argmax(d[ok])]]])))  # largest baseline = most scale-sensitive
    return out


def _score(data, pairs, direction, axes, scale, Kcal, cfg) -> tuple[float, int]:
    rc, ac = cfg["reconstruction"], cfg["auto_calibration"]
    errs = []
    for a, b in pairs:
        da, db = data[a], data[b]
        pts = depth_to_camera_points(da["raw"], da["K"], scale, da["conf"], ac["stride"], rc["min_depth_m"], rc["max_depth_m"],
                                     rc["confidence_min"] if rc["confidence_min"] else None)
        if len(pts) < 200:
            continue
        Ta = T_world_cvcamera(da["pos"], da["quat"], direction, axes)
        Tb = T_world_cvcamera(db["pos"], db["quat"], direction, axes)
        pb = transform_points(invert_transform(Tb), transform_points(Ta, pts))
        z = pb[:, 2]
        fz = z > rc["min_depth_m"]
        u = db["K"].fx * pb[:, 0] / np.where(fz, z, 1) + db["K"].cx
        v = db["K"].fy * pb[:, 1] / np.where(fz, z, 1) + db["K"].cy
        h, w = db["raw"].shape
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        ok = fz & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if ok.sum() < 150:
            continue
        meas = db["raw"][vi[ok], ui[ok]].astype(float) * scale
        good = meas >= rc["min_depth_m"]
        if good.sum() < 150:
            continue
        errs.append(float(np.median(np.abs(z[ok][good] - meas[good]) / meas[good])))
    return (float(np.median(errs)) if errs else float("inf")), len(errs)


def select_conventions(ds, cfg: dict) -> dict:
    """Returns a report; report['selected'] holds direction/axes/scale. Raises ConventionError if unusable."""
    ac = cfg["auto_calibration"]
    df = ds.odometry.df
    pairs = _pairs(df, ac["n_frames"], ac["pair_translation_m"], ac["pair_max_rotation_deg"])
    if len(pairs) < ac["min_valid_pairs"]:
        raise ConventionError(f"only {len(pairs)} usable frame pairs for convention selection (need {ac['min_valid_pairs']}); "
                              f"relax auto_calibration.pair_* or set conventions manually with auto_calibration.enabled=false")
    cal = DepthProjectionCalibration.from_config(cfg)
    data = {}
    for f in {x for p in pairs for x in p}:
        fr = ds.get_frame(f, load_rgb=False, load_imu=False)
        data[f] = {"raw": fr.depth.raw_values, "conf": fr.depth.confidence, "K": cal.depth_intrinsics(fr.intrinsics),
                   "pos": fr.pose.position, "quat": fr.pose.quaternion}
    rows = []
    for direction, axes, scale in itertools.product(DIRECTIONS, AXES, ac["scale_candidates"]):
        s, n = _score(data, pairs, direction, axes, scale, cal, cfg)
        rows.append({"direction": direction, "camera_axes": axes, "depth_scale": scale, "median_rel_error": s, "valid_pairs": n})
    rows.sort(key=lambda r: r["median_rel_error"])
    best, second = rows[0], rows[1]
    margin = second["median_rel_error"] / best["median_rel_error"] if best["median_rel_error"] > 0 else float("inf")
    rep = {"pairs_used": len(pairs), "candidates": rows, "selected": None, "margin_to_runner_up": margin, "status": "ok", "notes": []}
    if best["valid_pairs"] < ac["min_valid_pairs"] or not np.isfinite(best["median_rel_error"]):
        rep["status"] = "failed"
        rep["notes"].append("no candidate produced enough valid pairs")
    elif best["median_rel_error"] > ac["max_median_rel_error"]:
        rep["status"] = "failed"
        rep["notes"].append(f"best candidate error {best['median_rel_error']:.3f} exceeds {ac['max_median_rel_error']}")
    elif margin < ac["min_margin"]:
        rep["status"] = "ambiguous"
        rep["notes"].append(f"best/runner-up margin {margin:.2f} < {ac['min_margin']}")
    else:
        rep["selected"] = {k: best[k] for k in ("direction", "camera_axes", "depth_scale")}
    # informational fine-scale probe around the winner (NOT applied)
    if rep["selected"]:
        fine = []
        for f in np.linspace(0.9, 1.1, 9):
            s, _ = _score(data, pairs, best["direction"], best["camera_axes"], best["depth_scale"] * f, cal, cfg)
            fine.append({"scale_factor": float(f), "median_rel_error": s})
        rep["fine_scale_probe"] = fine
        e1 = next(f["median_rel_error"] for f in fine if abs(f["scale_factor"] - 1.0) < 1e-9)
        edge = 0.5 * (fine[0]["median_rel_error"] + fine[-1]["median_rel_error"])
        rep["scale_sensitivity"] = float(edge / e1) if e1 > 0 else float("inf")
        rep["scale_identified"] = bool(rep["scale_sensitivity"] >= ac.get("min_scale_sensitivity", 1.3))
        if not rep["scale_identified"]:
            rep["notes"].append(f"depth scale only weakly identified (error at +/-10% scale is {rep['scale_sensitivity']:.2f}x the error at 1.0; "
                                "baselines too short?) - scale status stays provisional")
    return rep


def apply_selection(cfg: dict, rep: dict) -> dict:
    if rep["status"] != "ok":
        raise ConventionError(f"convention selection {rep['status']}: {'; '.join(rep['notes'])} (see convention_report.json)")
    s = rep["selected"]
    cfg["pose"]["convention"], cfg["pose"]["camera_axes"] = s["direction"], s["camera_axes"]
    cfg["calibration"]["depth_scale"].update(value=s["depth_scale"], status="empirically_calibrated" if rep.get("scale_identified") else "provisional")
    return cfg
