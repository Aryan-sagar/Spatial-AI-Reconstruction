"""Evaluation metrics and assessment gates. Independent of the reconstruction code (reads scene.json + a ground-truth file)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


def pct_error(pred, gt): return abs(pred - gt) / gt * 100.0
def abs_err_cm(pred, gt): return abs(pred - gt) * 100.0
def spread(a, b): return abs(a - b), abs(a - b) / ((a + b) / 2) * 100.0


def match(pred: list[float], gt: list[float]):
    """Optimal one-to-one assignment minimising total absolute error (Hungarian). Unmatched items are reported, not dropped silently."""
    if not pred or not gt:
        return [], list(range(len(pred))), list(range(len(gt)))
    C = np.abs(np.subtract.outer(pred, gt))
    r, c = linear_sum_assignment(C)
    pairs = [(int(i), int(j)) for i, j in zip(r, c)]
    return pairs, sorted(set(range(len(pred))) - {i for i, _ in pairs}), sorted(set(range(len(gt))) - {j for _, j in pairs})


def gate(name, target, actual, passed, evidence) -> dict:
    return {"gate_name": name, "target": target, "actual": actual, "pass": passed, "evidence": evidence}


def evaluate_scene(scene: dict, gt: dict, repeat_scene: dict | None = None) -> dict:
    """gt format (ours, documented in docs/ground_truth_format.md):
       ceiling_height_m: float | wall_lengths_m: [..] | opening_widths_m: [..] | floor_area_m2: float
    Only quantities present in the ground-truth file are evaluated; everything else is reported as not_evaluated."""
    room = scene["rooms"][0]
    out = {"capture_id": scene["capture_id"], "tier": scene["tier"], "metrics": {}, "gates": []}
    m = out["metrics"]
    if "ceiling_height_m" in gt:
        v = room["ceiling_height"]["value"]
        if v is None:
            out["gates"].append(gate("ceiling_height", "<= 1.5 cm", "unobserved", False, "no ceiling observed"))
        else:
            e = abs_err_cm(v, gt["ceiling_height_m"])
            ci = room["ceiling_height"]["confidence_interval"]
            m["ceiling_height"] = {"pred": v, "gt": gt["ceiling_height_m"], "abs_err_cm": e, "ci_covers_gt": ci["lower"] <= gt["ceiling_height_m"] <= ci["upper"]}
            out["gates"].append(gate("ceiling_height", "<= 1.5 cm", f"{e:.2f} cm", e <= 1.5, "single room"))
    if "wall_lengths_m" in gt:
        pred = [w["length"]["value"] for w in room["walls"]]
        pairs, up, ug = match(pred, gt["wall_lengths_m"])
        rows = [{"pred": pred[i], "gt": gt["wall_lengths_m"][j], "err_pct": pct_error(pred[i], gt["wall_lengths_m"][j]),
                 "ci_covers_gt": room["walls"][i]["length"]["confidence_interval"]["lower"] <= gt["wall_lengths_m"][j] <= room["walls"][i]["length"]["confidence_interval"]["upper"]} for i, j in pairs]
        m["wall_lengths"] = {"matched": rows, "unmatched_pred": up, "unmatched_gt": ug}
        tol = {"lidar": 0.5, "video": 3.0, "photo": 8.0}[scene["tier"]]  # lidar: repeatability-style bound; video/photo: assessment tolerances
        ok = bool(rows) and not ug and all(r["err_pct"] <= tol for r in rows)
        out["gates"].append(gate(f"wall_lengths_{scene['tier']}", f"<= {tol}% per wall", f"max {max(r['err_pct'] for r in rows):.2f}%" if rows else "none", ok,
                                 f"{len(rows)} matched, {len(ug)} GT walls unmatched"))
    if "opening_widths_m" in gt:
        pred = [o["width"]["value"] for o in room["openings"]]
        pairs, up, ug = match(pred, gt["opening_widths_m"])
        rows = [{"pred": pred[i], "gt": gt["opening_widths_m"][j], "abs_err_cm": abs_err_cm(pred[i], gt["opening_widths_m"][j])} for i, j in pairs]
        # unmatched GT openings count as failures (a missed opening is a miss, not an exclusion)
        n_total = len(gt["opening_widths_m"])
        n_ok = sum(r["abs_err_cm"] <= 2.0 for r in rows)
        m["opening_widths"] = {"matched": rows, "unmatched_pred": up, "unmatched_gt": ug}
        out["gates"].append(gate("opening_width", ">= 85% of openings within 2 cm", f"{n_ok}/{n_total} = {100 * n_ok / n_total:.0f}%", n_ok / n_total >= 0.85,
                                 f"{len(ug)} GT openings missed, {len(up)} extra detections"))
    if "floor_area_m2" in gt:
        v = room["floor_area"]["value"]
        m["floor_area"] = {"pred": v, "gt": gt["floor_area_m2"], "err_pct": pct_error(v, gt["floor_area_m2"])}
    if repeat_scene is not None and "ceiling_height_m" in gt:
        a, b = room["ceiling_height"]["value"], repeat_scene["rooms"][0]["ceiling_height"]["value"]
        if a is not None and b is not None:
            d, _ = spread(a, b)
            out["gates"].append(gate("repeated_ceiling_spread", "<= 1 cm", f"{d * 100:.2f} cm", d * 100 <= 1.0, "two captures of the same room, same tier"))
    return out


def calibration_from_residuals(results: list[dict], alpha: float = 0.05) -> dict:
    """Conformal-style error quantiles per (tier, kind) from evaluation results -> JSON usable as uncertainty.calibration_file.
    Residuals in the unit of the measurement (metres / m2). Needs real ground truth; small n gets inflated at use time."""
    from ..uncertainty import conformal_quantile
    bucket: dict = {}
    for r in results:
        t = r["tier"]
        mm = r["metrics"]
        if "ceiling_height" in mm:
            bucket.setdefault(t, {}).setdefault("ceiling_height", []).append(mm["ceiling_height"]["abs_err_cm"] / 100)
        for row in mm.get("wall_lengths", {}).get("matched", []):
            bucket.setdefault(t, {}).setdefault("wall_length", []).append(abs(row["pred"] - row["gt"]))
        for row in mm.get("opening_widths", {}).get("matched", []):
            bucket.setdefault(t, {}).setdefault("opening_width", []).append(row["abs_err_cm"] / 100)
        if "floor_area" in mm:
            bucket.setdefault(t, {}).setdefault("floor_area", []).append(abs(mm["floor_area"]["pred"] - mm["floor_area"]["gt"]))
    return {t: {k: {"quantile": conformal_quantile(v, alpha), "n": len(v)} for k, v in d.items()} for t, d in bucket.items()}


def load_gt(path) -> dict:
    return yaml.safe_load(Path(path).read_text())
