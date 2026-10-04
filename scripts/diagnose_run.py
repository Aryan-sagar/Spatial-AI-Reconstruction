"""Read-only summary of a finished `run`, to debug wrong geometry.
usage: python scripts/diagnose_run.py <scan_dir> <output_dir>   e.g. benchmark/single_room outputs/single_room"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

scan, out = Path(sys.argv[1]), Path(sys.argv[2])
conv = json.loads((out / "convention_report.json").read_text())
print("== convention selection ==  status:", conv["status"], "| pairs:", conv["pairs_used"], "| margin:", round(conv["margin_to_runner_up"], 2),
      "| scale_sensitivity:", conv.get("scale_sensitivity"), "| scale_identified:", conv.get("scale_identified"))
for r in conv["candidates"][:6]:
    print(f"  {r['direction']:16s} {r['camera_axes']:7s} scale={r['depth_scale']:<7} err={r['median_rel_error']:.4f} pairs={r['valid_pairs']}")
print("  notes:", conv["notes"])

df = pd.read_csv(scan / "odometry.csv", skipinitialspace=True); df.columns = df.columns.str.strip()
P = df[["x", "y", "z"]].to_numpy()
print("\n== trajectory (pose translation) extents, metres ==")
print("  min", P.min(0).round(2), "max", P.max(0).round(2), "span", (P.max(0) - P.min(0)).round(2))
imu = pd.read_csv(scan / "imu.csv", skipinitialspace=True); imu.columns = imu.columns.str.strip()
print("  mean IMU accel (device frame):", imu[["a_x", "a_y", "a_z"]].mean().round(2).tolist())

fail = out / "up_axis_failure.json"
if fail.exists():
    f = json.loads(fail.read_text())
    print("\n== UP-AXIS FAILURE ==\n ", f["error"])
    print("  up hint info:", json.dumps({k: v for k, v in f["up_hint_info"].items() if k != "imu"}))
    print("  imu:", json.dumps({k: v for k, v in f["up_hint_info"].get("imu", {}).items()}))
    d = f["diagnostics"]
    print("  camera height along up:", d.get("camera_height_along_up"))
    print("  peaks relative to camera (rel_height, count, prominence):", [(round(p["rel_height"], 2), round(p["count"]), round(p["prominence"], 1)) for p in d.get("height_peaks_relative_to_camera", [])])
    print("  up_hint_vs_plane_deg:", d.get("up_hint_vs_plane_deg"), "| refined vs hint:", d.get("up_refined_from_hint_deg"))
    sys.exit(0)
sc = json.loads((out / "scene.json").read_text()); q = sc["diagnostics"]["quality"]; r = sc["rooms"][0]
print("\n== up axis ==", json.dumps(q["up_hint"], default=str)[:900])
print("  camera height above floor:", q["up_hint"].get("camera_height_above_floor_m"))
print("\n== floor / ceiling ==")
print("  up_hint_vs_plane_deg:", round(q["up_hint_vs_plane_deg"], 1), "| floor:", q["floor"], "| ceiling:", q["ceiling"])
print("  height peaks:", [(round(p["height"], 2), round(p["count"]), round(p["prominence"], 1)) for p in q["height_peaks"]])
print("\n== cloud ==", {k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in q["cloud"].items()})
print("\n== walls ==")
for w in r["walls"]:
    print(" ", w["id"], "len", round(w["length"]["value"], 2), "raw", round(w["raw_length_m"], 2), "cells", w["support_cells"])
print("  pruning:", q["manhattan"]["boundary_pruning"].get("pruned"), "| footprint:", q["footprint_method"], q["footprint_warnings"])
print("\n== openings ==")
for o in r["openings"]:
    print(" ", o["id"], o["type"], round(o["width"]["value"], 2), o["parent_wall"], o["evidence"])
print("  rejected:", [(o["wall_id"], round(o["width"], 2), o["see_through_points"]) for o in q["rejected_openings"]])
