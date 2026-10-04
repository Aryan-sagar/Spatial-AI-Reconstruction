"""End-to-end LiDAR-tier run: ScanDataset -> conventions -> cloud -> floor/ceiling -> walls -> openings -> floor plan -> Scene."""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np

from ..calibration.conventions import ConventionError, apply_selection, select_conventions
from ..geometry.coordinate_system import FloorCoordinateSystem, UpAxisError, camera_up_hint, find_floor_ceiling, gravity_up_from_imu, trajectory_up_hint
from ..geometry.floorplan import floor_footprint
from ..geometry.openings import detect_openings
from ..geometry.room_layout import extract_walls
from ..geometry.room_segmentation import segment_rooms
from ..geometry.rooms import build_room
from ..io.dataset import ScanDataset
from ..logging_utils import stage
from ..output.provenance import build_provenance
from ..output.renderer import render_plan, render_rooms, render_topdown
from ..output.schema import SCHEMA_VERSION, iter_measurements, validate_scene
from ..uncertainty import UncertaintyModel
from .pipeline import keyframe_ids, reconstruct_frame, reconstruct_scene_cloud
from .pointcloud import write_ply

log = logging.getLogger(__name__)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def run_lidar(scan: Path, cfg: dict, out_dir: Path, config_path: str | None = None, frame: int | None = None, tier: str = "lidar") -> dict:
    cfg = copy.deepcopy(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: list = []
    artifacts: list[str] = []
    with ScanDataset(scan, cfg) as ds:
        conv = None
        if cfg["auto_calibration"]["enabled"]:
            with stage("convention_selection", timings):
                conv = select_conventions(ds, cfg)
            (out_dir / "convention_report.json").write_text(json.dumps(conv, indent=2, default=_jsonable))
            apply_selection(cfg, conv)  # raises ConventionError (loudly) if failed/ambiguous
            log.info("conventions: %s (scale status: %s)", conv["selected"], cfg["calibration"]["depth_scale"]["status"])
        else:
            log.warning("auto_calibration disabled: using configured conventions UNVERIFIED (%s, %s, scale %s)", cfg["pose"]["convention"],
                        cfg["pose"]["camera_axes"], cfg["calibration"]["depth_scale"]["value"])
        if frame is not None:
            return reconstruct_frame(ds, frame, cfg, out_dir)
        cloud, kf_ids, cstats = reconstruct_scene_cloud(ds, cfg, timings)
        write_ply(out_dir / "pointcloud.ply", cloud)
        (out_dir / "pointcloud_stats.json").write_text(json.dumps(cstats, indent=2))
        artifacts += ["pointcloud.ply", "pointcloud_stats.json"]
        up_hint_cam = camera_up_hint(ds.odometry.df, kf_ids, cfg)
        imu_up, imu_diag = gravity_up_from_imu(ds.imu, ds.odometry.df, cfg)
        if cfg["planes"]["up_hint_source"] == "trajectory_pca":
            up_hint, up_diag = trajectory_up_hint(ds.odometry.df, kf_ids, cfg, imu_up)
            up_diag["imu"] = imu_diag
            if imu_up is not None:
                up_diag["imu_vs_trajectory_deg"] = float(np.degrees(np.arccos(np.clip(imu_up @ up_hint, -1, 1))))
        else:
            up_hint, up_diag = up_hint_cam, {"imu": imu_diag}
        if cfg["planes"]["up_flip"]:
            up_hint = -up_hint
            up_diag["manual_up_flip"] = True
        cam_pos = ds.odometry.df.set_index("frame").loc[kf_ids][["x", "y", "z"]].to_numpy(float)
        capture_id = Path(scan).name
        prov = build_provenance(Path(scan), cfg, config_path)

    with stage("floor_ceiling", timings, input_points=len(cloud)):
        try:
            fc = find_floor_ceiling(cloud, up_hint, cfg, cam_pos)
        except UpAxisError as e:
            (out_dir / "up_axis_failure.json").write_text(json.dumps({"error": str(e), "up_hint_info": up_diag, "diagnostics": e.diag}, indent=2, default=_jsonable))
            raise
    cs = FloorCoordinateSystem.from_plane(fc["up"], fc["floor"]["centroid"] - (fc["floor"]["normal"] @ fc["floor"]["centroid"] + fc["floor"]["d"]) * fc["floor"]["normal"])
    L = cs.to_local(cloud)
    ch = fc.get("ceiling_height")
    with stage("walls", timings):
        cam_uv = cs.to_local(cam_pos)[:, :2]
        interior = cam_uv.mean(0)  # camera path centre: defines which side of a wall is 'inside'
        wl = extract_walls(L, cam_uv, ch, cfg)
        cells, band, all_raw_walls, raw_walls, final, minfo, prune_info = (wl[k] for k in ("cells", "band", "all_raw_walls", "raw_walls", "final", "minfo", "prune_info"))
        col_stats = minfo["column_filter"]
    with stage("floorplan", timings):
        fp = floor_footprint(L, final, cfg, cam_uv)
    with stage("room_segmentation", timings):   # additive: does not change the single-room scene below; reported in diagnostics + debug/rooms.png
        seg = segment_rooms(L[np.abs(L[:, 2]) < cfg["floorplan"]["floor_band_m"], :2], cells, cam_uv, cfg)
    log.info("room segmentation: %d basins kept (%s m2), %d dropped, %d open passages", len(seg["rooms"]), [round(r["area_m2"], 1) for r in seg["rooms"]],
             len(seg["dropped_basins"]), len(seg["connections"]))
    if not fp.get("area_reliable", True):
        log.error("FLOOR AREA UNRELIABLE - do not treat as a measurement: %s", "; ".join(fp["unreliable_reasons"]))
    with stage("openings", timings):
        ops = detect_openings(L, final, interior, ch, cfg)
    um = UncertaintyModel(cfg, tier)
    rot = minfo["dominant_angle_deg"] or 0.0
    room = build_room("room_01", raw_walls, final, ops, fp, fc, rot, um, artifacts)
    if cfg.get("debug", {}).get("enabled", True):
        dbg = out_dir / "debug"
        render_topdown(dbg / "topdown.png", cells, L[np.abs(L[:, 2]) < cfg["floorplan"]["floor_band_m"], :2], cam_uv, all_raw_walls, final,
                       {p["id"] for p in prune_info.get("pruned", [])})
        render_rooms(dbg / "rooms.png", seg, cam_uv)
        write_ply(dbg / "floor_plane.ply", fc["floor"]["points"])
        if fc.get("ceiling") is not None:
            write_ply(dbg / "ceiling_plane.ply", fc["ceiling"]["points"])
        wp = np.vstack([cs.to_world(np.column_stack((w["_pts"], np.full(len(w["_pts"]), 1.0)))) for w in raw_walls])
        write_ply(dbg / "walls.ply", wp)
    scene = {"schema_version": SCHEMA_VERSION, "capture_id": capture_id, "tier": tier, "rooms": [room], "adjacency": [], "measurements": [],
             "damages": [], "concealed_damage_flags": [], "scope_line_items": [],
             "diagnostics": {
                 "calibration": {"conventions": conv["selected"] if conv else "configured_unverified", "depth_scale": cfg["calibration"]["depth_scale"],
                                 "scale_identified": (conv or {}).get("scale_identified"), "scale_sensitivity": (conv or {}).get("scale_sensitivity"),
                                 "convention_notes": (conv or {}).get("notes", []), "depth_intrinsics_mode": cfg["calibration"]["depth_intrinsics"]["mode"],
                                 "uncertainty": "intervals are UNCALIBRATED PRIORS unless uncertainty.calibration_file is set"},
                 "multiroom": {"status": "experimental_unvalidated", "note": "free-space segmentation of the whole scan; areas are obstacle-shrunk free-space estimates, not tape wall-to-wall; not yet part of the scene rooms",
                               **{k: v for k, v in seg.items() if not k.startswith("_")}},
                 "drift": {"status": "not_run", "note": "drift correction not implemented; poses are used as supplied (baseline)"},
                 "runtime": timings,
                 "quality": {"keyframes": len(kf_ids), "cloud": cstats, "floor": {"support": fc["floor"]["support_count"], "residual_std_m": fc["floor"]["residual_std"]},
                             "ceiling": None if fc.get("ceiling") is None else {"support": fc["ceiling"]["support_count"], "residual_std_m": fc["ceiling"]["residual_std"],
                                                                                 "tilt_vs_floor_deg": fc["ceiling_tilt_deg"]},
                             "up_hint_vs_plane_deg": fc["diagnostics"]["up_hint_vs_plane_deg"], "up_hint": {"source": cfg["planes"]["up_hint_source"], **up_diag, "camera_up_vs_hint_deg": float(np.degrees(np.arccos(np.clip(up_hint_cam @ up_hint, -1, 1)))), "refined_deg": fc["diagnostics"]["up_refined_from_hint_deg"], "camera_height_above_floor_m": fc["diagnostics"].get("camera_height_above_floor_m")}, "height_peaks": fc["diagnostics"]["height_peaks"],
                             "manhattan": minfo, "wall_band_m": list(band), "footprint_warnings": fp["warnings"], "footprint_area_reliable": fp.get("area_reliable"), "footprint_unreliable_reasons": fp.get("unreliable_reasons"), "footprint_method": fp["method"], "footprint_enclosure": fp.get("enclosure"), "inferred_corner_extension_m": {w["id"]: w.get("inferred_ext_m") for w in final}, "floor_evidence_area_m2": fp["floor_evidence_area_m2"],
                             "rejected_openings": [o for o in ops if o["status"] != "observed"]},
                 "not_implemented": ["damage detection", "concealed damage", "scope line items", "drift correction", "multi-room stitching"]},
             "provenance": prov}
    scene["measurements"] = [{"path": p, **m} for p, m in iter_measurements({"rooms": scene["rooms"]})]
    errs = validate_scene(scene)
    if errs:
        raise RuntimeError("scene failed validation: " + "; ".join(errs))
    (out_dir / "scene.json").write_text(json.dumps(scene, indent=2, default=_jsonable))
    (out_dir / "diagnostics.json").write_text(json.dumps(scene["diagnostics"], indent=2, default=_jsonable))
    render_plan(scene, out_dir / "plan.png", out_dir / "plan.svg")
    return scene
