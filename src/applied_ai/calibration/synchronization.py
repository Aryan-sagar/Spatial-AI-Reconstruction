"""FrameSynchronizer: tests (not assumes) the hypothesis frame i <-> RGB[i], depth[i], odometry[i]."""
from __future__ import annotations

import json
from pathlib import Path


class SynchronizationError(RuntimeError):
    pass


class FrameSynchronizer:
    def __init__(self, cfg: dict):
        self.tol_dur = cfg["synchronization"]["duration_tolerance_frac"]
        self.tol_fps = cfg["synchronization"]["fps_tolerance_frac"]
        self.strict = cfg["synchronization"]["strict"]
        self.hypothesis = cfg["synchronization"]["hypothesis"]

    def check(self, inventory_report: dict, odometry_df=None) -> dict:
        checks: list[dict] = []

        def add(name, ok, detail):
            checks.append({"check": name, "pass": bool(ok), "detail": detail})

        dep, rgb, odo = inventory_report.get("depth"), inventory_report.get("rgb"), inventory_report.get("odometry")
        if not (dep and rgb and odo and odo.get("rows")):
            add("streams_present", False, f"depth={bool(dep)} rgb={bool(rgb)} odometry={bool(odo)}")
        else:
            nd, nr, no = dep["frames"], rgb["frames"], odo["rows"]
            add("frame_counts_equal", nd == nr == no, f"depth={nd} rgb={nr} odometry={no}")
            add("depth_ids_match_odometry_frames", dep["range"] == odo["frame_range"] and not dep["missing_ids"],
                f"depth range {dep['range']} vs odometry {odo['frame_range']}")
            span = odo["duration_sec"]
            rgb_dur = (rgb["frames"] - 1) / rgb["fps"] if rgb["fps"] else None  # first-to-last frame time (n-1 intervals)
            if span and rgb_dur:
                rel = abs(span - rgb_dur) / span
                add("duration_agrees", rel <= self.tol_dur, f"odometry span={span:.4f}s rgb={rgb_dur:.4f}s rel_diff={rel:.5f} (tol {self.tol_dur})")
                eff = (no - 1) / span
                rel_f = abs(eff - rgb["fps"]) / rgb["fps"]
                add("effective_rate_matches_video_fps", rel_f <= self.tol_fps, f"odometry effective {eff:.3f} Hz vs video {rgb['fps']:.3f} fps")
            else:
                add("duration_agrees", False, "duration unavailable")
            # informational: non-uniform sampling does not break index alignment but must be visible
            f = lambda v: "n/a" if v is None else f"{v:.6f}s"
            add("timestamp_spacing_info", True,
                f"dt median={f(odo['dt_median'])} mean={f(odo['dt_mean'])} max={f(odo['dt_max'])} (median!=mean means non-uniform sampling)")
            add("timestamps_monotonic", odo["timestamps_monotonic"], "")
        ok = all(c["pass"] for c in checks)
        return {
            "hypothesis": self.hypothesis,
            "status": "consistent" if ok else "inconsistent",
            "note": "Count/duration agreement is necessary, not sufficient, for per-frame alignment; visual/depth cross-check is a later phase.",
            "checks": checks,
        }

    def run(self, inventory_report: dict, out_path: str | Path, odometry_df=None) -> dict:
        rep = self.check(inventory_report, odometry_df)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2))
        if rep["status"] != "consistent" and self.strict:
            failed = "; ".join(f"{c['check']}: {c['detail']}" for c in rep["checks"] if not c["pass"])
            raise SynchronizationError(f"streams disagree ({out}): {failed}")
        return rep
