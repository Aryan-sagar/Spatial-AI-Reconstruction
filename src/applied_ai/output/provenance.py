from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

from ..config import REPO_ROOT, config_hash


def input_fingerprint(scan: Path, cfg: dict) -> dict:
    """sha256 over odometry/camera_matrix/imu content plus (name,size) of every depth/confidence file and the RGB video size.
    (Hashing all depth bytes would be slow; sizes + names + full CSV content still detect any practical change.)"""
    f = cfg["dataset"]["files"]
    h = hashlib.sha256()
    for name in (f["odometry"], f["camera_matrix"], f["imu"]):
        h.update((scan / name).read_bytes())
    for d in (f["depth_dir"], f["confidence_dir"]):
        for p in sorted((scan / d).iterdir()):
            h.update(f"{p.name}:{p.stat().st_size}".encode())
    h.update(str((scan / f["rgb"]).stat().st_size).encode())
    return {"sha256": h.hexdigest(), "method": "csv-content + depth/confidence (name,size) + rgb size"}


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        return (r.stdout.strip() or "unknown") + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def build_provenance(scan: Path, cfg: dict, config_path: str | None) -> dict:
    import numpy, scipy, cv2
    return {"input_path": str(scan), "input_fingerprint": input_fingerprint(scan, cfg), "config_path": config_path, "config_hash": config_hash(cfg),
            "config": cfg, "code_version": git_commit(), "model_versions": [],
            "dependencies": {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__,
                             "platform": platform.platform()}}
