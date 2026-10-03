from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IMU_COLUMNS = ["timestamp", "a_x", "a_y", "a_z", "alpha_x", "alpha_y", "alpha_z"]


def load_imu(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"imu file missing: {path}")
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    missing = [c for c in IMU_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing IMU columns {missing}; found {list(df.columns)}")
    return df


def load_camera_matrix(path: str | Path) -> np.ndarray:
    """3x3 matrix, comma-separated, no header (observed format). Fails loudly otherwise."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"camera matrix missing: {path}")
    m = np.loadtxt(path, delimiter=",", ndmin=2)
    if m.shape != (3, 3):
        raise ValueError(f"{path}: expected 3x3 matrix, got {m.shape}")
    return m
