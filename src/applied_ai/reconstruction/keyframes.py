from __future__ import annotations

import numpy as np


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angle between unit quaternions (broadcasting), degrees."""
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return np.degrees(2 * np.arccos(d))


def select_keyframes(positions: np.ndarray, quats: np.ndarray, min_frame_gap: int, translation_threshold: float,
                     rotation_threshold_deg: float, max_frames: int) -> list[int]:
    """Greedy motion-based selection; if more than max_frames qualify, thin uniformly (keeps full coverage)."""
    n = len(positions)
    if n == 0:
        return []
    sel, last = [0], 0
    for i in range(1, n):
        if i - last < min_frame_gap:
            continue
        if np.linalg.norm(positions[i] - positions[last]) >= translation_threshold or quat_angle_deg(quats[i], quats[last]) >= rotation_threshold_deg:
            sel.append(i)
            last = i
    if len(sel) > max_frames:
        sel = [sel[k] for k in np.unique(np.linspace(0, len(sel) - 1, max_frames).round().astype(int))]
    return sel
