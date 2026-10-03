"""Depth PNG access. Raw uint16 values are preserved; no scaling is applied here."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


class DepthFormatError(ValueError):
    pass


def read_depth_raw(path: str | Path, expected_hw: tuple[int, int] | None = None, frame_label: str | None = None) -> np.ndarray:
    """Read a depth PNG as raw uint16, single channel (IMREAD_UNCHANGED). expected_hw=(height,width)."""
    path = Path(path)
    label = frame_label or path.name
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise DepthFormatError(f"depth frame {label}: unreadable file {path}")
    if img.ndim != 2:
        raise DepthFormatError(f"depth frame {label}: expected single channel, got shape {img.shape}")
    if img.dtype != np.uint16:
        raise DepthFormatError(f"depth frame {label}: expected uint16, got {img.dtype}")
    if expected_hw is not None and img.shape != tuple(expected_hw):
        raise DepthFormatError(f"depth frame {label}: expected (h,w)={tuple(expected_hw)}, got {img.shape}")
    return img


_NUM = re.compile(r"^(\d+)\.png$", re.IGNORECASE)


def list_depth_frames(depth_dir: str | Path) -> tuple[list[int], dict[int, Path], list[str]]:
    """Return (sorted frame ids, id->path, non-conforming filenames)."""
    ids: dict[int, Path] = {}
    odd: list[str] = []
    for p in Path(depth_dir).iterdir():
        m = _NUM.match(p.name)
        if m:
            ids[int(m.group(1))] = p
        elif p.is_file():
            odd.append(p.name)
    return sorted(ids), ids, sorted(odd)
