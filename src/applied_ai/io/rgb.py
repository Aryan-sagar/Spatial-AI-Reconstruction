"""RGB video reader: single open, sequential access without re-seeking."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np


class RGBVideoReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._cap: cv2.VideoCapture | None = None
        self._next = 0  # index of the frame the decoder will return next

    def open(self) -> "RGBVideoReader":
        if not self.path.exists():
            raise FileNotFoundError(f"RGB video missing: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"cannot open video: {self.path}")
        self._cap, self._next = cap, 0
        return self

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def metadata(self) -> dict:
        cap = self._require()
        n = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "frame_count": n,
            "duration_sec": n / fps if fps > 0 else None,
            "note": "frame_count is container metadata and may differ from decodable frames",
        }

    def get_frame(self, frame_id: int) -> np.ndarray:
        cap = self._require()
        if frame_id < 0:
            raise IndexError(f"negative frame id {frame_id}")
        if frame_id != self._next:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, img = cap.read()
        if not ok:
            self._next = -1  # unknown position; force a seek next time
            raise IndexError(f"cannot decode RGB frame {frame_id} of {self.path}")
        self._next = frame_id + 1
        return img  # BGR uint8

    def iter_frames(self, frame_ids: Iterable[int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
        if frame_ids is None:
            n = self.metadata()["frame_count"]
            frame_ids = range(n)
        for fid in sorted(frame_ids):
            yield fid, self.get_frame(fid)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _require(self) -> cv2.VideoCapture:
        if self._cap is None:
            raise RuntimeError("video not open; call open()")
        return self._cap
