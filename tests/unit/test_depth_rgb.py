import cv2
import numpy as np
import pytest
from applied_ai.io.depth import DepthFormatError, list_depth_frames, read_depth_raw
from applied_ai.io.rgb import RGBVideoReader
from conftest import make_scan


def test_depth_dtype_and_values_preserved(scan):
    a = read_depth_raw(scan / "depth" / "000000.png", expected_hw=(192, 256))
    assert a.dtype == np.uint16 and a.min() == 1000 and a.max() > 255  # not normalized to 8 bit


def test_depth_shape_mismatch_raises(scan):
    with pytest.raises(DepthFormatError, match="000000"):
        read_depth_raw(scan / "depth" / "000000.png", expected_hw=(100, 100), frame_label="000000")


def test_depth_wrong_dtype_raises(tmp_path):
    p = tmp_path / "000000.png"
    cv2.imwrite(str(p), np.zeros((4, 4), np.uint8))
    with pytest.raises(DepthFormatError, match="uint16"):
        read_depth_raw(p)


def test_depth_multichannel_raises(tmp_path):
    p = tmp_path / "000000.png"
    cv2.imwrite(str(p), np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(DepthFormatError):
        read_depth_raw(p)


def test_list_depth_frames_ids(scan):
    ids, paths, odd = list_depth_frames(scan / "depth")
    assert ids == list(range(6)) and odd == []


def test_rgb_metadata_and_random_access(scan):
    with RGBVideoReader(scan / "rgb.mp4") as r:
        m = r.metadata()
        assert (m["width"], m["height"], m["frame_count"]) == (192, 144, 6)
        assert abs(m["fps"] - 10.0) < 1e-6
        f3 = r.get_frame(3)
        assert f3.shape == (144, 192, 3)
        ids = [i for i, _ in r.iter_frames([4, 1, 2])]
        assert ids == [1, 2, 4]
        with pytest.raises(IndexError):
            r.get_frame(500)


def test_rgb_missing_and_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        RGBVideoReader(tmp_path / "x.mp4").open()
    r = RGBVideoReader(tmp_path / "x.mp4")
    with pytest.raises(RuntimeError):
        r.metadata()
