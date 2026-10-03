import numpy as np
import pytest
from applied_ai.calibration.synchronization import SynchronizationError
from applied_ai.io.dataset import ScanDataset
from conftest import make_scan


def test_arbitrary_frame_bundle(scan, cfg):
    with ScanDataset(scan, cfg) as ds:
        assert len(ds) == 6
        for fid in (4, 0, 5, 2):  # arbitrary order
            fr = ds.get_frame(fid)
            assert fr.frame_id == fr.rgb_frame_index == fr.depth_frame_index == fid
            assert fr.depth.raw_values.dtype == np.uint16 and fr.depth.raw_values.shape == (192, 256)
            assert fr.rgb.shape == (144, 192, 3)
            assert fr.pose.position[2] == pytest.approx(0.01 * fid)
            assert fr.pose.convention == "camera_to_world"
            assert fr.intrinsics.fx == 100.0 and fr.intrinsics.source == "odometry_per_frame"
            assert fr.depth.confidence.shape == (192, 256)


def test_depth_scale_provisional_and_to_meters(scan, cfg):
    with ScanDataset(scan, cfg) as ds:
        d = ds.get_depth(0)
        assert d.depth_scale.status == "provisional"
        assert d.to_meters().min() == pytest.approx(1.0)  # raw 1000 * 0.001
        assert d.raw_values.min() == 1000  # raw untouched


def test_optional_loads_and_imu_window(scan, cfg):
    with ScanDataset(scan, cfg) as ds:
        fr = ds.get_frame(1, load_rgb=False, load_imu=False, load_confidence=False)
        assert fr.rgb is None and fr.imu_window is None and fr.depth.confidence is None
        assert ds.get_frame(0).imu_window.shape[1] == 7  # t=100.0 within window of imu samples


def test_out_of_range_frame(scan, cfg):
    with ScanDataset(scan, cfg) as ds:
        for bad in (-1, 6, 9999):
            with pytest.raises(IndexError):
                ds.get_frame(bad)


def test_open_fails_on_invalid_scan(tmp_path, cfg):
    with pytest.raises(RuntimeError, match="depth_ids_missing"):
        ScanDataset(make_scan(tmp_path / "s", drop=(2,)), cfg).open()


def test_open_fails_loudly_on_sync_mismatch(tmp_path, cfg):
    with pytest.raises(SynchronizationError):
        ScanDataset(make_scan(tmp_path / "s", dt=0.5), cfg).open()


def test_iter_frames_and_sequential_rgb(scan, cfg):
    with ScanDataset(scan, cfg) as ds:
        ids = [f.frame_id for f in ds.iter_frames([3, 1, 2], load_confidence=False)]
        assert ids == [1, 2, 3]


def test_raw_scan_not_modified(scan, cfg):
    before = {p.name: p.stat().st_mtime_ns for p in scan.rglob("*") if p.is_file()}
    with ScanDataset(scan, cfg) as ds:
        ds.get_frame(3)
    assert before == {p.name: p.stat().st_mtime_ns for p in scan.rglob("*") if p.is_file()}
