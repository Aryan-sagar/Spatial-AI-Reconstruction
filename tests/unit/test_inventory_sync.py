import json
import pytest
from applied_ai.calibration.synchronization import FrameSynchronizer, SynchronizationError
from applied_ai.io.inventory import DatasetInventory
from conftest import make_scan


def codes(inv):
    return {i["code"] for i in inv.issues}


def run(scan, cfg):
    inv = DatasetInventory(cfg)
    inv.scan(scan)
    return inv


def test_happy_path(scan, cfg):
    inv = run(scan, cfg)
    r = inv.report()
    assert inv.validate() and r["valid"], inv.issues
    assert r["depth"]["frames"] == 6 and r["depth"]["range"] == [0, 5]
    assert r["depth"]["first_file"] == "000000.png" and r["depth"]["last_file"] == "000005.png"
    assert r["depth"]["resolution"] == [256, 192] and r["depth"]["dtype"] == "uint16"
    assert r["rgb"]["resolution"] == [192, 144] and r["rgb"]["frames"] == 6
    assert r["odometry"]["rows"] == 6


@pytest.mark.parametrize("name,code", [
    ("depth", "depth_dir_missing"), ("confidence", "confidence_dir_missing"), ("rgb.mp4", "rgb_missing"),
    ("odometry.csv", "odometry_missing"), ("camera_matrix.csv", "camera_matrix_missing"), ("imu.csv", "imu_missing"),
])
def test_missing_components(tmp_path, cfg, name, code):
    inv = run(make_scan(tmp_path / "s", skip_files=(name,)), cfg)
    assert code in codes(inv) and not inv.validate()


def test_missing_scan_dir(tmp_path, cfg):
    inv = run(tmp_path / "nope", cfg)
    assert "scan_missing" in codes(inv) and not inv.validate()


def test_missing_depth_ids(tmp_path, cfg):
    inv = run(make_scan(tmp_path / "s", drop=(2,)), cfg)
    assert "depth_ids_missing" in codes(inv)


def test_depth_shape_mismatch(tmp_path, cfg):
    inv = run(make_scan(tmp_path / "s", depth_hw=(100, 120)), cfg)
    assert "depth_shape_mismatch" in codes(inv)


def test_odometry_duplicate_missing_and_malformed(tmp_path, cfg):
    inv = run(make_scan(tmp_path / "s", n=5, odo_rows=[0, 1, 1, 3, 4]), cfg)
    assert {"odometry_duplicate_ids", "odometry_missing_ids"} <= codes(inv)


def test_intrinsics_disagreement_warns(tmp_path, cfg):
    root = make_scan(tmp_path / "s")
    (root / "camera_matrix.csv").write_text("120.0, 0, 50\n0, 120.0, 40\n0, 0, 1\n")
    inv = run(root, cfg)
    assert "intrinsics_disagree" in codes(inv) and inv.validate()  # warning only


def test_inventory_json_written(scan, cfg, tmp_path):
    inv = run(scan, cfg)
    p = inv.write(tmp_path / "out")
    assert p == tmp_path / "out" / "inventory" / "scan_ok.json"
    assert json.loads(p.read_text())["depth"]["frames"] == 6
    assert scan.joinpath("odometry.csv").exists()  # raw untouched


def test_sync_consistent_and_report_written(scan, cfg, tmp_path):
    rep = FrameSynchronizer(cfg).run(run(scan, cfg).report(), tmp_path / "sync.json")
    assert rep["status"] == "consistent", rep
    assert json.loads((tmp_path / "sync.json").read_text())["hypothesis"] == "index_aligned"


def test_sync_fails_loudly_on_count_mismatch(tmp_path, cfg):
    inv = run(make_scan(tmp_path / "s", drop=(5,)), cfg)  # depth has 5, odometry/rgb 6
    with pytest.raises(SynchronizationError, match="frame_counts_equal"):
        FrameSynchronizer(cfg).run(inv.report(), tmp_path / "sync.json")
    assert (tmp_path / "sync.json").exists()  # report still written


def test_sync_fails_on_duration_disagreement(tmp_path, cfg):
    inv = run(make_scan(tmp_path / "s", dt=0.5), cfg)  # odometry 2.5s vs video 0.6s
    with pytest.raises(SynchronizationError, match="duration_agrees"):
        FrameSynchronizer(cfg).run(inv.report(), tmp_path / "sync.json")


def test_sync_non_strict_returns_report(tmp_path, cfg):
    cfg["synchronization"]["strict"] = False
    inv = run(make_scan(tmp_path / "s", dt=0.5), cfg)
    assert FrameSynchronizer(cfg).run(inv.report(), tmp_path / "x.json")["status"] == "inconsistent"


def test_inspect_does_not_crash_when_odometry_unusable(tmp_path, cfg):
    root = make_scan(tmp_path / "s", n=3)
    p = root / "odometry.csv"
    p.write_text(p.read_text().replace("100.0, 100.0, 50.0, 40.0", "nan, nan, nan, nan"))
    inv = run(root, cfg)
    assert "odometry_invalid" in codes(inv)
    cfg["synchronization"]["strict"] = False
    assert FrameSynchronizer(cfg).check(inv.report())["status"] == "inconsistent"
