import pytest
from applied_ai.io.odometry import OdometryError, load_odometry
from conftest import make_scan


def test_odometry_columns_are_normalized(tmp_path):
    t = load_odometry(make_scan(tmp_path / "s") / "odometry.csv")
    assert "timestamp" in t.df.columns and all(c == c.strip() for c in t.df.columns)
    assert "distortion_center_x" in t.df.columns


def test_odometry_frames_contiguous(tmp_path):
    t = load_odometry(make_scan(tmp_path / "s", n=5) / "odometry.csv")
    assert t.missing_frame_ids == [] and t.n_rows == 5 and t.summary()["frame_range"] == [0, 4]


def test_missing_and_duplicate_frames_detected(tmp_path):
    t = load_odometry(make_scan(tmp_path / "s", n=5, odo_rows=[0, 1, 1, 3, 4]) / "odometry.csv")
    assert t.duplicate_frame_ids == [1] and t.missing_frame_ids == [2]


def test_quaternion_normalization_logged(tmp_path):
    t = load_odometry(make_scan(tmp_path / "s", quat=(0, 0, 0, 2.0)) / "odometry.csv")
    assert any("normalized" in w for w in t.warnings)
    assert abs(t.df["qw"].iloc[0] - 1.0) < 1e-12


def test_degenerate_quaternion_fails(tmp_path):
    with pytest.raises(OdometryError, match="degenerate"):
        load_odometry(make_scan(tmp_path / "s", quat=(0, 0, 0, 0)) / "odometry.csv")


def test_missing_columns_and_file(tmp_path):
    p = tmp_path / "o.csv"
    p.write_text(" timestamp, frame, x\n1,0,0\n")
    with pytest.raises(OdometryError, match="missing required columns"):
        load_odometry(p)
    with pytest.raises(FileNotFoundError):
        load_odometry(tmp_path / "none.csv")


def test_malformed_row_counted(tmp_path):
    root = make_scan(tmp_path / "s", n=4)
    p = root / "odometry.csv"
    lines = p.read_text().splitlines()
    lines[2] = lines[2].replace("0.0, 0.0, 0.01", "abc, 0.0, 0.01", 1)
    p.write_text("\n".join(lines) + "\n")
    t = load_odometry(p)
    assert t.n_malformed_rows == 1 and t.n_rows == 3


def test_non_monotonic_timestamps_warn(tmp_path):
    root = make_scan(tmp_path / "s", n=3, dt=-0.1)
    assert not load_odometry(root / "odometry.csv").summary()["timestamps_monotonic"]
