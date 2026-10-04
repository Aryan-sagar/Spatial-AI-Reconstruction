import numpy as np
import pytest
from applied_ai.reconstruction.keyframes import select_keyframes
from applied_ai.reconstruction.pointcloud import PointCloudAccumulator, read_ply_xyz, voxel_downsample, write_ply


def test_ply_roundtrip(tmp_path):
    p = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
    write_ply(tmp_path / "a.ply", p)
    assert np.allclose(read_ply_xyz(tmp_path / "a.ply"), p)


def test_voxel_downsample_means_and_counts():
    p = np.array([[0.0, 0, 0], [0.01, 0, 0], [1.0, 1.0, 1.0]])
    out, cnt = voxel_downsample(p, 0.1)
    assert len(out) == 2 and sorted(cnt.tolist()) == [1, 2]
    assert any(np.allclose(o, [0.005, 0, 0], atol=1e-6) for o in out)


def test_accumulator_bounded_and_merges():
    acc = PointCloudAccumulator(0.1, max_buffer_points=100)
    rng = np.random.default_rng(1)
    for _ in range(10):
        acc.add_frame(rng.uniform(0, 1, (60, 3)))
    s = acc.statistics()
    assert s["raw_points"] == 600 and s["voxel_points"] <= 1000


def test_keyframes_gap_motion_and_cap():
    n = 200
    pos = np.column_stack((np.linspace(0, 10, n), np.zeros(n), np.zeros(n)))
    q = np.tile([0, 0, 0, 1.0], (n, 1))
    ids = select_keyframes(pos, q, 10, 0.05, 3.0, 1000)
    assert ids[0] == 0 and all(b - a >= 10 for a, b in zip(ids, ids[1:]))
    assert len(select_keyframes(pos, q, 10, 0.05, 3.0, 5)) <= 5
    still = select_keyframes(np.zeros((50, 3)), q[:50], 5, 0.05, 3.0, 100)
    assert still == [0]  # no motion -> no extra keyframes
