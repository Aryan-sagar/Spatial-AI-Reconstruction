import numpy as np
from applied_ai.io.dataset import ScanDataset
from synth import make_room_scan


def test_synthetic_scan_opens_and_is_consistent(tmp_path, cfg):
    cfg["synchronization"]["duration_tolerance_frac"] = 0.05
    make_room_scan(tmp_path / "r", n=20)
    with ScanDataset(tmp_path / "r", cfg) as ds:
        fr = ds.get_frame(3)
        assert fr.depth.raw_values.dtype == np.uint16 and fr.depth.raw_values.min() > 100
