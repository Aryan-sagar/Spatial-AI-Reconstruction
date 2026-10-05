import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("fix_loop", Path(__file__).resolve().parents[2] / "scripts" / "fix_loop.py")
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)


def _g(name, ok, actual):
    return {"gate_name": name, "pass": ok, "actual": actual, "target": "t", "evidence": "e"}


def test_compare_gates_labels_every_movement():
    before = [_g("a", False, 5), _g("b", True, 1), _g("c", False, 9), _g("d", True, 2), _g("only_before", True, 0)]
    after = [_g("a", True, 1), _g("b", False, 4), _g("c", False, 7), _g("d", True, 2), _g("only_after", True, 0)]
    mv = {r["gate"]: r["movement"] for r in fl.compare_gates(before, after)}
    assert mv == {"a": "fail->pass", "b": "pass->fail", "c": "unchanged-fail", "d": "unchanged-pass", "only_before": "n/a", "only_after": "n/a"}
