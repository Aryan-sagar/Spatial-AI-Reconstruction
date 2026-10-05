"""Regenerate the fix-loop before/after on one scan with ground truth (one command, deterministic).

  python scripts/fix_loop.py --input benchmark/room_a_scan1 --ground-truth benchmark/room_a_gt.yaml --repeat-input benchmark/room_a_scan2 ^
         --before-config configs/ablation_no_high_band.yaml --after-config configs/lidar.yaml

Runs the pipeline with the 'before' config and the 'after' config (same code, same raw input), evaluates both against the same ground truth and writes
outputs/fix_loop/{before,after}/..., comparison.json and report.md (gate-by-gate table). The readable diff of the fix itself is the git diff of its commit."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def compare_gates(before: list[dict], after: list[dict]) -> list[dict]:
    b = {g["gate_name"]: g for g in before}
    a = {g["gate_name"]: g for g in after}
    rows = []
    for name in sorted(set(b) | set(a)):
        gb, ga = b.get(name), a.get(name)
        pb, pa = (gb or {}).get("pass"), (ga or {}).get("pass")
        move = "n/a" if gb is None or ga is None else ("fail->pass" if (not pb and pa) else "pass->fail" if (pb and not pa) else "unchanged-pass" if pa else "unchanged-fail")
        rows.append({"gate": name, "before_actual": (gb or {}).get("actual"), "after_actual": (ga or {}).get("actual"), "target": (ga or gb or {}).get("target"),
                     "before_pass": pb, "after_pass": pa, "movement": move, "before_evidence": (gb or {}).get("evidence"), "after_evidence": (ga or {}).get("evidence")})
    return rows


def _cli(*args: str) -> None:
    cmd = [sys.executable, "-m", "applied_ai.cli", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _variant(name: str, cfg: str, scan: str, gt: str, repeat: str | None, root: Path) -> list[dict]:
    out = root / name
    _cli("run", "--input", scan, "--tier", "lidar", "--config", cfg, "--output", str(out))
    repeat_args: list[str] = []
    if repeat:
        _cli("run", "--input", repeat, "--tier", "lidar", "--config", cfg, "--output", str(out / "repeat"))
        repeat_args = ["--repeat-scene", str(out / "repeat" / "scene.json")]
    _cli("evaluate", "--scene", str(out / "scene.json"), "--ground-truth", gt, "--output", str(out), *repeat_args)
    return json.loads((out / "gate_results.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--repeat-input", default=None)
    ap.add_argument("--before-config", required=True)
    ap.add_argument("--after-config", default="configs/lidar.yaml")
    ap.add_argument("--output", default="outputs/fix_loop")
    a = ap.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    before = _variant("before", a.before_config, a.input, a.ground_truth, a.repeat_input, root)
    after = _variant("after", a.after_config, a.input, a.ground_truth, a.repeat_input, root)
    rows = compare_gates(before, after)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    (root / "comparison.json").write_text(json.dumps({"git_commit": commit, "input": a.input, "ground_truth": a.ground_truth, "before_config": a.before_config,
                                                      "after_config": a.after_config, "gates": rows}, indent=2))
    lines = ["# Fix loop: before vs after", "", f"commit `{commit[:10]}`, input `{a.input}`, ground truth `{a.ground_truth}`, before `{a.before_config}`, after `{a.after_config}`", "",
             "| gate | target | before | after | movement |", "|---|---|---|---|---|"]
    lines += [f"| {r['gate']} | {r['target']} | {r['before_actual']} ({'PASS' if r['before_pass'] else 'FAIL'}) | {r['after_actual']} ({'PASS' if r['after_pass'] else 'FAIL'}) | {r['movement']} |" for r in rows]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
