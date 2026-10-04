"""Single CLI entry point. Phase 0 implements `inspect` only; other commands are added per phase."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .calibration.synchronization import FrameSynchronizer, SynchronizationError
from .config import config_hash, load_config
from .io.inventory import DatasetInventory
from .logging_utils import setup_logging, stage


def cmd_inspect(args, cfg) -> int:
    inv = DatasetInventory(cfg)
    timings: list = []
    with stage("inventory", timings):
        inv.scan(args.input)
    p = inv.write(args.output)
    rep = inv.report()
    sync_path = Path(args.output) / rep["scan"] / "synchronization_report.json"
    rc = 0 if inv.validate() else 1
    try:
        FrameSynchronizer(cfg).run(rep, sync_path)
    except SynchronizationError as e:
        print(f"SYNC FAILURE: {e}", file=sys.stderr)
        rc = 2
    print(json.dumps({k: v for k, v in rep.items() if k != "issues"}, indent=2)[:4000])
    print(f"inventory: {p}\nsync report: {sync_path}\nissues: {len(rep['issues'])}  config_hash: {config_hash(cfg)[:12]}")
    return rc


def cmd_run(args, cfg) -> int:
    from .reconstruction.lidar_run import run_lidar
    root = Path(args.input)
    tier = args.tier
    if tier == "auto":
        f = cfg["dataset"]["files"]
        tier = "lidar" if (root / f["odometry"]).exists() and (root / f["depth_dir"]).is_dir() else None
        if tier is None:
            raise SystemExit("cannot infer tier: only LiDAR/RGB-D scans (depth/ + odometry.csv) are supported so far; photo/video tiers are not implemented")
    if tier != "lidar":
        raise SystemExit(f"tier {tier!r} is not implemented yet (only lidar)")
    out = Path(args.output) if args.output else Path("outputs") / root.name
    setup_logging(cfg["logging"]["level"], out / "run.log")
    res = run_lidar(root, cfg, out, args.config, args.frame)
    if args.frame is None:
        r = res["rooms"][0]
        fa, ch = r["floor_area"], r["ceiling_height"]
        print(f"scene: {out / 'scene.json'}\nplan: {out / 'plan.png'}\nfloor area {fa['value']:.2f} m2 [{fa['confidence_interval']['lower']:.2f}, {fa['confidence_interval']['upper']:.2f}]")
        print("ceiling:", "unobserved" if ch["value"] is None else f"{ch['value']:.3f} m [{ch['confidence_interval']['lower']:.3f}, {ch['confidence_interval']['upper']:.3f}]")
        print("walls:", ", ".join(f"{w['id']}={w['length']['value']:.2f}m" for w in r["walls"]), "| openings:", ", ".join(f"{o['type']} {o['width']['value']:.2f}m" for o in r["openings"]) or "none")
    else:
        print(json.dumps({k: res[k] for k in ("frame_id", "points", "world_bounds")}, indent=2, default=str))
    return 0


def cmd_evaluate(args, cfg) -> int:
    from .evaluation.metrics import calibration_from_residuals, evaluate_scene, load_gt
    scene = json.loads(Path(args.scene).read_text())
    rep = json.loads(Path(args.repeat_scene).read_text()) if args.repeat_scene else None
    res = evaluate_scene(scene, load_gt(args.ground_truth), rep)
    out = Path(args.output or Path(args.scene).parent)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_results.json").write_text(json.dumps(res["gates"], indent=2))
    (out / "benchmark_results.json").write_text(json.dumps(res, indent=2))
    if args.calibrate:
        (out / "calibration.json").write_text(json.dumps(calibration_from_residuals([res]), indent=2))
    for g in res["gates"]:
        print(f"{'PASS' if g['pass'] else 'FAIL'}  {g['gate_name']}: {g['actual']} (target {g['target']}) - {g['evidence']}")
    return 0


def cmd_validate(args, cfg) -> int:
    from .output.schema import validate_scene
    errs = validate_scene(json.loads(Path(args.scene).read_text()))
    print("scene valid" if not errs else "\n".join(errs))
    return 1 if errs else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="applied_ai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="inventory + synchronization check of a scan folder")
    pi.add_argument("--input", required=True)
    pi.add_argument("--config", default=None)
    pi.add_argument("--output", default="outputs")
    pr = sub.add_parser("run", help="process one capture into scene.json + plan")
    pr.add_argument("--input", required=True); pr.add_argument("--tier", default="auto"); pr.add_argument("--config", default=None)
    pr.add_argument("--output", default=None); pr.add_argument("--frame", type=int, default=None, help="single-frame reconstruction only")
    pe = sub.add_parser("evaluate", help="compare a scene.json with ground truth and evaluate gates")
    pe.add_argument("--scene", required=True); pe.add_argument("--ground-truth", required=True); pe.add_argument("--repeat-scene", default=None)
    pe.add_argument("--output", default=None); pe.add_argument("--config", default=None); pe.add_argument("--calibrate", action="store_true")
    pv = sub.add_parser("validate", help="validate a scene.json against internal schema 0.1")
    pv.add_argument("--scene", required=True); pv.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["level"])
    return {"inspect": cmd_inspect, "run": cmd_run, "validate": cmd_validate, "evaluate": cmd_evaluate}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
