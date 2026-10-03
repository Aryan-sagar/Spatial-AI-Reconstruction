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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="applied_ai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="inventory + synchronization check of a scan folder")
    pi.add_argument("--input", required=True)
    pi.add_argument("--config", default=None)
    pi.add_argument("--output", default="outputs")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["level"])
    return {"inspect": cmd_inspect}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
