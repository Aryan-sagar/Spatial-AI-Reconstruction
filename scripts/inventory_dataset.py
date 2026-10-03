"""Inventory every scan under benchmark/ (read-only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from applied_ai.cli import main

root = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark")
rc = 0
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    rc |= main(["inspect", "--input", str(d)])
sys.exit(rc)
