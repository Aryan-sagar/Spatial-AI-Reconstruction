# applied-ai-case-study

Offline, reproducible spatial reconstruction pipeline (photo / video / LiDAR tiers).

**Status: Phase 0** (scaffold, config, logging, dataset inventory, loaders, sync report).
Raw data goes in `benchmark/` and is never modified; all outputs go to `outputs/`.

    pip install -e .
    python -m pytest -q
    python -m applied_ai.cli inspect --input benchmark/single_scan_with_ceiling

See `docs/assumptions.md` for unresolved assumptions.
