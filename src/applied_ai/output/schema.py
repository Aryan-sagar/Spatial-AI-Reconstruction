"""Internal scene schema v0.1 (NOT the official published schema - that was not supplied). Adapter layer isolates the swap."""
from __future__ import annotations

SCHEMA_VERSION = "0.1"
STATUSES = ("observed", "estimated", "unobserved")


class PublishedSchemaAdapter:
    """Replace `convert` once the official schema is available; nothing else in the pipeline should change."""

    def convert(self, internal_scene: dict) -> dict:
        raise NotImplementedError("official published schema not supplied; internal schema 0.1 is the only output format")


def iter_measurements(node, path="scene"):
    if isinstance(node, dict):
        if "value" in node and "unit" in node and "status" in node and "method" in node:
            yield path, node
        else:
            for k, v in node.items():
                yield from iter_measurements(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from iter_measurements(v, f"{path}[{i}]")


def validate_scene(scene: dict) -> list[str]:
    """Returns a list of problems (empty = valid). Every measurement needs a coherent interval or an explicit unobserved status."""
    errs = []
    for k in ("schema_version", "capture_id", "tier", "rooms", "adjacency", "measurements", "damages", "concealed_damage_flags",
              "scope_line_items", "diagnostics", "provenance"):
        if k not in scene:
            errs.append(f"missing top-level key {k}")
    if scene.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version != {SCHEMA_VERSION}")
    n = 0
    for path, m in iter_measurements(scene):
        n += 1
        if m["status"] not in STATUSES:
            errs.append(f"{path}: bad status {m['status']}")
        if m["status"] == "unobserved":
            if m["value"] is not None:
                errs.append(f"{path}: unobserved measurement must not carry a value")
            continue
        ci = m.get("confidence_interval")
        if not ci or not all(x in ci for x in ("lower", "upper", "confidence_level")):
            errs.append(f"{path}: missing confidence interval")
        elif not (ci["lower"] <= m["value"] <= ci["upper"]):
            errs.append(f"{path}: value outside its interval")
    if n == 0:
        errs.append("scene contains no measurements")
    return errs
