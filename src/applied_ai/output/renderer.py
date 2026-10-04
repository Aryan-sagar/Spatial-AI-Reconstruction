"""2-D plan renderer (matplotlib). Geometry in scene.json stays authoritative; this only draws it."""
from __future__ import annotations

from pathlib import Path

import logging
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path, **kw) -> Path:
    """Save a figure; if the target cannot be written (file locked by a viewer / OneDrive sync, odd path), fall back to a
    timestamped sibling instead of aborting the run. scene.json is the source of truth and is already written."""
    path = Path(path)
    try:
        fig.savefig(path, **kw)
        return path
    except OSError as e:
        alt = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        logging.getLogger(__name__).warning("could not write %s (%s); wrote %s instead", path, e, alt)
        fig.savefig(alt, **kw)
        return alt


def _rot(xy, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.asarray(xy, float) @ R.T


def render_plan(scene: dict, out_png: Path, out_svg: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    banners: list[str] = []
    for room in scene["rooms"]:
        rot = -room["plan_rotation_deg"]
        poly = _rot(room["polygon"], rot)
        ax.fill(poly[:, 0], poly[:, 1], color="#eef3f8", zorder=0)
        ax.plot(*np.vstack((poly, poly[:1])).T, color="#9aa7b4", lw=1, zorder=1)
        for w in room["walls"]:
            g = w["geometry"]["final"]
            p = _rot([g["start"], g["end"]], rot)
            ax.plot(p[:, 0], p[:, 1], color="#1f2d3d", lw=3, solid_capstyle="butt", zorder=3)
            m = p.mean(0)
            L = w["length"]
            if L["value"] is not None:
                d = p[1] - p[0]; nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) or 1)
                ctr = np.mean(_rot(room["polygon"], rot), axis=0)
                if (m - ctr) @ nrm < 0:
                    nrm = -nrm
                ax.text(*(m + 0.25 * nrm), f"{L['value']:.2f} m\n±{(L['confidence_interval']['upper'] - L['confidence_interval']['lower']) / 2:.2f}",
                        ha="center", va="center", fontsize=7, color="#33414f")
        for o in room["openings"]:
            wall = next(w for w in room["walls"] if w["id"] == o["parent_wall"])
            g = wall["geometry"]["final"]
            s, e = np.array(g["start"]), np.array(g["end"])
            t = (e - s) / np.linalg.norm(e - s)
            c = np.array(g["centroid"])
            a, b = c + o["position"]["start_s"] * t, c + o["position"]["end_s"] * t
            p = _rot([a, b], rot)
            ax.plot(p[:, 0], p[:, 1], color="#e07a1f" if o["type"] != "window" else "#2a7fd4", lw=5, solid_capstyle="butt", zorder=4)
        c = poly.mean(0)
        area, ch = room["floor_area"], room["ceiling_height"]
        ch_txt = f"{ch['value']:.2f} m" if ch["value"] is not None else "ceiling unobserved"
        if area.get("reliable", True):
            ax.text(*c, f"{room['id']}\n{area['value']:.1f} m² (±{(area['confidence_interval']['upper'] - area['confidence_interval']['lower']) / 2:.1f})\nh = {ch_txt}",
                    ha="center", va="center", fontsize=9)
        else:
            ci = area["confidence_interval"]
            ax.plot(*np.vstack((poly, poly[:1])).T, color="#c0392b", lw=1.5, ls="--", zorder=2)
            ax.text(*c, f"{room['id']}\nAREA UNRELIABLE\n{area['value']:.1f} m² (range {ci['lower']:.1f}-{ci['upper']:.1f})\nh = {ch_txt}",
                    ha="center", va="center", fontsize=9, color="#c0392b")
            banners.append(f"{room['id']} LAYOUT UNRELIABLE: " + "; ".join(room.get("layout_unreliable_reasons", [])))
    ax.margins(0.18); ax.set_aspect("equal"); ax.set_xlabel("m"); ax.set_ylabel("m")
    ax.set_title(f"{scene['capture_id']}  [{scene['tier']}]  schema {scene['schema_version']}", fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if banners:   # wrapped to the figure width, below the axes, so long reason lists are never clipped
        import textwrap
        txt = "\n".join(textwrap.fill(b, 120) for b in banners)
        fig.subplots_adjust(bottom=0.08 + 0.025 * txt.count("\n"))
        fig.text(0.5, 0.005, txt, ha="center", va="bottom", fontsize=6.5, color="#c0392b")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    _save(fig, out_png, dpi=150); _save(fig, out_svg)
    plt.close(fig)


def render_topdown(path: Path, wall_cells, floor_uv, cam_uv, raw_walls, final_walls, pruned_ids) -> None:
    """Debug view in floor-plane coordinates: wall-band cells (grey), floor evidence (blue), camera path (red), walls before/after pruning."""
    fig, ax = plt.subplots(figsize=(10, 9))
    if len(floor_uv):
        sub = floor_uv[:: max(len(floor_uv) // 60000, 1)]
        ax.scatter(sub[:, 0], sub[:, 1], s=0.6, c="#9ec5e8", marker=".", linewidths=0, label="floor evidence")
    sub = wall_cells[:: max(len(wall_cells) // 60000, 1)]
    ax.scatter(sub[:, 0], sub[:, 1], s=0.8, c="#555555", marker=".", linewidths=0, label="wall-band cells")
    ax.plot(cam_uv[:, 0], cam_uv[:, 1], "-", c="#d62728", lw=1, label="camera path")
    ax.plot(cam_uv[0, 0], cam_uv[0, 1], "o", c="#d62728")
    for w in raw_walls:
        a = w["centroid"] + w["s_min"] * w["direction"]; b = w["centroid"] + w["s_max"] * w["direction"]
        dropped = w["id"] in pruned_ids
        ax.plot([a[0], b[0]], [a[1], b[1]], "--" if dropped else "-", c="#ff7f0e" if dropped else "#2ca02c", lw=2)
        m = (a + b) / 2
        ax.text(m[0], m[1], f"{w['id']}{' (pruned)' if dropped else ''}", fontsize=7)
    ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.legend(loc="upper right", fontsize=7, markerscale=8)
    ax.set_title("top-down debug (floor-plane coordinates, metres)")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); _save(fig, path, dpi=130); plt.close(fig)
