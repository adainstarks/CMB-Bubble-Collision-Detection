"""Internal check of the transfer-function template: profiles of different disc sizes against depth.

Idea
----
The collision is plane-symmetric, so what the last-scattering surface shows at a point depends on how deep
that point lies behind the collision wall, d = D (cos theta - cos theta_c), and (through the Doppler term)
on the cosine of the angle to the collision axis, which is close to 1 for small discs. Discs of different
angular size must therefore trace one curve F(d). In the Sachs-Wolfe picture F(d) = d inside the wall and
0 outside. Anything the full transfer function adds (Doppler term, acoustic response of the wall, early
ISW) shows up as the difference from that line, and must be the same for every disc size.

Units: the temperature is multiplied by -5 / a (a = curvature slope), which turns the Sachs-Wolfe ramp into
the depth itself, in Mpc. The monopole and dipole that the template drops are restored for the Sachs-Wolfe
part only (the exact ramp is added back in place of its l >= 2 series), so curves of different sizes can be
compared directly; the l < 2 part of the non-Sachs-Wolfe terms stays missing (small for small discs).

Usage: python scripts/analysis/check_template_depth_collapse.py --out-json OUT.json [--plot OUT.png]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import collision_template_transfer as ct  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--lmax", type=int, default=1000)
    ap.add_argument("--sizes-deg", default="5,8,11.39,15,20,30")
    args = ap.parse_args()

    tr = ct.camb_transfer(lmax=args.lmax)
    d_star = tr.d_star
    a = 1.0 / d_star
    sw = ct.sachs_wolfe_transfer(tr.ells, tr.k, d_star, sign=-1.0)
    depth_grid = np.linspace(-600.0, 900.0, 301)
    rows = []
    for size in [float(x) for x in args.sizes_deg.split(",")]:
        tc = math.radians(size)
        r = d_star * math.cos(tc)
        theta = np.radians(np.linspace(0.0, min(3.0 * size + 20.0, 120.0), 4001))
        depth = d_star * (np.cos(theta) - math.cos(tc))
        b_full = ct.b_ell(tr.ells, tr.k, tr.delta_t, r, a_ramp=a)
        b_sw = ct.b_ell(tr.ells, tr.k, sw, r, a_ramp=a)
        keep = tr.ells >= 2
        full = ct.profile(theta, tr.ells[keep], b_full[keep])
        sw_series = ct.profile(theta, tr.ells[keep], b_sw[keep])
        sw_exact_depth = np.where(depth > 0, depth, 0.0)                    # exact ramp, in Mpc
        f_depth = -5.0 / a * (full - sw_series) + sw_exact_depth           # full profile, in Mpc of depth
        order = np.argsort(depth)
        on_grid = np.interp(depth_grid, depth[order], f_depth[order], left=np.nan, right=np.nan)
        centre_depth = d_star * (1.0 - math.cos(tc))
        rows.append({"theta_c_deg": size, "centre_depth_mpc": centre_depth,
                     "F_at_centre_mpc": float(f_depth[0]), "F_at_wall_mpc": float(np.interp(0.0, depth[order], f_depth[order])),
                     "offset_at_centre_mpc": float(f_depth[0] - centre_depth),
                     "edge_over_centre": float(np.interp(0.0, depth[order], f_depth[order]) / f_depth[0]),
                     "F_on_grid": [None if not np.isfinite(v) else float(v) for v in on_grid]})
        print(f"theta_c {size:6.2f} deg  centre depth {centre_depth:7.1f} Mpc  F(centre) {f_depth[0]:7.1f}  "
              f"offset {f_depth[0] - centre_depth:6.1f}  F(wall) {rows[-1]['F_at_wall_mpc']:6.1f}  edge/centre {rows[-1]['edge_over_centre']:.3f}")

    # spread between sizes at fixed depth, where at least three sizes reach
    stack = np.array([[np.nan if v is None else v for v in r["F_on_grid"]] for r in rows])
    n = np.sum(np.isfinite(stack), axis=0)
    spread = np.where(n >= 3, np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0), np.nan)
    sel = np.isfinite(spread) & (depth_grid >= -400) & (depth_grid <= 200)
    print(f"largest spread between sizes for depth -400..200 Mpc: {np.nanmax(spread[sel]):.1f} Mpc "
          f"(median {np.nanmedian(spread[sel]):.1f} Mpc)")
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"d_star_mpc": d_star, "depth_grid_mpc": depth_grid.tolist(), "rows": rows,
                               "max_spread_mpc_depth_-400_200": float(np.nanmax(spread[sel]))}, indent=1))
    print("wrote", out)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
        blues = ["#b9d3f2", "#8fb8ea", "#6a9fe2", "#4788da", "#2a6fc6", "#1b4f93"]  # one hue, light to dark = small to large disc
        fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=140)
        fig.patch.set_facecolor(surface)
        ax.set_facecolor(surface)
        ax.plot(depth_grid, np.where(depth_grid > 0, depth_grid, 0.0), color="#8a8983", lw=2, ls=(0, (4, 3)), label="Sachs-Wolfe ramp only")
        for i, r in enumerate(rows):
            y = np.array([np.nan if v is None else v for v in r["F_on_grid"]])
            ax.plot(depth_grid, y, color=blues[min(i, len(blues) - 1)], lw=2, label=f"{r['theta_c_deg']:g} degree disc")
        ax.axvline(0.0, color="#c9c8c2", lw=1)
        ax.set_xlim(-600, 900)
        ax.set_xlabel("depth behind the collision wall (Mpc)", color=muted)
        ax.set_ylabel("temperature, in Mpc of Sachs-Wolfe depth", color=muted)
        ax.set_title("Every disc size traces one curve: the wall adds about 400 Mpc of depth", color=ink, loc="left", fontsize=12, pad=12)
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c2")
        ax.tick_params(colors=muted)
        ax.legend(frameon=False, loc="upper left", labelcolor=ink, fontsize=9)
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, facecolor=surface)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
