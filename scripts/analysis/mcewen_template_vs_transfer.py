"""Are the defaults of the published code the transfer-function ramp of a 20 degree disc?

Notebook entries (ao) (pre-registration, pushed before this script was written) and (ap) (results).

The collision template of McEwen et al.'s public code (COMB, ``comb_tmplmap_bubble``) has the defaults
zc = 0.2 z0, cut-off out to 1.1 theta_c, theta_c = 20 deg. The transfer-function ramp of a 20 deg disc has
0.21 of its central value at the causal edge and reaches zero near 21.7 deg (entries (f), (g)). This script
measures how close the two templates are at every size, the way a matched filter sees them.

Definitions (all in ``compute``)
--------------------------------
Set-up: a data set of ``fullsky_filter_search.py`` (default "planck": Planck 2018 parameters, 5 arcmin beam
times the Nside 256 pixel window, 30 uK-arcmin white noise, l = 2 to 256). For two templates with Legendre
coefficients a_l, b_l (the window is applied to both),

    <a, b> = sum_{l=2}^{lmax} (2l+1)/(4 pi) a_l b_l W_l^2 / P_l,     O(a, b) = <a, b> / sqrt(<a, a> <b, b>).

O is the share of the best possible significance that a filter matched to a reaches on a signal of shape b
at the same centre (the quantity of the edge table of entry (ak)). Amplitudes drop out.

Templates at one nominal size theta_c:
  T        transfer-function ramp, wall at D* cos(theta_c) (``collision_template_transfer``);
  C(zc, f) their family: [1 - (1 - zc)(1 - cos t) / (1 - cos theta_c)] times the cut-off of their code,
           exp(1 - 1/(1 - s^2)) with s = (t - theta_c) / ((f - 1) theta_c); defaults C(0.2, 1.1);
  Z        the z_crit = 0 cosine, cut at theta_c (= C(0, 1)).
C is linear in zc, C = A_f + zc B_f, so for a given f the zc of largest overlap has a closed form.

Items: (1) O(C(0.2, 1.1), T) and O(Z, T) per size; (2) the zc and f of largest O(C(zc, f), T) per size, f from
1.00 to 2.00 in steps of 0.005; (1b) C(0.2, 1.1) at 20 deg against transfer templates of 10 to 30 deg; (1c)
item 1 with a second data set; (2b) zc and f fitted in real space (least squares between the profiles, both
scaled to 1 at the centre, uniform weight in angle from 0 to 1.5 theta_c); (3) the profiles for the picture.

Usage (from the project root)
-----------------------------
    python scripts/analysis/mcewen_template_vs_transfer.py compute --out report.json        (needs healpy, camb)
    python scripts/analysis/mcewen_template_vs_transfer.py plot report.json figure.png      (numpy, matplotlib)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SIZES = [5.0, 7.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0, 30.0, 40.0, 50.0, 60.0]
F_GRID = np.round(np.arange(1.0, 2.0 + 1e-9, 0.005), 3)
ZC_GRID = np.round(np.arange(-0.5, 1.5 + 1e-9, 0.005), 3)
DEFAULT_ZC, DEFAULT_F = 0.2, 1.1
PICTURE_SIZES = [10.0, 20.0, 40.0]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # slots 1 to 3 of the reference palette (validated together)


def schwartz(theta, t1, t2):
    """Cut-off of ``comb_tmplmap_bubble``: 1 up to t1, 0 from t2, exp(1 - 1/(1 - s^2)) in between."""
    theta = np.asarray(theta, dtype=np.float64)
    if t2 <= t1:
        return (theta <= t1).astype(np.float64)
    s = np.clip((theta - t1) / (t2 - t1), 0.0, 1.0)
    out = np.zeros_like(s)
    inner = s < 1.0
    out[inner] = np.exp(1.0 - 1.0 / (1.0 - s[inner] ** 2))
    return out


def family_inside(theta_c, lmax):
    """Legendre coefficients 2 pi int f P_l dmu of A = (mu - mu_c)/(1 - mu_c) and B = (1 - mu)/(1 - mu_c) over the
    disc itself (exact: polynomials of degree l + 1)."""
    mu_c = math.cos(theta_c)
    x, w = np.polynomial.legendre.leggauss(lmax // 2 + 8)
    mu = 0.5 * (1.0 - mu_c) * x + 0.5 * (1.0 + mu_c)
    v = np.polynomial.legendre.legvander(mu, lmax)
    jac = 0.5 * (1.0 - mu_c) * w
    return (2.0 * math.pi * ((jac * (mu - mu_c) / (1.0 - mu_c)) @ v), 2.0 * math.pi * ((jac * (1.0 - mu) / (1.0 - mu_c)) @ v))


def family_cutoff(theta_c, lmax, f):
    """The same two functions times the cut-off, over the cut-off region theta_c to f theta_c."""
    if f <= 1.0 + 1e-12:
        return np.zeros(lmax + 1), np.zeros(lmax + 1)
    mu_c = math.cos(theta_c)
    t2 = min(f * theta_c, math.pi)
    x, w = np.polynomial.legendre.leggauss(max(400, lmax))
    a = math.cos(t2)
    mu = 0.5 * (mu_c - a) * x + 0.5 * (mu_c + a)
    v = np.polynomial.legendre.legvander(mu, lmax)
    jac = 0.5 * (mu_c - a) * w * schwartz(np.arccos(mu), theta_c, t2)
    return (2.0 * math.pi * ((jac * (mu - mu_c) / (1.0 - mu_c)) @ v), 2.0 * math.pi * ((jac * (1.0 - mu) / (1.0 - mu_c)) @ v))


def best_zc(u, v, a, b, c, s):
    """Largest of (u + z v) / sqrt((a + 2 b z + c z^2) s) over real z; returns (overlap, z); z is None when the
    largest value is the limit of large |z| (the template is then B alone)."""
    cands = []
    den = v * b - u * c
    if abs(den) > 1e-300:
        z = (u * b - v * a) / den
        q = a + 2.0 * b * z + c * z * z
        if q > 0.0:
            cands.append(((u + z * v) / math.sqrt(q * s), z))
    cands.append((abs(v) / math.sqrt(c * s), None))
    return max(cands, key=lambda t: t[0])


class Setup:
    def __init__(self, dataset, data_dir):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fullsky_filter_search as fs

        self.fs = fs
        ds = fs.load_dataset(dataset, Path(data_dir), 1)
        self.name = dataset
        self.lmax = int(ds["lmax"])
        self.window = np.asarray(ds["window"], dtype=np.float64)
        ells = np.arange(self.lmax + 1)
        self.weight = np.where(ells >= 2, (2 * ells + 1) / (4 * math.pi) * self.window**2 / ds["total_power"], 0.0)
        self.synth = np.where(ells >= 2, (2 * ells + 1) / (4 * math.pi) * self.window, 0.0)      # coefficients -> profile
        self.tr = fs.ct.camb_transfer(lmax=self.lmax, params=ds["transfer_params"])

    def dot(self, a, b):
        return float(np.sum(self.weight * a * b))

    def overlap(self, a, b):
        return self.dot(a, b) / math.sqrt(self.dot(a, a) * self.dot(b, b))

    def transfer(self, size_deg):
        """Transfer-function ramp with a positive centre, as ``finish_template`` of the search makes it (a
        positive curvature ramp gives a cold disc, so the raw coefficients have the other sign)."""
        b = np.zeros(self.lmax + 1)
        b[self.tr.ells] = self.fs.ct.b_ell(self.tr.ells, self.tr.k, self.tr.delta_t, self.tr.d_star * math.cos(math.radians(size_deg)),
                                           a_ramp=1.0 / self.tr.d_star)
        return b if float(np.sum(self.synth * b)) >= 0.0 else -b

    def profile(self, coeff, theta):
        return np.polynomial.legendre.legval(np.cos(theta), self.synth * coeff)


def measure_size(S, size, with_fit=True):
    tc = math.radians(size)
    T = S.transfer(size)
    a_in, b_in = family_inside(tc, S.lmax)
    a_d, b_d = family_cutoff(tc, S.lmax, DEFAULT_F)
    code = a_in + a_d + DEFAULT_ZC * (b_in + b_d)
    # guard: the family as built here must be the two templates of the search script
    ref_code = S.fs.cosine_edge_coefficients(tc, S.lmax, DEFAULT_ZC, (1.0, DEFAULT_F, "schwartz"))
    ref_z = S.fs.cosine_coefficients(tc, S.lmax)
    assert np.max(np.abs(code - ref_code)) <= 1e-9 * np.max(np.abs(ref_code)), "family differs from cosine_edge_coefficients"
    assert np.max(np.abs(a_in - ref_z)) <= 1e-9 * np.max(np.abs(ref_z)), "A differs from cosine_coefficients"
    row = {"size_deg": size, "overlap_code_default": S.overlap(code, T), "overlap_zcrit0": S.overlap(a_in, T),
           "overlap_code_default_with_zcrit0": S.overlap(code, a_in)}
    if not with_fit:
        return row, T, code, a_in
    s = S.dot(T, T)
    theta = np.linspace(0.0, 1.5 * tc, 601)
    pt = S.profile(T, theta)
    pt = pt / pt[0]
    scan, real = [], []
    for f in F_GRID:
        a_c, b_c = family_cutoff(tc, S.lmax, float(f))
        A, B = a_in + a_c, b_in + b_c
        o, z = best_zc(S.dot(A, T), S.dot(B, T), S.dot(A, A), S.dot(A, B), S.dot(B, B), s)
        scan.append((float(f), o, z))
        pa, pb = S.profile(A, theta), S.profile(B, theta)
        centre = pa[0] + ZC_GRID * pb[0]
        model = (pa[None, :] + ZC_GRID[:, None] * pb[None, :]) / centre[:, None]
        cost = np.mean((model - pt[None, :]) ** 2, axis=1)
        cost[np.abs(centre) < 1e-6 * np.max(np.abs(pa))] = np.inf
        i = int(np.argmin(cost))
        real.append((float(f), float(ZC_GRID[i]), float(math.sqrt(cost[i]))))
    best = max(scan, key=lambda t: t[1])
    near = [t for t in scan if t[1] >= best[1] - 0.002]
    rbest = min(real, key=lambda t: t[2])
    pcode = S.profile(code, theta)
    pz = S.profile(a_in, theta)
    row.update({
        "best_fit": {"zc": best[2], "f": best[0], "overlap": best[1],
                     "default_loss": best[1] - row["overlap_code_default"],
                     "f_range_within_0p002": [near[0][0], near[-1][0]],
                     "zc_range_within_0p002": [min(t[2] for t in near if t[2] is not None), max(t[2] for t in near if t[2] is not None)]
                     if any(t[2] is not None for t in near) else None},
        "best_zc_at_default_f": dict(zip(("overlap", "zc"), next((t[1], t[2]) for t in scan if abs(t[0] - DEFAULT_F) < 1e-9))),
        "best_zc_at_sharp_cut": dict(zip(("overlap", "zc"), (scan[0][1], scan[0][2]))),
        "real_space_fit": {"zc": rbest[1], "f": rbest[0], "rms": rbest[2],
                           "rms_code_default": float(math.sqrt(np.mean((pcode / pcode[0] - pt) ** 2))),
                           "rms_zcrit0": float(math.sqrt(np.mean((pz / pz[0] - pt) ** 2)))},
        "transfer_edge_over_centre": float(S.profile(T, np.array([tc]))[0] / S.profile(T, np.array([0.0]))[0]),
        "transfer_first_zero_over_size": None})
    fine = np.linspace(0.3 * tc, 2.5 * tc, 4401)
    pf = S.profile(T, fine)
    sign = np.nonzero(np.sign(pf[:-1]) != np.sign(pf[1:]))[0]
    if sign.size:
        row["transfer_first_zero_over_size"] = float(fine[sign[0]] / tc)
    return row, T, code, a_in


def cmd_compute(args):
    out = {"sizes_deg": SIZES, "default": {"zc": DEFAULT_ZC, "f": DEFAULT_F}, "f_grid": [float(F_GRID[0]), float(F_GRID[-1]), 0.005]}
    S = Setup(args.dataset, args.data_dir)
    out["setup"] = {"dataset": S.name, "lmax": S.lmax, "d_star_mpc": float(S.tr.d_star)}
    rows, keep = [], {}
    print("%s set-up, l 2..%d. size | O(code default, T) | O(z_crit = 0, T) | best fit zc, f, O | default loses | f within 0.002 | real-space fit zc, f" % (S.name, S.lmax))
    for size in SIZES:
        row, T, code, z = measure_size(S, size)
        rows.append(row)
        keep[size] = (T, code, z)
        b, r = row["best_fit"], row["real_space_fit"]
        print("  %4.0f | %.4f | %.4f | %s, %.3f, %.4f | %.4f | %.3f to %.3f | %.3f, %.3f (rms %.3f; default %.3f, z_crit = 0 %.3f) | edge/centre of T %.3f, first zero at %s of the size" % (
            size, row["overlap_code_default"], row["overlap_zcrit0"], "inf" if b["zc"] is None else "%.3f" % b["zc"], b["f"], b["overlap"],
            b["default_loss"], b["f_range_within_0p002"][0], b["f_range_within_0p002"][1], r["zc"], r["f"], r["rms"], r["rms_code_default"],
            r["rms_zcrit0"], row["transfer_edge_over_centre"], "%.3f" % row["transfer_first_zero_over_size"] if row["transfer_first_zero_over_size"] else "-"))
    out["rows"] = rows

    # verdict of entry (ao)
    o = {r["size_deg"]: r["overlap_code_default"] for r in rows}
    mid = [o[s] for s in SIZES if 10.0 <= s <= 40.0]
    peak = max(o, key=o.get)
    b20 = next(r for r in rows if r["size_deg"] == 20.0)["best_fit"]
    flat = (max(mid) - min(mid)) < 0.02
    inner = b20["zc"] is not None and 0.15 <= b20["zc"] <= 0.25 and 1.05 <= b20["f"] <= 1.15
    outer = b20["zc"] is None or not (0.10 <= b20["zc"] <= 0.30) or not (1.03 <= b20["f"] <= 1.20)
    undetermined = (not inner) and b20["default_loss"] < 0.002
    if flat:
        verdict = "inconclusive (flat)"
    elif peak in (18.0, 20.0, 22.0) and inner:
        verdict = "support"
    elif peak <= 14.0 or peak >= 30.0 or (outer and not undetermined):
        verdict = "against"
    else:
        verdict = "partial"
    out["verdict"] = {"peak_size_deg": peak, "range_10_to_40_deg": max(mid) - min(mid), "flat": bool(flat), "best_fit_20_deg": b20,
                      "best_fit_in_inner_box": bool(inner), "best_fit_outside_outer_box": bool(outer),
                      "best_fit_undetermined": bool(undetermined), "verdict": verdict}
    print("verdict of (ao): peak at %g deg; range over 10 to 40 deg %.4f (flat: %s); best fit at 20 deg zc %s, f %.3f (inner box: %s, outside outer box: %s, default loses %.4f) -> %s" % (
        peak, max(mid) - min(mid), flat, b20["zc"], b20["f"], inner, outer, b20["default_loss"], verdict))

    # 1b: the code default at 20 deg against transfer templates of other sizes
    code20 = keep[20.0][1]
    scan = []
    for size in np.arange(10.0, 30.0 + 1e-9, 0.5):
        scan.append({"transfer_size_deg": float(size), "overlap": S.overlap(code20, S.transfer(float(size)))})
    best = max(scan, key=lambda t: t["overlap"])
    out["code_default_20deg_against_transfer_sizes"] = {"scan": scan, "best_transfer_size_deg": best["transfer_size_deg"], "best_overlap": best["overlap"]}
    print("1b: the code default at 20 deg matches the transfer template of %.1f deg best (overlap %.4f); at 20 deg %.4f" % (
        best["transfer_size_deg"], best["overlap"], o[20.0]))

    # 3: profiles for the picture (l >= 2, window applied, centre scaled to 1)
    prof = {}
    for size in PICTURE_SIZES:
        tc = math.radians(size)
        theta = np.linspace(0.0, 1.6 * tc, 801)
        T, code, z = keep[size]
        pt, pc, pz = S.profile(T, theta), S.profile(code, theta), S.profile(z, theta)
        prof[str(int(size))] = {"theta_deg": np.degrees(theta).tolist(), "transfer": (pt / pt[0]).tolist(), "code_default": (pc / pc[0]).tolist(),
                                "zcrit0": (pz / pz[0]).tolist(), "centre_values_per_unit_raw_template": [float(pt[0]), float(pc[0]), float(pz[0])]}
    out["profiles"] = prof

    # 1c: item 1 with a second data set
    if args.second_dataset:
        S2 = Setup(args.second_dataset, args.data_dir)
        rows2 = []
        for size in SIZES:
            row, _, _, _ = measure_size(S2, size, with_fit=(size == 20.0))
            rows2.append(row)
        o2 = {r["size_deg"]: r["overlap_code_default"] for r in rows2}
        out["second_setup"] = {"dataset": S2.name, "lmax": S2.lmax, "d_star_mpc": float(S2.tr.d_star), "rows": rows2, "peak_size_deg": max(o2, key=o2.get)}
        print("1c, %s set-up: O(code default, T) / O(z_crit = 0, T): %s | peak at %g deg" % (
            S2.name, "  ".join("%g: %.4f / %.4f" % (r["size_deg"], r["overlap_code_default"], r["overlap_zcrit0"]) for r in rows2), max(o2, key=o2.get)))
        b = next(r for r in rows2 if r["size_deg"] == 20.0)["best_fit"]
        print("    best fit at 20 deg: zc %s, f %.3f, overlap %.4f (default loses %.4f)" % (b["zc"], b["f"], b["overlap"], b["default_loss"]))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", args.out)


def cmd_plot(args):
    rep = json.load(open(args.report))
    rows = {r["size_deg"]: r for r in rep["rows"]}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": INK2, "ytick.labelcolor": INK2})
    sizes = [int(s) for s in PICTURE_SIZES]
    fig, axes = plt.subplots(1, len(sizes), figsize=(11.0, 4.8), sharey=True, facecolor=SURFACE)
    for ax, size in zip(axes, sizes):
        p = rep["profiles"][str(size)]
        th = np.array(p["theta_deg"])
        ax.set_facecolor(SURFACE)
        ax.axhline(0.0, color=AXIS, lw=0.8, zorder=1)
        ax.axvline(size, color=AXIS, lw=0.8, zorder=1)
        ax.plot(th, p["zcrit0"], color=SERIES[2], lw=2.0, zorder=3)
        ax.plot(th, p["code_default"], color=SERIES[1], lw=2.0, zorder=4)
        ax.plot(th, p["transfer"], color=SERIES[0], lw=2.0, zorder=5)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=3)
        ax.set_xlim(0.0, 1.6 * size)
        ax.set_xlabel("Angle from the centre (degrees)")
        r = rows[float(size)]
        ax.set_title("%d° disc" % size, fontsize=10.5, color=INK, loc="left")
        ax.text(0.655, 0.885, "overlap with the\ntransfer template\ncode: %.2f\nz_crit = 0: %.2f" % (r["overlap_code_default"], r["overlap_zcrit0"]),
                transform=ax.transAxes, fontsize=8.8, color=INK2, va="top", ha="left", linespacing=1.35)
        ax.annotate("edge", xy=(size, 1.0), xytext=(size * 1.02, 1.0), fontsize=9, color=INK2, va="center")
    axes[0].set_ylabel("Temperature / central value")
    lo = min(min(min(rep["profiles"][str(s)][k]) for k in ("transfer", "code_default", "zcrit0")) for s in sizes)
    axes[0].set_ylim(min(-0.35, 1.05 * lo), 1.08)
    handles = [Line2D([], [], color=SERIES[0], lw=2.0), Line2D([], [], color=SERIES[1], lw=2.0), Line2D([], [], color=SERIES[2], lw=2.0)]
    labels = ["transfer-function ramp (this project)", "template of the published code (0.2 at the edge, cut-off to 1.1 × size)",
              "z_crit = 0 cosine (as the papers describe it)"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(args.title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.905, "All three as the search sees them: multipoles 2 to %d, same beam and pixel window, no monopole or dipole, centre scaled to 1." % rep["setup"]["lmax"],
             fontsize=9.5, color=INK2, ha="left")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.77, bottom=0.21, wspace=0.07)
    fig.savefig(args.figure, dpi=160, facecolor=SURFACE)
    print("wrote", args.figure)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--dataset", default="planck")
    c.add_argument("--second-dataset", default="wmap7w")
    c.add_argument("--data-dir", default="data")
    c.add_argument("--out", required=True)
    p = sub.add_parser("plot")
    p.add_argument("report")
    p.add_argument("figure")
    p.add_argument("--title", default="Three collision templates at 10, 20 and 40 degrees")
    args = ap.parse_args()
    cmd_compute(args) if args.command == "compute" else cmd_plot(args)


if __name__ == "__main__":
    main()
