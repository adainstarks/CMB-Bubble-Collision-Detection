"""Three checks of the transfer-function collision template (notebook entry on the template rebuild).

1. Integrator self-test, no CAMB: with a pure Sachs-Wolfe transfer function the harmonic template must
   reproduce the analytic real-space ramp and step (monopole and dipole removed, same lmax).
2. CAMB normalisation: C_l rebuilt from the transfer functions must equal CAMB's own C_l.
3. Physics: ramp and step profiles through the full transfer function, compared with the Sachs-Wolfe
   profiles. Osborne, Senatore, Smith (arXiv:1305.1964) state the transfer function "corrects the ramp
   profile by 10% and the step profile at order unity".

Usage: python scripts/analysis/check_collision_template.py --out-dir OUT [--lmax 600]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import numpy as np

import collision_template_transfer as ct


def legendre_truncate(func_mu, lmax, drop_low=True):
    mu, wts = np.polynomial.legendre.leggauss(6 * lmax + 128)
    g = func_mu(mu)
    coeff = np.zeros(lmax + 1)
    for ell in range(lmax + 1):
        c = np.zeros(ell + 1)
        c[ell] = 1.0
        coeff[ell] = (2 * ell + 1) / 2.0 * np.sum(wts * g * np.polynomial.legendre.legval(mu, c))
    if drop_low:
        coeff[:2] = 0.0
    return coeff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lmax", type=int, default=600)
    ap.add_argument("--skip-camb", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    theta = np.radians(np.linspace(0.0, 90.0, 1801))

    # ---- check 1: integrator against the analytic Sachs-Wolfe result --------------------------------
    d_star = 13900.0
    lmax1 = 200
    ells1 = np.arange(2, lmax1 + 1)
    k1 = np.linspace(2e-6, (lmax1 * 6.0) / d_star, 12000)
    sw = ct.sachs_wolfe_transfer(ells1, k1, d_star, sign=-1.0)
    print("check 1: Sachs-Wolfe transfer, harmonic template against analytic profile (lmax %d)" % lmax1)
    report["check1"] = []
    for theta_c_deg in (10.0, 25.0, 60.0):
        tc = math.radians(theta_c_deg)
        r = d_star * math.cos(tc)
        a = 1.0 / d_star
        b = ct.b_ell(ells1, k1, sw, r, a_ramp=a, refine=4)
        got = ct.profile(theta, ells1, b)
        amp = -(a * d_star / 5.0) * (1.0 - math.cos(tc))          # T = -zeta/5 at the disc centre
        mu_c = math.cos(tc)
        want = np.polynomial.legendre.legval(np.cos(theta), legendre_truncate(lambda mu: np.where(mu > mu_c, -(a * d_star / 5.0) * (mu - mu_c), 0.0), lmax1))
        err_ramp = float(np.max(np.abs(got - want)) / abs(amp))
        bs = ct.b_ell(ells1, k1, sw, r, a_step=1.0, refine=4)
        got_s = ct.profile(theta, ells1, bs)
        # exact Legendre coefficients of a top-hat: f_l = c/2 [P_{l-1}(mu_c) - P_{l+1}(mu_c)]; a quadrature
        # reference is not accurate enough for a discontinuous function (it showed a spurious 4-8% "error")
        from scipy.special import eval_legendre
        cs = np.zeros(lmax1 + 1)
        cs[2:] = (-1.0 / 5.0) / 2.0 * (eval_legendre(ells1 - 1, mu_c) - eval_legendre(ells1 + 1, mu_c))
        want_s = np.polynomial.legendre.legval(np.cos(theta), cs)
        err_step = float(np.max(np.abs(got_s - want_s)) / (1.0 / 5.0))
        print(f"  theta_c {theta_c_deg:5.1f} deg: ramp max error {err_ramp:.2e} of the central amplitude; step max error {err_step:.2e} of the step height")
        report["check1"].append({"theta_c_deg": theta_c_deg, "ramp_max_err_rel": err_ramp, "step_max_err_rel": err_step})

    if args.skip_camb:
        (out_dir / "collision_template_check.json").write_text(json.dumps(report, indent=1))
        return

    # ---- check 2: CAMB transfer normalisation ---------------------------------------------------------
    tr = ct.camb_transfer(lmax=args.lmax)
    dk = np.diff(tr.k)
    print(f"\ncheck 2: CAMB transfer: {tr.ells.size} multipoles ({tr.ells.min()}..{tr.ells.max()}), {tr.k.size} k values "
          f"from {tr.k[0]:.2e} to {tr.k[-1]:.3f} /Mpc, median dk {np.median(dk):.2e}; D* = {tr.d_star:.1f} Mpc; "
          f"oscillation period 2pi/D* = {2 * math.pi / tr.d_star:.2e}")
    ratios = ct.check_cl_normalisation(tr)
    print("  C_l(from transfer) / C_l(CAMB):", {k: round(v, 4) for k, v in ratios.items()})
    report["check2"] = {"n_ell": int(tr.ells.size), "n_k": int(tr.k.size), "k_min": float(tr.k[0]), "k_max": float(tr.k[-1]),
                        "median_dk": float(np.median(dk)), "d_star_mpc": tr.d_star, "cl_ratio": ratios}
    # sign of the low-l transfer relative to j_l(kD)/5
    i10 = int(np.nonzero(tr.ells == 10)[0][0])
    swj = ct.sachs_wolfe_transfer(np.array([10]), tr.k, tr.d_star, sign=1.0)[0]
    sel = tr.k < 30.0 / tr.d_star
    sw_sign = float(np.sign(np.sum(tr.delta_t[i10][sel] * swj[sel])))
    amp_ratio = float(np.sum(tr.delta_t[i10][sel] * swj[sel]) / np.sum(swj[sel] ** 2))
    print(f"  low-l transfer relative to j_l(kD)/5 at l=10: sign {sw_sign:+.0f}, least-squares amplitude ratio {amp_ratio:+.3f}")
    report["check2"]["sw_sign"] = sw_sign
    report["check2"]["sw_amp_ratio_l10"] = amp_ratio

    # ---- check 3: full-transfer profiles against Sachs-Wolfe profiles --------------------------------
    sw_tr = ct.sachs_wolfe_transfer(tr.ells, tr.k, tr.d_star, sign=sw_sign)
    print("\ncheck 3: profiles through the full transfer function, in units of the Sachs-Wolfe central amplitude")
    print(f"{'theta_c':>8s} {'model':>6s} {'centre':>8s} {'0.5 theta_c':>12s} {'edge-2deg':>10s} {'edge':>8s} {'edge+2deg':>10s} {'rms diff inside':>16s}")
    report["check3"] = []
    curves = {"theta_deg": np.degrees(theta)}
    for theta_c_deg in (10.0, 20.0, 40.0, 70.0):
        tc = math.radians(theta_c_deg)
        r = tr.d_star * math.cos(tc)
        a = 1.0 / tr.d_star
        z0_sw = sw_sign * (a * tr.d_star / 5.0) * (1.0 - math.cos(tc))
        for model, kw, unit in (("ramp", {"a_ramp": a}, z0_sw), ("step", {"a_step": 1.0}, sw_sign / 5.0)):
            full = ct.profile(theta, tr.ells, ct.b_ell(tr.ells, tr.k, tr.delta_t, r, **kw)) / unit
            swp = ct.profile(theta, tr.ells, ct.b_ell(tr.ells, tr.k, sw_tr, r, **kw)) / unit
            at = lambda deg, arr: float(np.interp(deg, np.degrees(theta), arr))
            inside = np.degrees(theta) < theta_c_deg
            rms = float(np.sqrt(np.mean((full[inside] - swp[inside]) ** 2)))
            row = {"theta_c_deg": theta_c_deg, "model": model,
                   "full": {"centre": at(0, full), "half": at(theta_c_deg / 2, full), "edge_m2": at(theta_c_deg - 2, full), "edge": at(theta_c_deg, full), "edge_p2": at(theta_c_deg + 2, full)},
                   "sw": {"centre": at(0, swp), "half": at(theta_c_deg / 2, swp), "edge_m2": at(theta_c_deg - 2, swp), "edge": at(theta_c_deg, swp), "edge_p2": at(theta_c_deg + 2, swp)},
                   "rms_diff_inside": rms}
            report["check3"].append(row)
            f, s = row["full"], row["sw"]
            print(f"{theta_c_deg:8.1f} {model:>6s} {f['centre']:8.3f} {f['half']:12.3f} {f['edge_m2']:10.3f} {f['edge']:8.3f} {f['edge_p2']:10.3f} {rms:16.3f}   full")
            print(f"{'':8s} {'':>6s} {s['centre']:8.3f} {s['half']:12.3f} {s['edge_m2']:10.3f} {s['edge']:8.3f} {s['edge_p2']:10.3f} {'':16s}   Sachs-Wolfe")
            curves[f"{model}_{int(theta_c_deg)}_full"] = full
            curves[f"{model}_{int(theta_c_deg)}_sw"] = swp
    np.savez_compressed(out_dir / "collision_template_profiles.npz", **curves)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharex=False)
        for col, tcd in enumerate((10, 20, 40, 70)):
            for row, model in enumerate(("ramp", "step")):
                ax = axes[row, col]
                x = curves["theta_deg"]
                ax.plot(x, curves[f"{model}_{tcd}_sw"], color="0.55", lw=1.6, label="Sachs-Wolfe only")
                ax.plot(x, curves[f"{model}_{tcd}_full"], color="#1f4e9c", lw=1.8, label="full transfer function")
                ax.axvline(tcd, color="#b23a3a", lw=1.0, ls="--")
                ax.set_xlim(0, min(90, tcd * 2.2))
                ax.set_title(f"{model}, theta_c = {tcd} deg")
                ax.set_xlabel("angle from collision centre (deg)")
                if col == 0:
                    ax.set_ylabel("Delta T / T (Sachs-Wolfe central amplitude = 1)")
                ax.grid(alpha=0.3)
                if row == 0 and col == 0:
                    ax.legend(frameon=False)
        fig.suptitle(f"Collision templates from the curvature profile, lmax {args.lmax}, monopole and dipole removed")
        fig.tight_layout()
        fig.savefig(out_dir / "collision_template_profiles.png", dpi=110)
    except Exception as exc:  # the numbers above are the result; the figure is a convenience
        print("figure skipped:", exc)

    (out_dir / "collision_template_check.json").write_text(json.dumps(report, indent=1))
    print("\nwrote", out_dir)


if __name__ == "__main__":
    main()
