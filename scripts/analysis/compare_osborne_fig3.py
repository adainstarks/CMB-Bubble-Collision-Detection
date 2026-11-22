"""Which bubble does Fig. 3 of Osborne, Senatore and Smith (arXiv:1305.1970) show?

The figure is labelled r = 13886.6 Mpc, theta_bubble = 11.39 deg. With cos(theta) = r / D that is
D = 14165.5 Mpc and a collision wall d = D - r = 278.9 Mpc inside the last-scattering surface at the disc
centre. Its ramp curve has an edge-to-centre ratio of 0.158; this project's template gives 0.48 for that
depth (notebook entries (f), (g)).

This script computes the ramp and step profiles of the same template for a range of depths d = D* - r,
D* being the last-scattering distance of the cosmology the transfer function was computed for, and fits
each published curve with one free amplitude. Pre-registered in notebook entry (u).

Input: the published curves as numbers, read from the vector figure in the paper's arXiv source by
``read_osborne_fig3.py`` (columns curve, theta_deg, value).

Usage:
  python scripts/analysis/compare_osborne_fig3.py --curves work/results_2026-11-22/osborne_fig3_curves.csv \
      --out-dir runs/template_check/osborne_fig3
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import numpy as np
from scipy.interpolate import CubicSpline

import collision_template_transfer as ct

_trapz = getattr(np, "trapezoid", None) or np.trapz

R_FIGURE = 13886.6        # Mpc, from the caption
THETA_FIGURE = 11.39      # deg, from the caption
D_FIGURE = R_FIGURE / math.cos(math.radians(THETA_FIGURE))
DEPTH_FIGURE = D_FIGURE - R_FIGURE

COSMOLOGIES = {
    "planck2018": dict(ct.PLANCK_2018),
    "planck2013": {"H0": 67.11, "ombh2": 0.022068, "omch2": 0.12029, "tau": 0.0925, "As": 2.215e-9, "ns": 0.9624},
    "wmap7": {"H0": 70.4, "ombh2": 0.02260, "omch2": 0.1123, "tau": 0.087, "As": 2.441e-9, "ns": 0.963},
    "wmap9": {"H0": 70.0, "ombh2": 0.02264, "omch2": 0.1138, "tau": 0.089, "As": 2.41e-9, "ns": 0.972},
    "camb_default_2013": {"H0": 70.0, "ombh2": 0.0226, "omch2": 0.112, "tau": 0.09, "As": 2.1e-9, "ns": 0.96},
}


def background(params):
    import camb

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params["H0"], ombh2=params["ombh2"], omch2=params["omch2"], tau=params["tau"])
    pars.InitPower.set_params(As=params["As"], ns=params["ns"])
    res = camb.get_background(pars)
    der = res.get_derived_params()
    d_star = float(res.comoving_radial_distance(der["zstar"]))
    tau0 = float(res.conformal_time(0.0))
    return {"zstar": float(der["zstar"]), "D_star": d_star, "tau_0": tau0, "tau_star": tau0 - d_star,
            "r_sound": float(der["rstar"]), "D_star_minus_figure_D": d_star - D_FIGURE,
            "tau_0_minus_figure_D": tau0 - D_FIGURE, "depth_if_r_is_literal": d_star - R_FIGURE}


def template_coefficients(tr, depths, refine=8):
    """b_l of the unit ramp and the unit step for every depth; same numerics as ct.b_ell, vectorised over r."""
    ells = np.asarray(tr.ells, dtype=np.int64)
    r_values = tr.d_star - np.asarray(depths, dtype=np.float64)
    k_ext, d_ext = ct._extend_to_zero(ells, np.asarray(tr.k, dtype=np.float64), np.asarray(tr.delta_t, dtype=np.float64))
    kf = np.interp(np.linspace(0, k_ext.size - 1, (k_ext.size - 1) * refine + 1), np.arange(k_ext.size), k_ext)
    kr = kf[None, :] * r_values[:, None]
    ramp = np.zeros((r_values.size, ells.size))
    step = np.zeros((r_values.size, ells.size))
    for i, ell in enumerate(ells):
        if ell < 2:
            continue
        dd = CubicSpline(k_ext, d_ext[i])(kf)
        phase = kr - ell * math.pi / 2.0
        ramp[:, i] = -4.0 * _trapz(dd[None, :] * np.cos(phase) / kf[None, :] ** 2, kf, axis=1)
        step[:, i] = 4.0 * _trapz(dd[None, :] * np.cos(phase + math.pi / 2.0) / kf[None, :], kf, axis=1)
    return ells, r_values, ramp, step


def profiles(theta, ells, coeffs):
    """T(theta) for every row of coeffs: sum_l (2l+1)/(4 pi) b_l P_l(cos theta)."""
    lmax = int(ells.max())
    c = np.zeros((lmax + 1, coeffs.shape[0]))
    c[ells] = ((2.0 * ells + 1.0) / (4.0 * math.pi))[:, None] * coeffs.T
    return np.polynomial.legendre.legval(np.cos(theta), c)          # shape (n_depth, n_theta)


def restore_low_multipoles(theta_fit, prof_fit, theta_far, prof_far):
    """Subtract the constant and cos(theta) term fitted at theta >= 30 deg (where the true signal is zero)."""
    a = np.stack([np.ones_like(theta_far), np.cos(theta_far)], axis=1)
    coef, *_ = np.linalg.lstsq(a, prof_far.T, rcond=None)           # (2, n_depth)
    fixed = prof_fit - (coef[0][:, None] + coef[1][:, None] * np.cos(theta_fit)[None, :])
    far_left = prof_far - (a @ coef).T
    return fixed, coef, float(np.max(np.abs(far_left)))


def fit_curve(model, y):
    """One free amplitude, and (robustness) amplitude plus constant. Returns RMS / max|y| for both."""
    amp = float(np.dot(model, y) / np.dot(model, model))
    rms1 = float(np.sqrt(np.mean((y - amp * model) ** 2)) / np.max(np.abs(y)))
    a = np.stack([model, np.ones_like(model)], axis=1)
    c2, *_ = np.linalg.lstsq(a, y, rcond=None)
    rms2 = float(np.sqrt(np.mean((y - a @ c2) ** 2)) / np.max(np.abs(y)))
    return amp, rms1, float(c2[0]), float(c2[1]), rms2


def explore_smoothing(ells, depths, b_ramp, b_step, theta, theta_far, target, theta_deg):
    """NOT pre-registered (added after the depth fit was seen): the same fit with one more parameter, a
    smoothing of the model profiles, either a Gaussian beam of given FWHM or a sharp cut at l_max."""
    variants = [("gaussian_fwhm_deg", f) for f in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)]
    variants += [("sharp_lmax", l) for l in (40, 60, 80, 100, 130, 160, 200, 300, 500)]
    out = []
    print("\nNOT pre-registered: depth fit with a smoothing of the model")
    print("smoothing                 best depth (joint)   RMS ramp   RMS step   | ramp alone: depth, RMS | step alone: depth, RMS")
    for kind, value in variants:
        if kind == "gaussian_fwhm_deg":
            sigma = math.radians(value) / math.sqrt(8.0 * math.log(2.0))
            w = np.exp(-0.5 * ells * (ells + 1.0) * sigma**2)
        else:
            w = (ells <= value).astype(np.float64)
        rms = {}
        for name, coeffs in (("ramp", b_ramp), ("step", b_step)):
            c = coeffs * w[None, :]
            fixed, _, _ = restore_low_multipoles(theta, profiles(theta, ells, c), theta_far, profiles(theta_far, ells, c))
            rms[name] = np.array([fit_curve(fixed[i], target[name])[1] for i in range(depths.size)])
        joint = np.sqrt(rms["ramp"] ** 2 + rms["step"] ** 2)
        i_j, i_r, i_s = int(np.argmin(joint)), int(np.argmin(rms["ramp"])), int(np.argmin(rms["step"]))
        i_l = int(np.argmin(np.abs(depths - DEPTH_FIGURE)))
        out.append({"kind": kind, "value": value, "joint_depth": float(depths[i_j]), "rms_ramp": float(rms["ramp"][i_j]),
                    "rms_step": float(rms["step"][i_j]), "ramp_alone": [float(depths[i_r]), float(rms["ramp"][i_r])],
                    "step_alone": [float(depths[i_s]), float(rms["step"][i_s])],
                    "label_depth": [float(rms["ramp"][i_l]), float(rms["step"][i_l])],
                    "rms_ramp_by_depth": rms["ramp"].tolist(), "rms_step_by_depth": rms["step"].tolist()})
        print("%-18s %6.2f   %6.0f Mpc           %.4f     %.4f     | %6.0f Mpc, %.4f     | %6.0f Mpc, %.4f   (label depth: %.4f, %.4f)" % (
            kind, value, depths[i_j], rms["ramp"][i_j], rms["step"][i_j], depths[i_r], rms["ramp"][i_r], depths[i_s], rms["step"][i_s],
            rms["ramp"][i_l], rms["step"][i_l]))
    best = min(out, key=lambda o: math.hypot(o["rms_ramp"], o["rms_step"]))
    print("best overall: %s = %s, depth %.0f Mpc, RMS ramp %.4f, step %.4f" % (best["kind"], best["value"], best["joint_depth"], best["rms_ramp"], best["rms_step"]))
    # the table of the best variant at key angles
    kind, value = best["kind"], best["value"]
    w = np.exp(-0.5 * ells * (ells + 1.0) * (math.radians(value) / math.sqrt(8.0 * math.log(2.0))) ** 2) if kind == "gaussian_fwhm_deg" else (ells <= value).astype(np.float64)
    i_j = int(np.argmin(np.abs(depths - best["joint_depth"])))
    print("theta   published ramp / smoothed model   |   published step / smoothed model")
    rows = []
    cols = {}
    for name, coeffs in (("ramp", b_ramp), ("step", b_step)):
        c = coeffs[i_j:i_j + 1] * w[None, :]
        fixed, _, _ = restore_low_multipoles(theta, profiles(theta, ells, c), theta_far, profiles(theta_far, ells, c))
        cols[name] = fit_curve(fixed[0], target[name])[0] * fixed[0]
    for ang in (0.05, 2.0, 4.0, 5.695, 8.0, 9.0, 10.0, 11.0, 11.39, 12.0, 13.39, 15.0, 15.9):
        row = [ang] + [float(np.interp(ang, theta_deg, x)) for x in (target["ramp"], cols["ramp"], target["step"], cols["step"])]
        rows.append(row)
        print("%6.3f   %.4f / %.4f   |   %.4f / %.4f" % tuple(row))
    return {"variants": out, "best": {k: best[k] for k in ("kind", "value", "joint_depth", "rms_ramp", "rms_step")}, "best_table": rows}


def distance_without_radiation(params, z):
    """Comoving distance to redshift z for matter + Lambda only (no radiation), Mpc. Reported for comparison
    with the figure's D = 14165.5 Mpc; NOT part of the pre-registered test."""
    from scipy.integrate import quad

    h = params["H0"] / 100.0
    om = (params["ombh2"] + params["omch2"] + 0.06 / 93.14) / h**2
    val, _ = quad(lambda x: 1.0 / math.sqrt(om * (1.0 + x) ** 3 + 1.0 - om), 0.0, z, limit=400)
    return 299792.458 / params["H0"] * val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore-smoothing", action="store_true", help="add the fit with a smoothing parameter (not pre-registered)")
    ap.add_argument("--curves", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lmax", type=int, default=2000)
    ap.add_argument("--depths", default="-60:320:5", help="start:stop:step in Mpc (stop included)")
    ap.add_argument("--cosmologies", default="planck2018,wmap7")
    ap.add_argument("--theta-max", type=float, default=16.0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    published = {}
    with open(args.curves, newline="") as f:
        for row in csv.DictReader(f):
            published.setdefault(row["curve"], []).append((float(row["theta_deg"]), float(row["value"])))
    theta_deg = np.arange(0.05, args.theta_max - 0.049, 0.05)
    target = {}
    for name in ("ramp", "step"):
        pts = np.array(sorted(published[name]))
        target[name] = np.interp(theta_deg, pts[:, 0], pts[:, 1])
    print("figure label: r = %.1f Mpc, theta = %.2f deg -> D = %.1f Mpc, depth %.1f Mpc" % (R_FIGURE, THETA_FIGURE, D_FIGURE, DEPTH_FIGURE))
    print("published ramp: centre %.4f, at 11.39 deg %.4f, ratio %.4f" % (
        target["ramp"][0], np.interp(11.39, theta_deg, target["ramp"]), np.interp(11.39, theta_deg, target["ramp"]) / target["ramp"][0]))

    report = {"figure": {"r_mpc": R_FIGURE, "theta_deg": THETA_FIGURE, "D_mpc": D_FIGURE, "depth_mpc": DEPTH_FIGURE},
              "background": {}, "fits": {}, "lmax": args.lmax}
    print("\nbackground distances (Mpc):")
    for name, params in COSMOLOGIES.items():
        bg = background(params)
        report["background"][name] = bg
        bg["D_without_radiation"] = {str(z): distance_without_radiation(params, z) for z in (round(bg["zstar"], 1), 1089.0, 1100.0)}
        print("  %-18s D* %9.1f (%+7.1f vs figure D)   tau_0 %9.1f (%+7.1f)   tau* %6.1f   r_s %6.1f   depth if r literal %+7.1f"
              "   distance to z* without radiation %9.1f (%+7.1f)" % (
            name, bg["D_star"], bg["D_star_minus_figure_D"], bg["tau_0"], bg["tau_0_minus_figure_D"], bg["tau_star"],
            bg["r_sound"], bg["depth_if_r_is_literal"], bg["D_without_radiation"][str(round(bg["zstar"], 1))],
            bg["D_without_radiation"][str(round(bg["zstar"], 1))] - D_FIGURE))

    start, stop, step = (float(x) for x in args.depths.split(":"))
    depths = np.unique(np.concatenate([np.arange(start, stop + 0.5 * step, step), [round(DEPTH_FIGURE, 1)]]))
    theta = np.radians(theta_deg)
    theta_far = np.radians(np.arange(30.0, 180.01, 1.0))
    key_angles = [0.05, 2.0, 4.0, 5.695, 8.0, 9.0, 10.0, 11.0, 11.39, 12.0, 13.39, 15.0, 15.9]

    for cosmo in args.cosmologies.split(","):
        t0 = time.time()
        tr = ct.camb_transfer(lmax=args.lmax, params=COSMOLOGIES[cosmo])
        norm = ct.check_cl_normalisation(tr)
        print("\n===== %s: D* = %.1f Mpc, %d k values, %d multipoles, C_l check %s (%.0f s)" % (
            cosmo, tr.d_star, tr.k.size, tr.ells.size, {k: round(v, 4) for k, v in norm.items()}, time.time() - t0))
        t0 = time.time()
        ells, r_values, b_ramp, b_step = template_coefficients(tr, depths)
        # self-check against the module's own integrator for one depth
        j = int(np.argmin(np.abs(depths - 200.0)))
        ref_r = ct.b_ell(tr.ells, tr.k, tr.delta_t, r_values[j], a_ramp=1.0)
        ref_s = ct.b_ell(tr.ells, tr.k, tr.delta_t, r_values[j], a_step=1.0)
        err = max(float(np.max(np.abs(b_ramp[j] - ref_r)) / np.max(np.abs(ref_r))),
                  float(np.max(np.abs(b_step[j] - ref_s)) / np.max(np.abs(ref_s))))
        print("coefficients for %d depths in %.0f s; agreement with ct.b_ell at depth %.0f: %.1e" % (depths.size, time.time() - t0, depths[j], err))
        assert err < 1e-7, "vectorised integrator disagrees with ct.b_ell"

        result = {"D_star": tr.d_star, "depths": depths.tolist(), "cl_check": norm}
        model = {}
        for name, coeffs in (("ramp", b_ramp), ("step", b_step)):
            raw = profiles(theta, ells, coeffs)
            far = profiles(theta_far, ells, coeffs)
            fixed, coef, far_left = restore_low_multipoles(theta, raw, theta_far, far)
            scale = np.max(np.abs(fixed), axis=1)
            print("%s: largest residual beyond 30 deg after the constant + cos(theta) fit, relative to the profile maximum: %.1e" % (
                name, float(np.max(far_left / scale.max()))))
            model[name] = fixed
            fits = [fit_curve(fixed[i], target[name]) for i in range(depths.size)]
            result[name] = {"amplitude": [f[0] for f in fits], "rms": [f[1] for f in fits],
                            "rms_with_constant": [f[4] for f in fits], "constant": [f[3] for f in fits]}
        rms_r = np.array(result["ramp"]["rms"])
        rms_s = np.array(result["step"]["rms"])
        joint = np.sqrt(rms_r**2 + rms_s**2)
        i_r, i_s, i_j = int(np.argmin(rms_r)), int(np.argmin(rms_s)), int(np.argmin(joint))
        i_l = int(np.argmin(np.abs(depths - DEPTH_FIGURE)))
        result["best"] = {
            "ramp": {"depth": float(depths[i_r]), "rms": float(rms_r[i_r])},
            "step": {"depth": float(depths[i_s]), "rms": float(rms_s[i_s])},
            "joint": {"depth": float(depths[i_j]), "rms_ramp": float(rms_r[i_j]), "rms_step": float(rms_s[i_j])},
            "label_depth": {"depth": float(depths[i_l]), "rms_ramp": float(rms_r[i_l]), "rms_step": float(rms_s[i_l])},
        }
        d_j = float(depths[i_j])
        result["best"]["joint"]["true_radius_deg"] = math.degrees(math.acos(min(1.0, 1.0 - d_j / tr.d_star))) if d_j > 0 else 0.0
        result["best"]["joint"]["implied_D_of_figure_transfer_function"] = R_FIGURE + d_j
        print("best depth: ramp %.0f Mpc (RMS %.4f), step %.0f Mpc (RMS %.4f), joint %.0f Mpc (ramp %.4f, step %.4f)" % (
            depths[i_r], rms_r[i_r], depths[i_s], rms_s[i_s], d_j, rms_r[i_j], rms_s[i_j]))
        print("label depth %.1f Mpc: ramp RMS %.4f, step RMS %.4f" % (depths[i_l], rms_r[i_l], rms_s[i_l]))
        print("with a free constant: ramp best %.0f Mpc (RMS %.4f), step best %.0f Mpc (RMS %.4f)" % (
            depths[int(np.argmin(result["ramp"]["rms_with_constant"]))], min(result["ramp"]["rms_with_constant"]),
            depths[int(np.argmin(result["step"]["rms_with_constant"]))], min(result["step"]["rms_with_constant"])))
        print("depth   RMS ramp   RMS step")
        for i in range(0, depths.size):
            if i % 4 == 0 or i in (i_r, i_s, i_j, i_l):
                print("%6.1f   %.4f     %.4f%s" % (depths[i], rms_r[i], rms_s[i], "   <- label" if i == i_l else "   <- joint best" if i == i_j else ""))
        table = []
        print("theta   published ramp / model at joint best / model at label depth   |   published step / joint best / label depth")
        for ang in key_angles:
            row = {"theta_deg": ang}
            for name in ("ramp", "step"):
                amp_j = result[name]["amplitude"][i_j]
                amp_l = result[name]["amplitude"][i_l]
                row[name] = [float(np.interp(ang, theta_deg, target[name])), float(amp_j * np.interp(ang, theta_deg, model[name][i_j])),
                             float(amp_l * np.interp(ang, theta_deg, model[name][i_l]))]
            table.append(row)
            print("%6.3f   %.4f / %.4f / %.4f   |   %.4f / %.4f / %.4f" % (ang, *row["ramp"], *row["step"]))
        result["table"] = table
        if args.explore_smoothing:
            result["smoothing_exploration"] = explore_smoothing(ells, depths, b_ramp, b_step, theta, theta_far, target, theta_deg)
        report["fits"][cosmo] = result
        np.savez_compressed(out_dir / ("profiles_%s.npz" % cosmo), theta_deg=theta_deg, depths=depths,
                            ramp=model["ramp"], step=model["step"], published_ramp=target["ramp"], published_step=target["step"])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            for ax, name in zip(axes, ("ramp", "step")):
                ax.plot(theta_deg, target[name], "k-", lw=2, label="published (Fig. 3)")
                ax.plot(theta_deg, result[name]["amplitude"][i_j] * model[name][i_j], "r--", label="template, depth %.0f Mpc" % d_j)
                ax.plot(theta_deg, result[name]["amplitude"][i_l] * model[name][i_l], "b:", label="template, depth %.1f Mpc (label)" % depths[i_l])
                ax.axvline(THETA_FIGURE, color="0.7", lw=0.8)
                ax.set_xlabel("theta [deg]")
                ax.set_title("%s profile, %s" % (name, cosmo))
                ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / ("osborne_fig3_%s.png" % cosmo), dpi=130)
            plt.close(fig)
        except Exception as exc:  # the numbers do not depend on the plot
            print("plot skipped:", exc)

    with open(out_dir / "compare_osborne_fig3_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("\nwrote", out_dir / "compare_osborne_fig3_report.json")


if __name__ == "__main__":
    main()
