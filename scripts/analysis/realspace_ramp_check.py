"""The ramp template rebuilt in real space, by two routes that share nothing with the CAMB harmonic template.

Question (notebook entry (y)): this project's template for a collision wall 279 Mpc inside the last-scattering
surface is 2.4 times brighter at its centre than the Sachs-Wolfe ramp and has an edge-to-centre ratio of 0.48;
Fig. 3 of Osborne, Senatore and Smith shows 0.16. The difference is a plateau that the template assigns to the
motion of the plasma inside the collision region. Is that plateau real, or an error of this project?

Neither route uses CAMB, a transfer function Delta_l(k), or a sum over multipoles.

  "toy"    A fluid model written here from the textbook equations (conformal Newtonian gauge, Ma and Bertschinger
           1995): CDM, photons and baryons as one tightly coupled fluid, neutrinos as a second radiation fluid, no
           anisotropic stress, instantaneous recombination, Silk damping and the thickness of last scattering as two
           Gaussian factors. Accuracy aimed at: 10 to 20%.
  "class"  The Boltzmann code CLASS in Newtonian gauge: photon density, baryon velocity divergence and the
           potentials on a grid of times, weighted by CLASS's visibility function, plus the integrated
           Sachs-Wolfe term. The polarization (quadrupole) source is left out.

Common part. The collision perturbs zeta along one axis: zeta = a x for x > 0 (x = depth behind the wall). A
uniform gradient is unobservable, so zeta is replaced by the symmetric a |x| / 2, whose Fourier transform
-a / k^2 has no term at k = 0. For a field with transfer T(k) per unit zeta,
    density-like (even):   A(x) = -(1/pi) int dk T(k) (cos kx - 1) / k^2
    velocity along the axis: B(x) = -(1/pi) int dk V(k) sin kx / k^2,   V = theta / k,
where theta is the velocity divergence (so the velocity follows from the continuity equation and not from a
sign convention). Far outside the wall A is linear and B constant (free fall); that part is a monopole and a
dipole on the sky and is subtracted, which leaves the fields of the one-sided ramp. The temperature in
direction theta from the collision axis is
    T(theta) = sum over emission times of  g [ A(x) - cos(theta) B(x) ]  +  exp(-kappa) I(x),
    x = (tau_0 - tau) cos(theta) - r,
with A from Theta_0 + Psi, B from the baryon velocity (a receding emitter is colder) and I from d(Phi+Psi)/dtau.

Reference: the CAMB harmonic template with its monopole and dipole restored, written by
``compare_osborne_fig3.py`` (profiles_planck2018.npz).

Usage: python scripts/analysis/realspace_ramp_check.py --reference runs/template_check/osborne_fig3/profiles_planck2018.npz
           --out-dir OUT [--depths 279,135] [--no-class]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

_trapz = getattr(np, "trapezoid", None) or np.trapz
C_KMS = 299792.458
PLANCK = {"h": 0.6736, "ombh2": 0.02237, "omch2": 0.1200, "Tcmb": 2.7255, "Neff": 3.046, "zstar": 1089.9}
X_GRID = np.arange(-1500.0, 700.5, 1.0)          # Mpc, depth behind the wall
K_MAX, DK = 0.5, 1.0e-4
KF = (np.arange(int(K_MAX / DK)) + 0.5) * DK    # midpoint grid, avoids k = 0
COS_KX = None                                    # filled on first use (KF x X_GRID)
SIN_KX = None


# --------------------------------------------------------------------------------------------- real-space fields
def _trig():
    global COS_KX, SIN_KX
    if COS_KX is None:
        kx = KF[:, None] * X_GRID[None, :]
        COS_KX, SIN_KX = np.cos(kx) - 1.0, np.sin(kx)          # COS_KX holds cos(kx) - 1
    return COS_KX, SIN_KX


def on_fine_grid(k, f, low_k_power):
    """Transfer function on the fine k grid; below the first sample f is continued as k^low_k_power."""
    out = CubicSpline(k, f)(np.clip(KF, k[0], k[-1]))
    low = KF < k[0]
    out[low] = f[0] * (KF[low] / k[0]) ** low_k_power
    out[KF > k[-1]] = 0.0
    return out


def even_field(k, f):
    c, _ = _trig()
    return -(1.0 / math.pi) * DK * ((on_fine_grid(k, f, 0) / KF**2) @ c)


def odd_field(k, v):
    _, s = _trig()
    return -(1.0 / math.pi) * DK * ((on_fine_grid(k, v, 1) / KF**2) @ s)


def one_sided(field, even):
    """Subtract the free-fall form far outside the wall (linear for a density-like field, constant for a velocity)."""
    far = (X_GRID >= -1500.0) & (X_GRID <= -900.0)
    if even:
        p = np.polyfit(X_GRID[far], field[far], 1)
        out = field - np.polyval(p, X_GRID)
    else:
        out = field - field[far].mean()
    mid = (X_GRID > -900.0) & (X_GRID < -400.0)
    return out, float(np.max(np.abs(out[mid])))


def sample(field, x):
    """Field at depths x; zero beyond the grid on the outside, flat/linear continuation is not needed inside."""
    return np.interp(x, X_GRID, field, left=0.0)


# --------------------------------------------------------------------------------------------- toy fluid model
def toy_background(p=PLANCK):
    h = p["h"]
    h0 = 100.0 * h / C_KMS
    og = 2.4729e-5 * (p["Tcmb"] / 2.7255) ** 4 / h**2
    onu = og * 0.22711 * p["Neff"]
    ob, oc = p["ombh2"] / h**2, p["omch2"] / h**2
    om, orad = ob + oc, og + onu
    aeq = orad / om
    s = h0 * math.sqrt(om)

    def tau_of_a(a):
        return 2.0 / s * (math.sqrt(a + aeq) - math.sqrt(aeq))

    def a_of_tau(tau):
        return (0.5 * s * tau) ** 2 + s * math.sqrt(aeq) * tau

    a_star = 1.0 / (1.0 + p["zstar"])
    tau_star = tau_of_a(a_star)
    aa = np.linspace(1e-9, a_star, 200001)
    cs = 1.0 / np.sqrt(3.0 * (1.0 + 0.75 * ob / og * aa))
    r_s = float(_trapz(cs / (aa**2 * h0 * np.sqrt(om / aa**3 + orad / aa**4)), aa))
    return {"h0": h0, "og": og, "onu": onu, "ob": ob, "oc": oc, "om": om, "orad": orad, "aeq": aeq, "a_of_tau": a_of_tau,
            "tau_star": tau_star, "a_star": a_star, "r_s": r_s, "R_star": 0.75 * ob / og * a_star}


def toy_rhs(tau, y, k, bg):
    a = bg["a_of_tau"](tau)
    h0, og, onu, ob, oc = bg["h0"], bg["og"], bg["onu"], bg["ob"], bg["oc"]
    cal_h = h0 * math.sqrt(bg["om"] / a + bg["orad"] / a**2)
    r = 0.75 * ob / og * a
    dc, tc, dg, tg, dn, tn, phi = y
    # momentum constraint: k^2 (phi' + calH phi) = 4 pi G a^2 sum (rho + p) theta
    phidot = -cal_h * phi + 1.5 * h0**2 / k**2 * (oc / a * tc + ob / a * tg + (4.0 / 3.0) * og / a**2 * tg + (4.0 / 3.0) * onu / a**2 * tn)
    return [-tc + 3.0 * phidot, -cal_h * tc + k * k * phi,
            -(4.0 / 3.0) * tg + 4.0 * phidot, -(cal_h * r / (1.0 + r)) * tg + k * k * dg / (4.0 * (1.0 + r)) + k * k * phi,
            -(4.0 / 3.0) * tn + 4.0 * phidot, k * k * dn / 4.0 + k * k * phi,
            phidot]


def superhorizon_f(y):
    """Phi / Phi_primordial for an adiabatic mode outside the horizon, y = a / a_eq (1 early, 9/10 late), and df/dy."""
    if y < 0.05:
        return 1.0 - y / 16.0 + 7.0 * y * y / 160.0 - 21.0 * y**3 / 640.0, -1.0 / 16.0 + 7.0 * y / 80.0 - 63.0 * y * y / 640.0
    n = 16.0 * math.sqrt(1.0 + y) + 9.0 * y**3 + 2.0 * y * y - 8.0 * y - 16.0
    dn = 8.0 / math.sqrt(1.0 + y) + 27.0 * y * y + 4.0 * y - 8.0
    return n / (10.0 * y**3), (dn * y - 3.0 * n) / (10.0 * y**4)


def toy_initial(bg, k, tau_i, phi_p=-2.0 / 3.0):
    """State at tau_i (k tau_i << 1) on the exact superhorizon adiabatic solution. The leading radiation-era
    forms alone violate the density constraint at order a / a_eq, and the equations then keep Phi constant
    through equality instead of letting it fall by a tenth."""
    a = bg["a_of_tau"](tau_i)
    y = a / bg["aeq"]
    f, df = superhorizon_f(y)
    cal_h = bg["h0"] * math.sqrt(bg["om"] / a + bg["orad"] / a**2)
    phi = phi_p * f
    phidot = phi_p * df * cal_h * y
    d_rad = -2.0 * phi_p + 4.0 * (phi - phi_p)
    u = (phidot + cal_h * phi) / (1.5 * bg["h0"] ** 2 * (bg["om"] / a + (4.0 / 3.0) * bg["orad"] / a**2))   # common velocity potential
    return [0.75 * d_rad, k * k * u, d_rad, k * k * u, d_rad, k * k * u, phi]


def toy_transfer(bg, k_values):
    """Theta_0 + Phi, theta_b / k and Phi at recombination, per unit zeta (Phi_primordial = -2/3 zeta)."""
    out = np.zeros((k_values.size, 4))
    worst = 0.0
    for i, k in enumerate(k_values):
        tau_i = min(1.0e-3 / k, 0.5)
        y0 = toy_initial(bg, k, tau_i)
        sol = solve_ivp(toy_rhs, (tau_i, bg["tau_star"]), y0, args=(k, bg), method="LSODA", rtol=1e-8, atol=1e-12)
        dc, tc, dg, tg, dn, tn, phi = sol.y[:, -1]
        out[i] = dg / 4.0 + phi, tg / k, phi, dc
        # density constraint, as a check of the integration: k^2 phi + 3 calH (phi' + calH phi) = -4 pi G a^2 delta rho
        a = bg["a_star"]
        cal_h = bg["h0"] * math.sqrt(bg["om"] / a + bg["orad"] / a**2)
        phidot = toy_rhs(bg["tau_star"], sol.y[:, -1], k, bg)[6]
        lhs = k * k * phi + 3.0 * cal_h * (phidot + cal_h * phi)
        rhs = -1.5 * bg["h0"] ** 2 * (bg["oc"] / a * dc + bg["ob"] / a * 0.75 * dg + bg["og"] / a**2 * dg + bg["onu"] / a**2 * dn)
        worst = max(worst, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30))
    return out, worst


def toy_profile(bg, d_star, depths, theta, k_damp=0.14, sigma_vis=17.0):
    k = np.concatenate([np.geomspace(1.0e-5, 2.0e-3, 40, endpoint=False), np.arange(2.0e-3, K_MAX + 1e-9, 2.0e-3)])
    tr, constraint = toy_transfer(bg, k)
    mono, vel, phi = tr[:, 0], tr[:, 1], tr[:, 2]
    damp = np.exp(-(k / k_damp) ** 2)
    los = np.exp(-0.5 * (k * sigma_vis) ** 2)
    r_star = bg["R_star"]
    mono_d = ((mono + r_star * phi) * damp - r_star * phi) * los        # oscillation about the baryon-shifted zero point is damped
    vel_d = vel * damp * los
    a1, res_a = one_sided(even_field(k, mono_d), True)
    b1, res_b = one_sided(odd_field(k, vel_d), False)
    # integrated Sachs-Wolfe term after recombination, approximately: the potential keeps its shape and falls with
    # the superhorizon factor f(y) (0.936 at recombination to 0.9 late)
    p1, _ = one_sided(even_field(k, phi * los), True)
    tau_grid = np.arange(bg["tau_star"], bg["tau_star"] + 3000.0, 5.0)
    f_grid = np.array([superhorizon_f(bg["a_of_tau"](t) / bg["aeq"])[0] for t in tau_grid])
    df_grid = np.gradient(f_grid / f_grid[0], tau_grid)
    info = {"tau_star": bg["tau_star"], "r_sound": bg["r_s"], "R_star": r_star, "constraint_residual": constraint,
            "monopole_low_k": float(mono[0]), "phi_low_k": float(phi[0]), "phi_low_k_expected": -2.0 / 3.0 * superhorizon_f(bg["a_star"] / bg["aeq"])[0],
            "velocity_over_k_low_k": float(vel[0] / k[0]),
            "doppler_depth_equivalent_mpc_low_k": float(vel[0] / k[0] / mono[0]),
            "velocity_inside_at_300_mpc_times_5": float(5.0 * np.interp(300.0, X_GRID, b1)),
            "outside_residual_density": res_a / float(np.interp(279.0, X_GRID, np.abs(a1))),
            "outside_residual_velocity": res_b / float(np.interp(300.0, X_GRID, np.abs(b1)))}
    profs = {}
    for d in depths:
        x = d_star * np.cos(theta) - (d_star - d)
        isw = np.zeros(theta.size)
        for t, dfdt in zip(tau_grid, df_grid):
            isw += 2.0 * dfdt * 5.0 * sample(p1, x - (t - bg["tau_star"]) * np.cos(theta))
        profs[d] = {"total": sample(a1, x) - np.cos(theta) * sample(b1, x) + isw, "density": sample(a1, x),
                    "doppler": -np.cos(theta) * sample(b1, x), "isw": isw}
    return profs, info


# --------------------------------------------------------------------------------------------- CLASS
def class_profile(depths, theta, classy_path=None):
    import sys
    if classy_path:
        sys.path.insert(0, classy_path)
    from classy import Class

    cosmo = Class()
    cosmo.set({"h": PLANCK["h"], "omega_b": PLANCK["ombh2"], "omega_cdm": PLANCK["omch2"], "A_s": 2.1e-9, "n_s": 0.9649,
               "tau_reio": 0.0544, "N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": 0.06, "T_ncdm": 0.71611,
               "output": "mTk,vTk", "gauge": "newtonian", "z_max_pk": 3000.0, "P_k_max_1/Mpc": 0.6,
               "matter_source_in_current_gauge": "yes",
               "k_per_decade_for_pk": 40, "k_per_decade_for_bao": 150})
    cosmo.compute()
    th = cosmo.get_thermodynamics()
    print("CLASS thermodynamics columns:", sorted(th.keys()), flush=True)
    z_t, tau_t = np.asarray(th["z"]), np.asarray(th["conf. time [Mpc]"])
    g_t, ek_t = np.asarray(th["g [Mpc^-1]"]), np.asarray(th["exp(-kappa)"])
    order = np.argsort(tau_t)
    z_t, tau_t, g_t, ek_t = z_t[order], tau_t[order], g_t[order], ek_t[order]
    tau0 = float(tau_t[-1])
    rec = (z_t > 500.0) & (z_t < 2500.0)
    tau_rec = float(tau_t[rec][np.argmax(g_t[rec])])
    z_rec = float(z_t[rec][np.argmax(g_t[rec])])
    d_star = tau0 - tau_rec
    h = PLANCK["h"]

    def fields(z):
        t = cosmo.get_transfer(float(z))
        k = np.asarray(t["k (h/Mpc)"]) * h
        return k, np.asarray(t["d_g"]), np.asarray(t["d_b"]), np.asarray(t["t_b"]), np.asarray(t["phi"]), np.asarray(t["psi"])

    # sign and size of the curvature normalisation: Psi -> -(3/5) zeta at low k in matter domination
    k, dg, db, tb, phi, psi = fields(50.0)
    norm = -0.6 / psi[0]
    keys = sorted(cosmo.get_transfer(50.0).keys())
    # emission times: every 2 Mpc through recombination, then every 5 Mpc for the integrated Sachs-Wolfe term
    tau_a, tau_b, tau_c = float(np.interp(2200.0, z_t[::-1], tau_t[::-1])), float(np.interp(600.0, z_t[::-1], tau_t[::-1])), tau_rec + 900.0
    taus = np.concatenate([np.arange(tau_a, tau_b, 2.0), np.arange(tau_b, tau_c, 5.0)])
    zs = np.interp(taus, tau_t, z_t)
    gs = np.interp(taus, tau_t, g_t)
    eks = np.interp(taus, tau_t, ek_t) / np.interp(tau_c, tau_t, ek_t)     # reionization left out: exp(-kappa) -> 1 after recombination
    w = np.gradient(taus)
    rows = [fields(z) for z in zs]
    kk = rows[0][0]
    pot = np.array([r[4] + r[5] for r in rows]) * norm                    # (n_tau, n_k)
    pot_dot = np.gradient(pot, taus, axis=0)
    # check of the meaning of t_b: continuity, d(delta_b)/dtau = -theta_b + 3 dphi/dtau, at the time of last scattering
    j = int(np.argmin(np.abs(taus - tau_rec)))
    db_dot = (rows[j + 1][2] - rows[j - 1][2]) / (taus[j + 1] - taus[j - 1])
    phi_dot = (rows[j + 1][4] - rows[j - 1][4]) / (taus[j + 1] - taus[j - 1])
    sel = (kk > 0.005) & (kk < 0.1)
    continuity = float(np.max(np.abs(db_dot[sel] + rows[j][3][sel] - 3.0 * phi_dot[sel])) / np.max(np.abs(rows[j][3][sel])))
    info = {"tau_0": tau0, "tau_rec": tau_rec, "z_rec": z_rec, "D_star": d_star, "normalisation": float(norm), "n_k": int(kk.size),
            "k_max": float(kk.max()), "n_times": int(taus.size), "visibility_integral": float(np.sum(gs * w)),
            "continuity_residual": continuity, "transfer_keys": keys,
            "monopole_low_k_at_rec": float((rows[j][1][0] / 4.0 + rows[j][5][0]) * norm),
            "doppler_depth_equivalent_mpc_low_k": float((rows[j][3][0] / kk[0] ** 2) / (rows[j][1][0] / 4.0 + rows[j][5][0]))}
    profs = {d: {"total": np.zeros(theta.size), "density": np.zeros(theta.size), "doppler": np.zeros(theta.size),
                 "isw": np.zeros(theta.size)} for d in depths}
    worst = 0.0
    for i in range(taus.size):
        kq, dgq, _, tbq, _, psiq = rows[i]
        a1 = b1 = None
        if gs[i] * w[i] > 1e-7:
            a1, ra = one_sided(even_field(kq, (dgq / 4.0 + psiq) * norm), True)
            b1, rb = one_sided(odd_field(kq, tbq / kq * norm), False)
            worst = max(worst, ra / max(float(np.interp(279.0, X_GRID, np.abs(a1))), 1e-30))
        i1, _ = one_sided(even_field(kq, pot_dot[i]), True)
        for d in depths:
            x = (tau0 - taus[i]) * np.cos(theta) - (d_star - d)
            if a1 is not None:
                dens = gs[i] * w[i] * sample(a1, x)
                dop = -gs[i] * w[i] * np.cos(theta) * sample(b1, x)
                profs[d]["density"] += dens
                profs[d]["doppler"] += dop
                profs[d]["total"] += dens + dop
            isw = eks[i] * w[i] * sample(i1, x)
            profs[d]["isw"] += isw
            profs[d]["total"] += isw
    info["outside_residual_density"] = worst
    cosmo.struct_cleanup()
    return profs, info, d_star


# --------------------------------------------------------------------------------------------- comparison
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="profiles_planck2018.npz from compare_osborne_fig3.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--depths", default="279,135")
    ap.add_argument("--classy-path", default="")
    ap.add_argument("--no-class", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depths = [float(x) for x in args.depths.split(",")]
    ref = np.load(args.reference)
    theta_deg = ref["theta_deg"]
    theta = np.radians(theta_deg)
    d_star_camb = 13872.8                       # CAMB, Planck 2018 (notebook entry (f))
    report = {"depths": depths, "theta_deg": theta_deg.tolist(), "methods": {}}

    bg = toy_background()
    print("toy model: tau* = %.1f Mpc, sound horizon %.1f Mpc, R* = %.3f" % (bg["tau_star"], bg["r_s"], bg["R_star"]), flush=True)
    toy, toy_info = toy_profile(bg, d_star_camb, depths, theta)
    print("toy model:", json.dumps(toy_info), flush=True)
    report["methods"]["toy"] = {"info": toy_info}
    results = {"toy": toy}
    if not args.no_class:
        cls, cls_info, d_star_class = class_profile(depths, theta, args.classy_path or None)
        print("CLASS:", json.dumps({k: v for k, v in cls_info.items() if k != "transfer_keys"}), flush=True)
        print("CLASS transfer keys:", cls_info["transfer_keys"], flush=True)
        report["methods"]["class"] = {"info": cls_info}
        results["class"] = cls

    for d in depths:
        i_ref = int(np.argmin(np.abs(ref["depths"] - d)))
        assert abs(ref["depths"][i_ref] - d) < 0.5
        sw = -d / 5.0                                               # Sachs-Wolfe central value per unit ramp amplitude
        camb = ref["ramp"][i_ref] / sw
        theta_c = math.degrees(math.acos(1.0 - d / d_star_camb))
        angles = [0.05, 0.5 * theta_c, theta_c, theta_c + 2.0, theta_c + 4.0]
        print("\ndepth %.0f Mpc (disc radius %.2f deg), in units of the Sachs-Wolfe central value" % (d, theta_c))
        print("  angle (deg):                 " + "  ".join("%7.2f" % a for a in angles) + "   edge/centre   largest difference from CAMB / CAMB centre")
        print("  CAMB harmonic template:      " + "  ".join("%7.3f" % np.interp(a, theta_deg, camb) for a in angles)
              + "   %.3f" % (np.interp(theta_c, theta_deg, camb) / camb[0]))
        entry = {"theta_c_deg": theta_c, "angles_deg": angles, "camb": [float(np.interp(a, theta_deg, camb)) for a in angles]}
        for name, res in results.items():
            tot = res[d]["total"] / sw
            diff = float(np.max(np.abs(tot - camb)) / abs(camb[0]))
            print("  %-28s " % (name + ", total:") + "  ".join("%7.3f" % np.interp(a, theta_deg, tot) for a in angles)
                  + "   %.3f         %.3f" % (np.interp(theta_c, theta_deg, tot) / tot[0], diff))
            entry[name] = {"total": [float(np.interp(a, theta_deg, tot)) for a in angles], "max_difference_over_camb_centre": diff}
            for part in ("density", "doppler", "isw"):
                if part in res[d]:
                    pr = res[d][part] / sw
                    print("  %-28s " % (name + ", " + part + ":") + "  ".join("%7.3f" % np.interp(a, theta_deg, pr) for a in angles))
                    entry[name][part] = [float(np.interp(a, theta_deg, pr)) for a in angles]
        report["depth_%g" % d] = entry
        np.savez_compressed(out_dir / ("realspace_profiles_depth%g.npz" % d), theta_deg=theta_deg, camb=camb,
                            **{name: res[d]["total"] / sw for name, res in results.items()})
    with open(out_dir / "realspace_ramp_check_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print("\nwrote", out_dir / "realspace_ramp_check_report.json")


if __name__ == "__main__":
    main()
