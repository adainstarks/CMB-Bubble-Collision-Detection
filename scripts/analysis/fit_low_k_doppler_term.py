"""Template-independent check: fit CAMB's low-k temperature transfer as A*j_l(kD) + B*k*j_l'(kD).

Long-wavelength expectation: A = -1/5 (Sachs-Wolfe), B = -tau*/5 (Doppler from v = (tau*/5) grad zeta),
tau* = conformal time at recombination. 5*|B| is the Doppler plateau in depth-equivalent Mpc.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import spherical_jn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import camb
import collision_template_transfer as ct

tr = ct.camb_transfer(lmax=200, accuracy_boost=3.0)
P = ct.PLANCK_2018
pars = camb.CAMBparams()
pars.set_cosmology(H0=P["H0"], ombh2=P["ombh2"], omch2=P["omch2"], tau=P["tau"])
res = camb.get_background(pars)
zstar = res.get_derived_params()["zstar"]
tau_star = float(res.conformal_time(zstar))
print(f"D* = {tr.d_star:.1f} Mpc, conformal time at recombination tau* = {tau_star:.1f} Mpc, r_s = {res.get_derived_params()['rstar']:.1f} Mpc")
print(f"{'l':>4s} {'k range (1/Mpc)':>22s} {'A (expect -0.20)':>18s} {'B Mpc':>10s} {'5|B| Mpc':>10s} {'fit residual':>13s}")
for ell in (10, 20, 40, 60):
    i = int(np.nonzero(tr.ells == ell)[0][0])
    kmax = (ell + 25.0) / tr.d_star
    sel = (tr.k > 0.2 * ell / tr.d_star) & (tr.k < kmax)
    k = tr.k[sel]; y = tr.delta_t[i][sel]
    x = k * tr.d_star
    basis = np.stack([spherical_jn(ell, x), k * spherical_jn(ell, x, derivative=True)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    resid = float(np.sqrt(np.mean((basis @ coef - y) ** 2)) / np.sqrt(np.mean(y ** 2)))
    print(f"{ell:4d} {k[0]:10.2e}..{k[-1]:9.2e} {coef[0]:18.4f} {coef[1]:10.1f} {5*abs(coef[1]):10.1f} {resid:13.3f}")
