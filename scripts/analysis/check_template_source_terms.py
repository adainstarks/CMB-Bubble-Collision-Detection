"""Which part of the CMB temperature source produces the edge feature of the collision template?

Uses camb.symbolic to split the scalar temperature source into monopole (Sachs-Wolfe plus density),
integrated Sachs-Wolfe, Doppler and quadrupole parts, checks that the four add up to CAMB's total, and
builds the ramp template from each part.

Usage: python scripts/analysis/check_template_source_terms.py [--lmax 600]
Needs sympy and a Fortran compiler (camb.symbolic compiles the source functions).
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import camb
import numpy as np
from camb import symbolic as cs

import collision_template_transfer as ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmax", type=int, default=600)
    args = ap.parse_args()

    mon, isw, dop, quad = cs.get_scalar_temperature_sources()
    p0 = ct.PLANCK_2018
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=p0["H0"], ombh2=p0["ombh2"], omch2=p0["omch2"], tau=p0["tau"])
    pars.InitPower.set_params(As=p0["As"], ns=p0["ns"])
    pars.set_for_lmax(args.lmax, lens_potential_accuracy=0)
    pars.WantTensors = False
    pars.DoLensing = False
    pars.Accuracy.lSampleBoost = 50.0
    pars.set_custom_scalar_sources([mon, isw, dop, quad], source_names=["monopole", "isw", "doppler", "quadrupole"])
    data = camb.get_transfer_functions(pars)
    tr = data.get_cmb_transfer_data("scalar")
    d = np.asarray(tr.delta_p_l_k, dtype=np.float64)
    ells = np.asarray(tr.L)
    k = np.asarray(tr.q)
    print("transfer array shape (sources, multipoles, k):", d.shape)
    results = camb.get_results(pars)
    d_star = float(results.comoving_radial_distance(results.get_derived_params()["zstar"]))
    names = ["total T", "E", "lensing", "monopole", "isw", "doppler", "quadrupole"][: d.shape[0]]
    if d.shape[0] >= 7:
        resid = float(np.max(np.abs(d[3:7].sum(axis=0) - d[0])) / np.max(np.abs(d[0])))
        print("max |sum of the four parts - total T| / max |total T| =", resid)

    theta = np.radians(np.linspace(0.0, 30.0, 1201))
    deg = np.degrees(theta)

    def at(x, arr):
        return float(np.interp(x, deg, arr))

    for theta_c_deg in (11.39, 20.0):
        tc = math.radians(theta_c_deg)
        r = d_star * math.cos(tc)
        a = 1.0 / d_star
        unit = -(a * d_star / 5.0) * (1.0 - math.cos(tc))
        print("\nramp, theta_c = %.2f deg, in units of the Sachs-Wolfe central amplitude" % theta_c_deg)
        print("%-12s %9s %9s %9s %9s %9s" % ("part", "centre", "half", "edge", "edge+2", "edge+4"))
        for idx, name in enumerate(names):
            if name in ("E", "lensing"):
                continue
            prof = ct.profile(theta, ells, ct.b_ell(ells, k, d[idx], r, a_ramp=a)) / unit
            vals = (at(0, prof), at(theta_c_deg / 2, prof), at(theta_c_deg, prof),
                    at(min(theta_c_deg + 2, 30), prof), at(min(theta_c_deg + 4, 30), prof))
            print("%-12s %9.3f %9.3f %9.3f %9.3f %9.3f" % ((name,) + vals))


if __name__ == "__main__":
    main()
