"""How much does the assumed signal shape matter? Fitting factors between collision templates.

Question
--------
A matched-filter bank is optimal only for the shape it was built from. The literature does not agree on the
shape (Sachs-Wolfe ramp; ramp through the full transfer function; ramp plus step; a sharp disc). If a bank
built from one shape recovers nearly all of the signal-to-noise of another, the choice hardly matters and a
shape-agnostic detector has little to add. If it does not, that is where a learned detector trained on a
family of shapes, or a wider bank, has room.

Definition
----------
For a true signal s and a filter f, both azimuthally symmetric with harmonic coefficients b_l,
    <f, s>(alpha) = sum_l (2l+1)/(4 pi) f_l s_l w_l P_l(cos alpha),   w_l = W_l^2 / (C_l W_l^2 + N_l),
is the filter's response at angle alpha from the signal centre. The fitting factor of a bank for s is
    FF = max over filters in the bank and over alpha of |<f, s>(alpha)| / sqrt(<f, f>(0) <s, s>(0)).
FF is the fraction of the ideal signal-to-noise the bank recovers (1 = no loss). The bank holds one
template type at many sizes; the maximum over alpha lets the filter sit off-centre, as in a real search.

Shapes (curvature profile through a transfer function; D = distance to last scattering)
    tf_ramp      ramp, full CAMB temperature transfer function (this project's template)
    sw_ramp      ramp, pure Sachs-Wolfe transfer (the cosine profile used in the WMAP filter searches)
    tf_step      step, full transfer function
    sw_step      step, pure Sachs-Wolfe (a sharp-edged flat disc)
    tf_mix_f     tf_ramp plus f times a step whose height equals the ramp's Sachs-Wolfe central depth

Noise: CMB (unlensed TT) plus 30 uK-arcmin white noise, 5 arcmin beam, lmax 767, monopole and dipole out.

Usage: python scripts/analysis/template_fitting_factors.py --out-json OUT.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import collision_template_transfer as ct  # noqa: E402

T_CMB_UK = 2.7255e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--lmax", type=int, default=767)
    ap.add_argument("--truth-sizes-deg", default="5,10,15,20,30,45,60")
    ap.add_argument("--bank-sizes-deg", default="2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,33,36,40,45,50,55,60,65,70,80,90")
    ap.add_argument("--mix-fractions", default="0.25,0.5,1.0")
    args = ap.parse_args()
    lmax = args.lmax
    truth_sizes = [float(x) for x in args.truth_sizes_deg.split(",")]
    bank_sizes = sorted(set([float(x) for x in args.bank_sizes_deg.split(",")] + truth_sizes))
    mixes = [float(x) for x in args.mix_fractions.split(",")]

    tr = ct.camb_transfer(lmax=lmax)
    d = tr.d_star
    sw = ct.sachs_wolfe_transfer(tr.ells, tr.k, d, sign=-1.0)
    ell = np.arange(lmax + 1, dtype=np.float64)
    cl = np.zeros(lmax + 1)
    n = min(lmax + 1, tr.cl_tt_camb.size)
    cl[:n] = tr.cl_tt_camb[:n]
    w2 = ct.gaussian_beam(lmax, 5.0) ** 2
    noise = (30.0 / T_CMB_UK * math.radians(1.0 / 60.0)) ** 2
    weight = np.where(ell >= 2, w2 / np.maximum(cl * w2 + noise, 1e-300), 0.0) * (2 * ell + 1) / (4 * math.pi)

    def coeffs(delta, r, ramp, step):
        b = np.zeros(lmax + 1)
        b[tr.ells] = ct.b_ell(tr.ells, tr.k, delta, r, a_ramp=ramp, a_step=step)
        b[:2] = 0.0
        return b

    t0 = time.time()
    base = {}  # (type, size) -> b_l for unit slope 1/D (ramp) or unit step
    for i, size in enumerate(bank_sizes):
        r = d * math.cos(math.radians(size))
        base[("tf_ramp", size)] = coeffs(tr.delta_t, r, 1.0 / d, 0.0)
        base[("sw_ramp", size)] = coeffs(sw, r, 1.0 / d, 0.0)
        base[("tf_step", size)] = coeffs(tr.delta_t, r, 0.0, 1.0)
        base[("sw_step", size)] = coeffs(sw, r, 0.0, 1.0)
        if (i + 1) % 5 == 0 or i == len(bank_sizes) - 1:
            print(f"templates {i + 1}/{len(bank_sizes)}  ({time.time() - t0:.0f} s)", flush=True)

    def shape(name, size):
        if name.startswith("tf_mix_"):
            f = float(name[len("tf_mix_"):])
            # ramp with slope 1/D has Sachs-Wolfe central depth (1 - cos theta_c) in curvature units
            return base[("tf_ramp", size)] + f * (1.0 - math.cos(math.radians(size))) * base[("tf_step", size)]
        return base[(name, size)]

    alpha = np.radians(np.linspace(0.0, 40.0, 401))
    cos_alpha = np.cos(alpha)

    def norm(b):
        return float(np.sum(weight * b * b))

    def response_max(bf, bs):
        c = weight * bf * bs
        return float(np.max(np.abs(np.polynomial.legendre.legval(cos_alpha, c))))

    truth_names = ["tf_ramp", "sw_ramp"] + [f"tf_mix_{f:g}" for f in mixes] + ["tf_step", "sw_step"]
    banks = {"tf_ramp bank": ["tf_ramp"], "sw_ramp bank": ["sw_ramp"], "tf_ramp + tf_step bank": ["tf_ramp", "tf_step"],
             "sw_step bank": ["sw_step"]}
    rows = []
    for size in truth_sizes:
        for tname in truth_names:
            bs = shape(tname, size)
            ns = norm(bs)
            rec = {"truth": tname, "theta_c_deg": size, "fitting_factor": {}, "best_filter": {}}
            for bname, types in banks.items():
                best, arg = 0.0, None
                for ftype in types:
                    for fsize in bank_sizes:
                        bf = base[(ftype, fsize)]
                        v = response_max(bf, bs) / math.sqrt(norm(bf) * ns)
                        if v > best:
                            best, arg = v, f"{ftype} {fsize:g} deg"
                rec["fitting_factor"][bname] = best
                rec["best_filter"][bname] = arg
            rows.append(rec)
        print(f"fitting factors done for {size:g} deg  ({time.time() - t0:.0f} s)", flush=True)

    print("\nfitting factor (fraction of the ideal signal-to-noise recovered); rows = true signal")
    names = list(banks)
    print(f"{'true signal':>14s} {'size':>5s}" + "".join(f"{n:>26s}" for n in names))
    for rec in rows:
        print(f"{rec['truth']:>14s} {rec['theta_c_deg']:5.0f}" + "".join(f"{rec['fitting_factor'][n]:26.3f}" for n in names))
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"lmax": lmax, "bank_sizes_deg": bank_sizes, "rows": rows}, indent=1))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
