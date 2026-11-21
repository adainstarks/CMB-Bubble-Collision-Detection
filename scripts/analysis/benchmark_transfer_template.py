"""Benchmark on physically motivated signals: matched-filter bank and saved U-Nets on the same patches.

Assumptions
-----------
* Signals are the ramp template of ``collision_template_transfer`` (primordial curvature ramp through CAMB
  transfer functions), with the 5 arcmin beam and the Nside 256 pixel window applied, monopole and dipole
  excluded. The amplitude A is the observed peak |Delta T / T| after beam and pixel window; the sign is
  random. Injection is additive, Delta T = A * T_CMB * profile(angle).
* Backgrounds are Gaussian CMB skies from the project's observing model (CAMB lensed TT, ``synfast`` with
  beam and pixel window) plus white noise of 30 uK-arcmin, all in Kelvin.
* Patches are ``hp.gnomview`` projections (256 x 256, 13 arcmin), the same as every April data set. The
  signal is added on the patch using the true pixel directions (``patch_geometry``), which is equivalent
  to adding it on the sphere before projecting, up to nearest-pixel sampling.
* Signal centres lie within 5 degrees of the patch centre.
* Matched filter: for each bank size, SNR(n) = [sum_lm (t_l / D_l) a_lm Y_lm(n)] / sqrt(sum_l (2l+1)/(4 pi)
  t_l^2 / D_l), t_l = template coefficient with beam and pixel window, D_l = total map power. The patch
  statistic is the maximum of |SNR| over the bank and over all sky pixels within 27 degrees of the patch
  centre. Because filtering is linear, the with-signal SNR map is the null SNR map plus the analytic
  filter response to the template.
* Networks are scored exactly as in ``phase3_sensitivity_curve.score_ml_model`` (maximum of the mask
  probability over the patch).
* Every method gets its own threshold at the target false-positive rate on the signal-free patches.

Usage (from the project root):
    python scripts/analysis/benchmark_transfer_template.py --out-dir OUT \
        --model name:run_dir:checkpoint [--model ...] [--n-skies 450] [--positions-per-sky 8]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

import h5py
import healpy as hp
import numpy as np

import collision_template_transfer as ct
import patch_geometry as pg
from phase2_observing_model import camb_tt_cls
from phase3_thresholds import conformal_threshold_from_scores
from phase_config import T_CMB

NSIDE = 256
LMAX = 3 * NSIDE - 1
T_CMB_K = float(T_CMB.to_value("K"))


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [centre - half, centre + half]


def build_templates(sizes_deg, window, total_power):
    """Per size: coefficient t_l in Kelvin per unit peak amplitude, filter weights, real-space profile."""
    tr = ct.camb_transfer(lmax=LMAX)
    ells = np.arange(LMAX + 1)
    fine = np.radians(np.linspace(0.0, 180.0, 18001))
    out = {}
    for size in sizes_deg:
        tc = math.radians(size)
        b = np.zeros(LMAX + 1)
        b[tr.ells] = ct.b_ell(tr.ells, tr.k, tr.delta_t, tr.d_star * math.cos(tc), a_ramp=1.0 / tr.d_star)
        coeff = (2 * ells + 1) / (4 * math.pi) * b * window
        coeff[:2] = 0.0
        prof = np.polynomial.legendre.legval(np.cos(fine), coeff)
        peak_index = int(np.argmax(np.abs(prof)))
        scale = 1.0 / prof[peak_index]                      # unit positive peak
        t_l = b * window * scale * T_CMB_K                 # Kelvin per unit amplitude
        t_l[:2] = 0.0
        norm = float(np.sum((2 * ells[2:] + 1) / (4 * math.pi) * t_l[2:] ** 2 / total_power[2:]))
        out[size] = {"t_l": t_l, "weights": np.where(ells >= 2, t_l / np.maximum(total_power, 1e-300), 0.0) / math.sqrt(norm),
                     "norm": norm, "profile_k": prof * scale * T_CMB_K, "peak_angle_deg": float(np.degrees(fine[peak_index])),
                     "ideal_snr_per_1e5": 1e-5 * math.sqrt(norm)}
    # response of filter f to a unit-amplitude template s, as a function of angle from the signal centre
    resp = {}
    for f in sizes_deg:
        for s in sizes_deg:
            c = (2 * ells + 1) / (4 * math.pi) * out[f]["weights"] * out[s]["t_l"]
            resp[(f, s)] = np.polynomial.legendre.legval(np.cos(fine), c)
    return out, resp, fine


def random_positions(rng, n, min_sep_deg):
    pts = []
    tries = 0
    while len(pts) < n and tries < 20000:
        tries += 1
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        if all(np.degrees(np.arccos(np.clip(v @ p, -1, 1))) >= min_sep_deg for p in pts):
            pts.append(v)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", action="append", default=[], help="name:run_dir:checkpoint")
    ap.add_argument("--n-skies", type=int, default=450)
    ap.add_argument("--positions-per-sky", type=int, default=8)
    ap.add_argument("--sizes-deg", default="5,10,15,20,25")
    ap.add_argument("--amplitudes", default="1e-5,2e-5,3e-5,5e-5,7e-5,1e-4")
    ap.add_argument("--fpr-target", type=float, default=0.05)
    ap.add_argument("--max-offset-deg", type=float, default=5.0)
    ap.add_argument("--footprint-deg", default="27", help="search radius of the filter around the patch centre; comma list for several")
    ap.add_argument("--noise-uk-arcmin", type=float, default=30.0)
    ap.add_argument("--beam-fwhm-arcmin", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-ml", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sizes = [float(x) for x in args.sizes_deg.split(",")]
    amps = [float(x) for x in args.amplitudes.split(",")]
    cells = [(a, s) for a in amps for s in sizes]

    # ---- covariance model, identical to how the skies are made ---------------------------------------
    cl_tt, _ = camb_tt_cls(lmax=LMAX)
    window = ct.gaussian_beam(LMAX, args.beam_fwhm_arcmin) * hp.pixwin(NSIDE, lmax=LMAX)
    pix_area = hp.nside2pixarea(NSIDE)
    sigma_pix = args.noise_uk_arcmin * 1e-6 * math.radians(1.0 / 60.0) / math.sqrt(pix_area)
    noise_cl = sigma_pix**2 * pix_area
    total_power = np.asarray(cl_tt[: LMAX + 1], dtype=np.float64) * window**2 + noise_cl
    templates, resp, fine = build_templates(sizes, window, total_power)
    for s in sizes:
        print(f"template {s:4.0f} deg: ideal SNR per 1e-5 of peak {templates[s]['ideal_snr_per_1e5']:.3f}, peak at {templates[s]['peak_angle_deg']:.2f} deg", flush=True)

    n_pos_total = args.n_skies * args.positions_per_sky
    patches = np.zeros((n_pos_total, pg.PATCH_PIX, pg.PATCH_PIX), dtype=np.float32)
    masks = np.zeros((n_pos_total, pg.PATCH_PIX, pg.PATCH_PIX), dtype=np.uint8)
    labels = np.zeros(n_pos_total, dtype=np.uint8)
    amp_true = np.zeros(n_pos_total)
    size_true = np.zeros(n_pos_total)
    sign_true = np.zeros(n_pos_total)
    footprints = [float(x) for x in str(args.footprint_deg).split(",")]
    mf_stat = {r: np.zeros(n_pos_total) for r in footprints}
    mf_best_size = np.zeros(n_pos_total)
    centre_snr = np.full(n_pos_total, np.nan)
    null_std = []
    vec_all = np.array(hp.pix2vec(NSIDE, np.arange(hp.nside2npix(NSIDE))))
    npix = hp.nside2npix(NSIDE)
    c0 = (pg.PATCH_PIX - 1) / 2.0
    max_off_px = args.max_offset_deg * 60.0 / pg.RESO_ARCMIN
    row = 0
    fwhm_rad = math.radians(args.beam_fwhm_arcmin / 60.0)

    def vec2pix(x, y, z):
        return hp.vec2pix(NSIDE, x, y, z)

    t0 = time.time()
    for isky in range(args.n_skies):
        # same call as phase2_observing_model.synthesize_cmb_maps, without re-running CAMB for every sky
        np.random.seed(int(rng.integers(0, 2**32 - 1)))
        sky = np.asarray(hp.synfast(cl_tt, NSIDE, lmax=LMAX, new=True, pixwin=True, fwhm=fwhm_rad), dtype=np.float64)
        sky = sky + rng.normal(0.0, sigma_pix, size=npix)
        alm = hp.map2alm(sky, lmax=LMAX, iter=1)
        snr_maps = {s: hp.alm2map(hp.almxfl(alm, templates[s]["weights"]), NSIDE, lmax=LMAX) for s in sizes}
        if isky < 20:
            null_std.append([float(np.std(snr_maps[s])) for s in sizes])
        for v in random_positions(rng, args.positions_per_sky, 50.0):
            lon, lat = hp.vec2ang(v, lonlat=True)
            lon, lat = float(lon[0]), float(lat[0])
            proj = hp.projector.GnomonicProj(rot=(lon, lat), xsize=pg.PATCH_PIX, ysize=pg.PATCH_PIX, reso=pg.RESO_ARCMIN)
            patch = np.asarray(proj.projmap(sky, vec2pix), dtype=np.float64)
            if row == 0:  # guard: the projector must return exactly what hp.gnomview returns
                ref = np.asarray(hp.gnomview(sky, rot=(lon, lat), reso=pg.RESO_ARCMIN, xsize=pg.PATCH_PIX,
                                             return_projected_map=True, no_plot=True), dtype=np.float64)
                if not np.array_equal(ref, patch):
                    raise SystemExit("GnomonicProj.projmap differs from hp.gnomview; refusing to continue")
                print("check: projector output identical to hp.gnomview", flush=True)
            foots = {r: hp.query_disc(NSIDE, v, math.radians(r)) for r in footprints}
            positive = (row % 2 == 1)
            if positive:
                a, s = cells[(row // 2) % len(cells)]
                sign = 1.0 if rng.random() < 0.5 else -1.0
                rr = max_off_px * math.sqrt(rng.random())
                phi = 2 * math.pi * rng.random()
                cx, cy = c0 + rr * math.cos(phi), c0 + rr * math.sin(phi)
                slon, slat = pg.pixel_to_lonlat(lon, lat, cx, cy)
                sv = hp.ang2vec(float(slon), float(slat), lonlat=True)
                ang_patch = pg.angular_distance_grid(cx, cy)
                patch = patch + sign * a * np.interp(ang_patch, fine, templates[s]["profile_k"])
                masks[row] = (ang_patch <= math.radians(s)).astype(np.uint8)
                for r, foot in foots.items():
                    ang_foot = np.arccos(np.clip(sv @ vec_all[:, foot], -1.0, 1.0))
                    best, best_size = 0.0, sizes[0]
                    for f in sizes:
                        val = float(np.max(np.abs(snr_maps[f][foot] + sign * a * np.interp(ang_foot, fine, resp[(f, s)]))))
                        if val > best:
                            best, best_size = val, f
                    mf_stat[r][row] = best
                ic = int(hp.vec2pix(NSIDE, *sv))
                centre_snr[row] = sign * (snr_maps[s][ic] + sign * a * float(np.interp(0.0, fine, resp[(s, s)])))
                labels[row], amp_true[row], size_true[row], sign_true[row] = 1, a, s, sign
            else:
                for r, foot in foots.items():
                    best, best_size = 0.0, sizes[0]
                    for f in sizes:
                        val = float(np.max(np.abs(snr_maps[f][foot])))
                        if val > best:
                            best, best_size = val, f
                    mf_stat[r][row] = best
            mf_best_size[row] = best_size
            patches[row] = patch.astype(np.float32)
            row += 1
        if (isky + 1) % 25 == 0 or isky == args.n_skies - 1:
            el = time.time() - t0
            print(f"sky {isky + 1}/{args.n_skies}  rows {row}  {el:.0f} s  ({el / (isky + 1):.1f} s per sky)", flush=True)
    n = row
    patches, masks, labels = patches[:n], masks[:n], labels[:n]
    amp_true, size_true, sign_true, mf_best_size, centre_snr = (x[:n] for x in (amp_true, size_true, sign_true, mf_best_size, centre_snr))
    mf_stat = {r: v[:n] for r, v in mf_stat.items()}

    # ---- self-checks ---------------------------------------------------------------------------------
    checks = {"null_snr_map_std_mean": [float(x) for x in np.mean(np.asarray(null_std), axis=0)], "centre_snr": []}
    print("check: std of null SNR maps per bank size (should be 1):", [round(x, 3) for x in checks["null_snr_map_std_mean"]], flush=True)
    for a, s in cells:
        sel = (labels == 1) & (amp_true == a) & (size_true == s)
        if sel.sum():
            expect = a * math.sqrt(templates[s]["norm"])
            checks["centre_snr"].append({"amplitude": a, "size_deg": s, "n": int(sel.sum()), "measured_mean": float(np.mean(centre_snr[sel])), "expected": expect})
    worst = max(abs(c["measured_mean"] - c["expected"]) / math.sqrt(1.0 / max(c["n"], 1)) for c in checks["centre_snr"])
    print(f"check: matched-size SNR at the signal centre, measured mean against A*sqrt(norm): worst deviation {worst:.2f} standard errors", flush=True)

    scores = {f"matched_filter_bank_{r:g}deg": v for r, v in mf_stat.items()}
    h5_path = out / "benchmark_data.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("patches", data=patches, compression="gzip", compression_opts=1, shuffle=True)
        h5.create_dataset("masks", data=masks, compression="gzip", compression_opts=1, shuffle=True)
        h5.create_dataset("labels", data=labels)
        g = h5.create_group("truth")
        g.create_dataset("amplitude", data=amp_true)
        g.create_dataset("theta_crit_deg", data=size_true)
        g.create_dataset("sign", data=sign_true)
        h5.attrs["description"] = "transfer-function ramp templates on Gaussian CMB skies; amplitude = observed peak Delta T / T"

    if not args.skip_ml and args.model:
        import argparse as _ap
        import phase3_train_unet as p3
        from phase3_sensitivity_curve import parse_model_spec, score_ml_model

        device = p3.resolve_device(args.device)
        for text in args.model:
            spec = parse_model_spec(text)
            sargs = _ap.Namespace(batch_size=128, num_workers=0, cache_ml_data=True, score_mode="segmentation_max")
            s, _meta = score_ml_model(spec, h5_path, sargs, device, data_parallel_device_ids=None)
            scores[spec.name] = np.asarray(s, dtype=np.float64)

    # ---- detection probabilities -----------------------------------------------------------------------
    report = {"n_rows": int(n), "n_positive": int(labels.sum()), "n_negative": int((labels == 0).sum()), "fpr_target": args.fpr_target,
              "sizes_deg": sizes, "amplitudes": amps, "checks": checks, "methods": {},
              "ideal_snr_per_1e5": {str(s): templates[s]["ideal_snr_per_1e5"] for s in sizes}}
    for name, s in scores.items():
        rec = conformal_threshold_from_scores(s, labels, args.fpr_target)
        thr = float(rec["threshold"])
        cellrows = []
        for a, size in cells:
            sel = (labels == 1) & (amp_true == a) & (size_true == size)
            k = int(np.sum(s[sel] > thr))
            cellrows.append({"amplitude": a, "size_deg": size, "k": k, "n": int(sel.sum()), "p": k / max(int(sel.sum()), 1), "ci95": wilson(k, int(sel.sum()))})
        by_amp = []
        for a in amps:
            sel = (labels == 1) & (amp_true == a)
            k = int(np.sum(s[sel] > thr))
            by_amp.append({"amplitude": a, "k": k, "n": int(sel.sum()), "p": k / max(int(sel.sum()), 1), "ci95": wilson(k, int(sel.sum()))})
        report["methods"][name] = {"threshold": thr, "negative_fpr": float(np.mean(s[labels == 0] > thr)), "cells": cellrows, "by_amplitude": by_amp}

    print(f"\ndetection probability at {args.fpr_target:.0%} false positives; amplitude = observed peak Delta T / T; {n} patches", flush=True)
    for name, m in report["methods"].items():
        print(f"\n{name}  (threshold {m['threshold']:.4g})")
        print(f"{'amplitude':>10s}" + "".join(f"{('%.0f deg' % s):>9s}" for s in sizes) + f"{'all sizes':>20s}")
        for a, ba in zip(amps, m["by_amplitude"]):
            line = "".join(f"{next(c['p'] for c in m['cells'] if c['amplitude'] == a and c['size_deg'] == s):9.2f}" for s in sizes)
            print(f"{a:10.0e}{line}   {ba['p']:.3f} [{ba['ci95'][0]:.3f},{ba['ci95'][1]:.3f}]")
    np.savez_compressed(out / "benchmark_scores.npz", labels=labels, amplitude=amp_true, size_deg=size_true, sign=sign_true,
                        mf_best_size=mf_best_size, **{f"score__{k}": v for k, v in scores.items()})
    (out / "benchmark_report.json").write_text(json.dumps(report, indent=1))
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
