"""How far can a network go? Train detectors on collision templates and measure them against the ceiling.

Purpose
-------
The November benchmarks (notebook entries (l) to (s)) compared one fine-tuned network with a matched-filter
bank whose search radius was chosen after the fact. This script repeats the comparison in a form that can
carry a claim:

* the signal prior (sizes, centre offsets, amplitudes, signs, shapes) is fixed in one place and shared by
  the training data, the test set and the classical references;
* the references are (1) the matched-filter bank searched over exactly the prior's support and (2) the
  Bayes test for that prior, the marginal likelihood ratio, which no detector can beat on these skies;
* networks are trained from an ImageNet or random initialisation (no dependence on the April checkpoints),
  several seeds per setting, signals re-drawn every epoch, last epoch used, no selection on the test set;
* a "shape family" setting trains one network on ramp, ramp+step mixtures at once, to measure what a
  learned detector gains and loses when the signal shape is not known in advance.

Assumptions
-----------
* Skies: Gaussian CMB (CAMB lensed TT, Planck 2018 parameters), ``synfast`` at Nside 256 with a 5 arcmin
  beam and the pixel window, plus 30 uK-arcmin white noise. Kelvin throughout. No mask in this script.
* Patches: ``hp.gnomview``-equivalent projections, 256 x 256 at 13 arcmin (``patch_geometry``).
* Signals: curvature ramp through the full CAMB transfer function (``collision_template_transfer``),
  optionally plus ``mix`` times a step whose height equals the ramp's Sachs-Wolfe central depth; beam and
  pixel window applied; monopole and dipole removed; scaled to unit observed peak. The amplitude A is the
  observed peak |Delta T / T|; injection is additive, Delta T = sign * A * T_CMB * profile(angle).
  The step part is band-limited: CAMB's default k range (k < 0.17 per Mpc) holds a step's multipoles only
  up to about 2370 sin(theta_c), that is 206 at 5 degrees, 411 at 10, 613 at 15, complete from 19 degrees
  (measured 2026-11-22: extending the range needs a k sampling CAMB does not provide at high k). The step
  shapes below 19 degrees are therefore smoother at the edge than the full calculation would give. Signals,
  filters and networks all use the same templates, so comparisons between methods are unaffected; the
  shapes themselves are a family to test shape freedom, not a prediction.
* Prior: size uniform over ``--sizes``; centre uniform in a disc of ``--max-offset-deg`` around the patch
  centre (5 degrees covers the 3.7 degree spacing of the Nside 16 tile scan); sign +-1; strength
  log-uniform in ideal matched-filter SNR between 1 and 12 in training (for the ramp that is an observed
  peak of about 1e-5 to 1.2e-4, as in November; using SNR keeps shapes of different detectability on one
  scale). Test sets use a fixed grid, either of amplitudes (``--amplitudes``) or of ideal SNR
  (``--target-snr``).
* References (computed in ``make-test`` from full-sky filter outputs; exact for these skies):
    glrt   max |SNR| over the bank and over sky pixels within the prior disc.
    bayes_known[j]     log of the likelihood ratio averaged over bank, pixels in the prior disc and sign,
                       for the known strength of test level j (one statistic per level).
    bayes_prior        the same averaged also over the training strength prior.
  With-signal filter outputs are the signal-free outputs plus the analytic filter response to the template.
* Network scores: the image-level logit and the largest mask logit (logits, not probabilities, so that
  float32 saturation cannot produce ties).
* Thresholds: each score's own quantile on the test set's signal-free patches (in-sample, as before).

Modes
-----
    python scripts/ml/ml_ceiling_study.py make-test --out-dir TEST [--mix 0] [--n-skies 1500]
    python scripts/ml/ml_ceiling_study.py train --out-dir RUN --test-dir TEST [--test-dir TEST2 ...]
           --seed 1 [--init imagenet|random] [--train-mixes 0] [--epochs 30]
    python scripts/ml/ml_ceiling_study.py summarize --runs-dir RUNS

``--smoke`` shrinks everything to a two-minute check of the whole path.
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
import patch_geometry as pg  # noqa: E402

NSIDE = 256
LMAX = 3 * NSIDE - 1
T_CMB_K = 2.7255
P = pg.PATCH_PIX
N_FINE = 18001
TRAIN_SNR_LO, TRAIN_SNR_HI = 1.0, 12.0  # ideal matched-filter SNR of training signals, log-uniform


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [centre - half, centre + half]


def null_threshold(null_scores, fpr):
    """Smallest threshold that at most a fraction ``fpr`` of the signal-free scores exceed."""
    s = np.sort(np.asarray(null_scores, dtype=np.float64))
    k = int(math.floor(fpr * s.size))
    return float(s[s.size - k - 1]) if k < s.size else float("-inf")


# --------------------------------------------------------------------------------------------------------
# sky model and templates
# --------------------------------------------------------------------------------------------------------
class World:
    def __init__(self, beam_fwhm_arcmin=5.0, noise_uk_arcmin=30.0):
        import camb
        import healpy as hp

        p = ct.PLANCK_2018
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=p["H0"], ombh2=p["ombh2"], omch2=p["omch2"], tau=p["tau"])
        pars.InitPower.set_params(As=p["As"], ns=p["ns"])
        pars.set_for_lmax(LMAX, lens_potential_accuracy=1)
        pars.Accuracy.AccuracyBoost = 1.5
        pars.Accuracy.lSampleBoost = 1.5
        res = camb.get_results(pars)
        self.cl = np.asarray(res.get_cmb_power_spectra(pars, CMB_unit="K", raw_cl=True)["lensed_scalar"][: LMAX + 1, 0], dtype=np.float64)
        self.window = ct.gaussian_beam(LMAX, beam_fwhm_arcmin) * hp.pixwin(NSIDE, lmax=LMAX)
        pix_area = hp.nside2pixarea(NSIDE)
        self.npix = hp.nside2npix(NSIDE)
        self.sigma_pix = noise_uk_arcmin * 1e-6 * math.radians(1.0 / 60.0) / math.sqrt(pix_area)
        self.total_power = self.cl * self.window**2 + self.sigma_pix**2 * pix_area
        self.fwhm_rad = math.radians(beam_fwhm_arcmin / 60.0)
        self.tr = ct.camb_transfer(lmax=LMAX)
        self.ells = np.arange(LMAX + 1)
        self.fine = np.radians(np.linspace(0.0, 180.0, N_FINE))
        self._base = {}
        self._tmpl = {}

    def _b(self, kind, size):
        key = (kind, float(size))
        if key not in self._base:
            r = self.tr.d_star * math.cos(math.radians(size))
            b = np.zeros(LMAX + 1)
            if kind == "ramp":
                b[self.tr.ells] = ct.b_ell(self.tr.ells, self.tr.k, self.tr.delta_t, r, a_ramp=1.0 / self.tr.d_star)
            else:
                b[self.tr.ells] = ct.b_ell(self.tr.ells, self.tr.k, self.tr.delta_t, r, a_step=1.0)
            self._base[key] = b
        return self._base[key]

    def template(self, size, mix=0.0):
        """Unit-peak template: t_l (Kelvin per unit amplitude), real-space profile, filter weights, norm."""
        key = (float(size), float(mix))
        if key in self._tmpl:
            return self._tmpl[key]
        b = self._b("ramp", size).copy()
        if mix != 0.0:
            b = b + float(mix) * (1.0 - math.cos(math.radians(size))) * self._b("step", size)
        coeff = (2 * self.ells + 1) / (4 * math.pi) * b * self.window
        coeff[:2] = 0.0
        prof = np.polynomial.legendre.legval(np.cos(self.fine), coeff)
        peak = int(np.argmax(np.abs(prof)))
        scale = 1.0 / prof[peak]
        prof = prof * scale
        t_l = b * self.window * scale * T_CMB_K
        t_l[:2] = 0.0
        norm = float(np.sum((2 * self.ells[2:] + 1) / (4 * math.pi) * t_l[2:] ** 2 / self.total_power[2:]))
        weights = np.where(self.ells >= 2, t_l / np.maximum(self.total_power, 1e-300), 0.0) / math.sqrt(norm)
        below = np.nonzero((np.abs(prof) < 0.1) & (np.arange(N_FINE) > peak))[0]
        out = {"size": float(size), "mix": float(mix), "t_l": t_l, "profile_k": prof * T_CMB_K, "norm": norm, "weights": weights,
               "mask_radius": float(self.fine[below[0]]) if below.size else math.radians(size),
               "peak_angle_deg": float(np.degrees(self.fine[peak]))}
        self._tmpl[key] = out
        return out

    def response(self, filt, sig):
        """Filter output (in sigma) per unit amplitude of ``sig``, as a function of angle from its centre."""
        c = (2 * self.ells + 1) / (4 * math.pi) * filt["weights"] * sig["t_l"]
        return np.polynomial.legendre.legval(np.cos(self.fine), c)

    def sky(self, rng):
        import healpy as hp

        np.random.seed(int(rng.integers(0, 2**32 - 1)))
        m = np.asarray(hp.synfast(self.cl, NSIDE, lmax=LMAX, new=True, pixwin=True, fwhm=self.fwhm_rad), dtype=np.float64)
        return m + rng.normal(0.0, self.sigma_pix, size=self.npix)


def random_positions(rng, n, min_sep_deg):
    pts, tries = [], 0
    while len(pts) < n and tries < 20000:
        tries += 1
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        if all(np.degrees(np.arccos(np.clip(v @ q, -1, 1))) >= min_sep_deg for q in pts):
            pts.append(v)
    return pts


def full_positions(rng, n, min_sep_deg=50.0):
    """``n`` directions at least ``min_sep_deg`` apart; redraws if a draw jams before reaching ``n``."""
    while True:
        pts = random_positions(rng, n, min_sep_deg)
        if len(pts) == n:
            return pts


def project(sky, lon, lat):
    import healpy as hp

    proj = hp.projector.GnomonicProj(rot=(lon, lat), xsize=P, ysize=P, reso=pg.RESO_ARCMIN)
    return np.asarray(proj.projmap(sky, lambda x, y, z: hp.vec2pix(NSIDE, x, y, z)), dtype=np.float64)


def background_patches(world, n_skies, per_sky, rng, label):
    """Signal-free patches for training or validation."""
    import healpy as hp

    out = np.zeros((n_skies * per_sky, P, P), dtype=np.float32)
    row, t0 = 0, time.time()
    for isky in range(n_skies):
        sky = world.sky(rng)
        for v in full_positions(rng, per_sky):
            lon, lat = hp.vec2ang(v, lonlat=True)
            out[row] = project(sky, float(lon[0]), float(lat[0])).astype(np.float32)
            row += 1
        if (isky + 1) % 250 == 0 or isky == n_skies - 1:
            print(f"  {label}: skies {isky + 1}/{n_skies}  {time.time() - t0:.0f} s", flush=True)
    return out[:row]


# --------------------------------------------------------------------------------------------------------
# make-test
# --------------------------------------------------------------------------------------------------------
def logmeanexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.mean(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis) if axis is not None else float(m.squeeze() + np.log(np.mean(np.exp(a - m))))


def cmd_make_test(args):
    import healpy as hp

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    world = World()
    bank_sizes = [float(x) for x in args.sizes.split(",")]
    test_sizes = [float(x) for x in args.test_sizes.split(",")]
    kind = "snr" if args.target_snr else "amplitude"
    levels = [float(x) for x in (args.target_snr or args.amplitudes).split(",")]
    mix = float(args.mix)
    if args.smoke:
        bank_sizes, test_sizes = [5.0, 15.0, 25.0], [5.0, 25.0]
        levels = [3.0, 8.0] if kind == "snr" else [3e-5, 1e-4]
        args.n_skies = 6
    cells = [(lv, s) for lv in levels for s in test_sizes]

    # banks: "ramp" = what the published searches use; "family" = every shape a family-trained network saw
    # (its prior, so its Bayes test is that network's ceiling); "matched" = the test set's exact shape
    # (an oracle that knows the shape; only built when the test shape is not a pure ramp)
    family_mixes = [float(x) for x in args.family_mixes.split(",")] if args.family_mixes else []
    banks = {"ramp": [world.template(s, 0.0) for s in bank_sizes]}
    if len(family_mixes) > 1 and not args.smoke:
        banks["family"] = [world.template(s, m) for m in family_mixes for s in bank_sizes]
    if mix != 0.0:
        banks["matched"] = [world.template(s, mix) for s in bank_sizes]
    sig_t = {s: world.template(s, mix) for s in test_sizes}
    resp = {bn: {s: np.stack([world.response(f, sig_t[s]) for f in bank]) for s in test_sizes} for bn, bank in banks.items()}
    norms = {bn: np.array([f["norm"] for f in bank]) for bn, bank in banks.items()}
    prior_snrs = np.exp(np.linspace(math.log(TRAIN_SNR_LO), math.log(TRAIN_SNR_HI), 24))
    for s in test_sizes:
        t = sig_t[s]
        print(f"test template {s:4.0f} deg mix {mix:g}: ideal SNR per 1e-5 of peak {1e-5 * math.sqrt(t['norm']):.3f}, peak at {t['peak_angle_deg']:.2f} deg", flush=True)

    n = args.n_skies * args.positions_per_sky
    patches = np.lib.format.open_memmap(out / "patches.npy", mode="w+", dtype=np.float32, shape=(n, P, P))
    labels = np.zeros(n, dtype=np.uint8)
    amp_true, size_true, sign_true, level_true = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    glrt = {bn: np.zeros(n) for bn in banks}
    bayes_known = {bn: np.zeros((n, len(levels))) for bn in banks}
    bayes_prior = {bn: np.zeros(n) for bn in banks}
    centre_snr = np.full(n, np.nan)
    null_std = []
    vec_all = np.array(hp.pix2vec(NSIDE, np.arange(world.npix)))
    c0 = (P - 1) / 2.0
    max_off_px = args.max_offset_deg * 60.0 / pg.RESO_ARCMIN
    fine = world.fine
    row, t0 = 0, time.time()
    for isky in range(args.n_skies):
        sky = world.sky(rng)
        alm = hp.map2alm(sky, lmax=LMAX, iter=1)
        snr = {bn: [hp.alm2map(hp.almxfl(alm, f["weights"]), NSIDE, lmax=LMAX) for f in bank] for bn, bank in banks.items()}
        if isky < 10:
            null_std.append([float(np.std(m)) for m in snr["ramp"]])
        for v in full_positions(rng, args.positions_per_sky):
            lon, lat = hp.vec2ang(v, lonlat=True)
            lon, lat = float(lon[0]), float(lat[0])
            patch = project(sky, lon, lat)
            if row == 0:
                ref = np.asarray(hp.gnomview(sky, rot=(lon, lat), reso=pg.RESO_ARCMIN, xsize=P, return_projected_map=True, no_plot=True), dtype=np.float64)
                if not np.array_equal(ref, patch):
                    raise SystemExit("projector output differs from hp.gnomview; refusing to continue")
                print("check: projector output identical to hp.gnomview", flush=True)
            foot = hp.query_disc(NSIDE, v, math.radians(args.max_offset_deg))
            positive = row % 2 == 1
            add = None
            if positive:
                lv, s = cells[(row // 2) % len(cells)]
                a = lv if kind == "amplitude" else lv / math.sqrt(sig_t[s]["norm"])
                sign = 1.0 if rng.random() < 0.5 else -1.0
                rr = max_off_px * math.sqrt(rng.random())
                phi = 2 * math.pi * rng.random()
                cx, cy = c0 + rr * math.cos(phi), c0 + rr * math.sin(phi)
                slon, slat = pg.pixel_to_lonlat(lon, lat, cx, cy)
                sv = hp.ang2vec(float(slon), float(slat), lonlat=True)
                patch = patch + sign * a * np.interp(pg.angular_distance_grid(cx, cy), fine, sig_t[s]["profile_k"])
                ang_foot = np.arccos(np.clip(sv @ vec_all[:, foot], -1.0, 1.0))
                add = {bn: sign * a * np.stack([np.interp(ang_foot, fine, r) for r in resp[bn][s]]) for bn in banks}
                labels[row], amp_true[row], size_true[row], sign_true[row], level_true[row] = 1, a, s, sign, lv
                if s in bank_sizes and mix == 0.0:
                    i = bank_sizes.index(s)
                    ic = int(hp.vec2pix(NSIDE, *sv))
                    centre_snr[row] = sign * (snr["ramp"][i][ic] + sign * a * float(resp["ramp"][s][i][0]))
            for bn in banks:
                z = np.stack([m[foot] for m in snr[bn]])                      # (bank, pixels in the prior disc)
                if add is not None:
                    z = z + add[bn]
                glrt[bn][row] = float(np.max(np.abs(z)))
                root = np.sqrt(norms[bn])[:, None]
                for j, lv_j in enumerate(levels):
                    if kind == "amplitude":   # known amplitude: each bank template has its own SNR
                        e = lv_j * root * z
                        half = 0.5 * lv_j**2 * norms[bn][:, None]
                    else:                     # known ideal SNR
                        e = lv_j * z
                        half = 0.5 * lv_j**2
                    bayes_known[bn][row, j] = logmeanexp(np.concatenate([(e - half).ravel(), (-e - half).ravel()]))
                per_s = [logmeanexp(np.concatenate([(sp * z - 0.5 * sp**2).ravel(), (-sp * z - 0.5 * sp**2).ravel()])) for sp in prior_snrs]
                bayes_prior[bn][row] = logmeanexp(np.asarray(per_s))
            patches[row] = patch.astype(np.float32)
            row += 1
        if (isky + 1) % 25 == 0 or isky == args.n_skies - 1:
            el = time.time() - t0
            print(f"sky {isky + 1}/{args.n_skies}  rows {row}  {el:.0f} s  ({el / (isky + 1):.1f} s per sky)", flush=True)
    patches.flush()
    if row != n:
        raise SystemExit(f"expected {n} rows, made {row} (random_positions ran out); rerun with fewer positions per sky")

    checks = {"null_snr_map_std_mean": [float(x) for x in np.mean(np.asarray(null_std), axis=0)], "centre_snr": []}
    print("check: std of signal-free SNR maps per bank size (should be 1):", [round(x, 3) for x in checks["null_snr_map_std_mean"]], flush=True)
    for lv, s in cells:
        sel = (labels == 1) & (level_true == lv) & (size_true == s) & np.isfinite(centre_snr)
        if sel.sum():
            checks["centre_snr"].append({"level": lv, "size_deg": s, "n": int(sel.sum()), "measured_mean": float(np.mean(centre_snr[sel])),
                                         "expected": float(np.mean(amp_true[sel])) * math.sqrt(sig_t[s]["norm"])})
    if checks["centre_snr"]:
        worst = max(abs(c["measured_mean"] - c["expected"]) * math.sqrt(c["n"]) for c in checks["centre_snr"])
        print(f"check: SNR at the signal centre against A*sqrt(norm): worst deviation {worst:.2f} standard errors", flush=True)

    save = {"labels": labels, "level": level_true, "amplitude": amp_true, "size_deg": size_true, "sign": sign_true}
    for bn in banks:
        save[f"ref__glrt_{bn}_bank"] = glrt[bn]
        save[f"ref__bayes_prior_{bn}_bank"] = bayes_prior[bn]
        save[f"refA__bayes_known_{bn}_bank"] = bayes_known[bn]
    np.savez_compressed(out / "meta.npz", **save)
    meta = {"mix": mix, "bank_sizes_deg": bank_sizes, "test_sizes_deg": test_sizes, "levels": levels, "level_kind": kind,
            "banks": {bn: len(b) for bn, b in banks.items()}, "max_offset_deg": args.max_offset_deg,
            "seed": args.seed, "n_rows": int(n), "checks": checks, "ideal_snr_per_1e5": {str(s): 1e-5 * math.sqrt(sig_t[s]["norm"]) for s in test_sizes}}
    (out / "test_config.json").write_text(json.dumps(meta, indent=1))
    table = detection_table(save, levels, test_sizes, [0.05, 0.01])
    (out / "reference_report.json").write_text(json.dumps(table, indent=1))
    print_table(table, levels, 0.05)
    print("\nwrote", out, flush=True)


def detection_table(scores, amps, sizes, fprs):
    """Detection probability per test level (all sizes) and per (level, size) for every score in ``scores``."""
    labels, amp, size = scores["labels"], scores["level"], scores["size_deg"]
    neg = labels == 0
    out = {}
    for key, s in scores.items():
        if key.startswith("ref__") or key.startswith("net__"):
            columns = {None: np.asarray(s, dtype=np.float64)}
        elif key.startswith("refA__"):
            columns = {j: np.asarray(s, dtype=np.float64)[:, j] for j in range(len(amps))}
        else:
            continue
        name = key.split("__", 1)[1]
        rec = {}
        for fpr in fprs:
            by_amp, cells = [], []
            for j, a in enumerate(amps):
                col = columns[None] if None in columns else columns[j]
                thr = null_threshold(col[neg], fpr)
                sel = (labels == 1) & np.isclose(amp, a, rtol=1e-9)
                k, n = int(np.sum(col[sel] > thr)), int(np.sum(sel))
                by_amp.append({"amplitude": a, "k": k, "n": n, "p": k / max(n, 1), "ci95": wilson(k, n)})
                for sz in sizes:
                    sel2 = sel & (size == sz)
                    k2, n2 = int(np.sum(col[sel2] > thr)), int(np.sum(sel2))
                    cells.append({"amplitude": a, "size_deg": sz, "k": k2, "n": n2, "p": k2 / max(n2, 1)})
            rec[f"fpr_{fpr:g}"] = {"by_amplitude": by_amp, "cells": cells}
        out[name] = rec
    return out


def print_table(table, amps, fpr):
    print(f"\ndetection probability at {fpr:.0%} false positives, all sizes pooled")
    print(f"{'score':40s}" + "".join(f"{a:>9.3g}" for a in amps))
    for name, rec in table.items():
        print(f"{name:40s}" + "".join(f"{r['p']:9.3f}" for r in rec[f"fpr_{fpr:g}"]["by_amplitude"]))


# --------------------------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------------------------
class Injector:
    """Adds freshly drawn signals to a batch of background patches on the GPU (geometry of ``patch_geometry``)."""

    def __init__(self, world, sizes, mixes, max_offset_deg, device):
        import torch

        self.torch = torch
        tmpl = [world.template(s, m) for m in mixes for s in sizes]
        self.profiles = torch.tensor(np.stack([t["profile_k"] for t in tmpl]), dtype=torch.float32, device=device)
        self.mask_radius = torch.tensor([t["mask_radius"] for t in tmpl], dtype=torch.float32, device=device)
        self.root_norm = torch.tensor([math.sqrt(t["norm"]) for t in tmpl], dtype=torch.float32, device=device)
        c0 = (P - 1) / 2.0
        self.c0 = c0
        self.dx = math.radians(pg.RESO_ARCMIN / 60.0)
        iy, ix = torch.meshgrid(torch.arange(P, device=device, dtype=torch.float32), torch.arange(P, device=device, dtype=torch.float32), indexing="ij")
        v = torch.stack([-(ix - c0) * self.dx, (iy - c0) * self.dx, torch.ones_like(ix)], dim=0)
        self.v = (v / v.norm(dim=0, keepdim=True)).reshape(3, -1)             # (3, P*P)
        self.max_off_px = max_offset_deg * 60.0 / pg.RESO_ARCMIN
        self.step = math.pi / (N_FINE - 1)
        self.device = device

    def __call__(self, x, gen, snr_lo=TRAIN_SNR_LO, snr_hi=TRAIN_SNR_HI, p_signal=0.5):
        torch = self.torch
        b = x.shape[0]
        u = torch.rand((b, 6), generator=gen, device=self.device)
        is_pos = u[:, 0] < p_signal
        shape = torch.clamp((u[:, 1] * self.profiles.shape[0]).long(), max=self.profiles.shape[0] - 1)
        snr = torch.exp(math.log(snr_lo) + u[:, 2] * (math.log(snr_hi) - math.log(snr_lo)))
        amp = snr / self.root_norm[shape]
        sign = torch.where(u[:, 3] < 0.5, 1.0, -1.0)
        rr = self.max_off_px * torch.sqrt(u[:, 4])
        phi = 2 * math.pi * u[:, 5]
        cx, cy = self.c0 + rr * torch.cos(phi), self.c0 + rr * torch.sin(phi)
        cv = torch.stack([-(cx - self.c0) * self.dx, (cy - self.c0) * self.dx, torch.ones_like(cx)], dim=1)
        cv = cv / cv.norm(dim=1, keepdim=True)                                 # (b, 3)
        ang = torch.arccos(torch.clamp(cv @ self.v, -1.0, 1.0))                # (b, P*P)
        pos = ang / self.step
        i0 = torch.clamp(pos.floor().long(), max=N_FINE - 2)
        frac = pos - i0
        prof = self.profiles[shape]                                            # (b, N_FINE)
        val = torch.gather(prof, 1, i0) * (1 - frac) + torch.gather(prof, 1, i0 + 1) * frac
        scale = (is_pos.float() * sign * amp)[:, None]
        x = x + (scale * val).reshape(b, P, P)
        mask = ((ang <= self.mask_radius[shape][:, None]) & is_pos[:, None]).float().reshape(b, P, P)
        return x, mask, is_pos.float(), snr


def build_model(init, device):
    import segmentation_models_pytorch as smp

    weights = "imagenet" if init == "imagenet" else None
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=weights, in_channels=1, classes=1,
                     aux_params={"classes": 1, "dropout": 0.2})
    return model.to(device)


def score_patches(model, patches, std, device, batch=256):
    import torch

    model.eval()
    seg = np.zeros(len(patches), dtype=np.float64)
    img = np.zeros(len(patches), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(patches), batch):
            x = torch.from_numpy(np.ascontiguousarray(patches[i:i + batch])).to(device)[:, None] / std
            mask_logits, image_logits = model(x)
            seg[i:i + batch] = mask_logits.flatten(1).max(dim=1).values.double().cpu().numpy()
            img[i:i + batch] = image_logits.reshape(-1).double().cpu().numpy()
    return seg, img


def cmd_train(args):
    import torch
    import torch.nn.functional as F

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    sizes = [float(x) for x in args.sizes.split(",")]
    mixes = [float(x) for x in args.train_mixes.split(",")]
    if args.smoke:
        sizes, args.train_skies, args.val_skies, args.epochs = [5.0, 15.0, 25.0], 12, 4, 1
    world = World()
    inject = Injector(world, sizes, mixes, args.max_offset_deg, device)
    print(f"training shapes: {len(sizes)} sizes x mixes {mixes}; init {args.init}; seed {args.seed}", flush=True)

    # data seeds are separate from the model seed so that every seed of a setting sees different skies
    xtr = background_patches(world, args.train_skies, args.positions_per_sky, np.random.default_rng(1_000_000 + args.seed), "train")
    xva = background_patches(world, args.val_skies, args.positions_per_sky, np.random.default_rng(2_000_000 + args.seed), "validation")
    std = float(np.std(xtr[: min(len(xtr), 2000)]))
    print(f"input scale {std:.3e} K; {len(xtr)} training and {len(xva)} validation backgrounds", flush=True)

    # fixed validation signals
    vgen = torch.Generator(device=device)
    vgen.manual_seed(777 + args.seed)
    xv, yv, av = [], [], []
    for i in range(0, len(xva), 256):
        xb, _, yb, ab = inject(torch.from_numpy(xva[i:i + 256]).to(device), vgen)
        xv.append(xb.cpu().numpy())
        yv.append(yb.cpu().numpy())
        av.append(ab.cpu().numpy())
    xv, yv, av = np.concatenate(xv), np.concatenate(yv), np.concatenate(av)
    mid = (yv == 1) & (av >= 2.0) & (av <= 5.0)   # ideal SNR 2 to 5: neither at chance nor saturated

    model = build_model(args.init, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(xtr) / args.batch_size)
    total, warm = args.epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(xtr))
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(order), args.batch_size):
            idx = np.sort(order[i:i + args.batch_size])
            x, m, y, _ = inject(torch.from_numpy(xtr[idx]).to(device), gen)
            x, m = (x / std)[:, None], m[:, None]
            k = int(rng.integers(0, 4))
            if k:
                x, m = torch.rot90(x, k, dims=(2, 3)), torch.rot90(m, k, dims=(2, 3))
            if rng.random() < 0.5:
                x, m = torch.flip(x, dims=(3,)), torch.flip(m, dims=(3,))
            mask_logits, image_logits = model(x)
            prob = torch.sigmoid(mask_logits)
            inter = (prob * m).flatten(1).sum(1)
            dice = 1.0 - (2.0 * inter + 1.0) / (prob.flatten(1).sum(1) + m.flatten(1).sum(1) + 1.0)
            loss = F.binary_cross_entropy_with_logits(mask_logits, m) + dice.mean() \
                + args.image_loss_weight * F.binary_cross_entropy_with_logits(image_logits.reshape(-1), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        segv, imgv = score_patches(model, xv, std, device)
        rec = {"epoch": epoch, "train_loss": tot / nb, "seconds": time.time() - t0}
        for name, s in (("mask_max", segv), ("image_head", imgv)):
            thr = null_threshold(s[yv == 0], 0.05)
            rec[f"val_detect_mid_{name}"] = float(np.mean(s[mid] > thr)) if mid.any() else float("nan")
        history.append(rec)
        print(f"epoch {epoch:2d}  loss {rec['train_loss']:.4f}  validation detection (ideal SNR 2-5) at 5%: mask max {rec['val_detect_mid_mask_max']:.3f}, "
              f"image head {rec['val_detect_mid_image_head']:.3f}  ({rec['seconds']:.0f} s)", flush=True)

    torch.save({"model_state_dict": model.state_dict(), "input_std": std, "args": vars(args)}, out / "checkpoint_last.pt")
    report = {"args": vars(args), "history": history, "input_std": std, "tests": {}}
    for test_dir in args.test_dir:
        td = Path(test_dir)
        cfg = json.loads((td / "test_config.json").read_text())
        meta = dict(np.load(td / "meta.npz"))
        patches = np.load(td / "patches.npy", mmap_mode="r")
        seg, img = score_patches(model, patches, std, device)
        scores = {"labels": meta["labels"], "level": meta["level"], "size_deg": meta["size_deg"],
                  "net__mask_max": seg, "net__image_head": img}
        scores.update({k: v for k, v in meta.items() if k.startswith("ref")})
        table = detection_table(scores, cfg["levels"], cfg["test_sizes_deg"], [0.05, 0.01])
        report["tests"][td.name] = {"mix": cfg["mix"], "level_kind": cfg["level_kind"], "table": table}
        np.savez_compressed(out / f"scores__{td.name}.npz", mask_max=seg, image_head=img)
        print(f"\ntest set {td.name} (mix {cfg['mix']:g}, levels are {cfg['level_kind']})")
        print_table(table, cfg["levels"], 0.05)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("\nwrote", out, flush=True)


# --------------------------------------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------------------------------------
def cmd_summarize(args):
    runs = sorted(p for p in Path(args.runs_dir).glob("*/report.json"))
    groups = {}
    for p in runs:
        rep = json.loads(p.read_text())
        a = rep["args"]
        groups.setdefault((a["init"], a["train_mixes"]), []).append(rep)
    fpr = f"fpr_{args.fpr:g}"
    summary = {}
    for (init, mixes), reps in sorted(groups.items()):
        tests = sorted(set(t for r in reps for t in r["tests"]))
        for t in tests:
            have = [r for r in reps if t in r["tests"]]
            table0 = have[0]["tests"][t]["table"]
            amps = [x["amplitude"] for x in table0["mask_max"][fpr]["by_amplitude"]]
            print(f"\ninit {init}, trained on mixes {mixes}, test set {t} (mix {have[0]['tests'][t]['mix']:g}), {len(have)} seeds, {args.fpr:.0%} false positives")
            print(f"{'score':40s}" + "".join(f"{a:>15.3g}" for a in amps))
            block = {}
            for name in table0:
                if name in ("mask_max", "image_head"):
                    vals = np.array([[x["p"] for x in r["tests"][t]["table"][name][fpr]["by_amplitude"]] for r in have])
                    cells = ["%.3f +- %.3f" % (vals[:, j].mean(), vals[:, j].std(ddof=1) if len(have) > 1 else float("nan")) for j in range(len(amps))]
                    block[name] = {"mean": vals.mean(0).tolist(), "sd": (vals.std(0, ddof=1) if len(have) > 1 else np.full(len(amps), np.nan)).tolist(), "per_seed": vals.tolist()}
                else:
                    v = [x["p"] for x in table0[name][fpr]["by_amplitude"]]
                    cells = ["%.3f" % x for x in v]
                    block[name] = {"value": v}
                print(f"{name:40s}" + "".join(f"{c:>15s}" for c in cells))
            # by size at the third test level (3e-5 on the amplitude grid, ideal SNR 3 on the SNR grid)
            j3 = min(2, len(amps) - 1)
            sizes = sorted(set(c["size_deg"] for c in table0["mask_max"][fpr]["cells"]))
            print(f"by size at level {amps[j3]:.3g}:   " + "".join(f"{s:>9.0f}" for s in sizes))
            for name in table0:
                rows = []
                for r in (have if name in ("mask_max", "image_head") else have[:1]):
                    cells = r["tests"][t]["table"][name][fpr]["cells"]
                    rows.append([next(c["p"] for c in cells if c["size_deg"] == s and abs(c["amplitude"] - amps[j3]) < 1e-12) for s in sizes])
                m = np.mean(np.array(rows), axis=0)
                block[name]["by_size"] = {"sizes_deg": sizes, "amplitude": amps[j3], "mean": m.tolist()}
                print(f"{name:27s}" + "".join(f"{x:9.3f}" for x in m))
            summary[f"{init}|{mixes}|{t}"] = block
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=1))
        print("\nwrote", args.out_json)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sizes_default = ",".join(str(x) for x in range(5, 26))

    t = sub.add_parser("make-test")
    t.add_argument("--out-dir", required=True)
    t.add_argument("--mix", type=float, default=0.0)
    t.add_argument("--family-mixes", default="0,0.25,0.5,1", help="step fractions of the family bank; empty = no family bank")
    t.add_argument("--n-skies", type=int, default=1500)
    t.add_argument("--positions-per-sky", type=int, default=8)
    t.add_argument("--sizes", default=sizes_default, help="bank sizes (the prior's sizes)")
    t.add_argument("--test-sizes", default="5,10,15,20,25")
    t.add_argument("--amplitudes", default="1e-5,2e-5,3e-5,5e-5,7e-5,1e-4")
    t.add_argument("--target-snr", default=None, help="comma list of ideal SNR levels; replaces --amplitudes")
    t.add_argument("--max-offset-deg", type=float, default=5.0)
    t.add_argument("--seed", type=int, default=20261201)
    t.add_argument("--smoke", action="store_true")
    t.set_defaults(func=cmd_make_test)

    r = sub.add_parser("train")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--test-dir", action="append", default=[])
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--init", choices=["imagenet", "random"], default="imagenet")
    r.add_argument("--train-mixes", default="0", help="comma list of step fractions the network is trained on")
    r.add_argument("--sizes", default=sizes_default)
    r.add_argument("--max-offset-deg", type=float, default=5.0)
    r.add_argument("--train-skies", type=int, default=4000)
    r.add_argument("--val-skies", type=int, default=300)
    r.add_argument("--positions-per-sky", type=int, default=8)
    r.add_argument("--epochs", type=int, default=30)
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--lr", type=float, default=3e-4)
    r.add_argument("--image-loss-weight", type=float, default=1.0)
    r.add_argument("--device", default="cuda")
    r.add_argument("--smoke", action="store_true")
    r.set_defaults(func=cmd_train)

    s = sub.add_parser("summarize")
    s.add_argument("--runs-dir", required=True)
    s.add_argument("--fpr", type=float, default=0.05)
    s.add_argument("--out-json", default=None)
    s.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
