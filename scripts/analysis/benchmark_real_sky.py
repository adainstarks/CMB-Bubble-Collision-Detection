"""Real-sky test: matched-filter bank and U-Nets on the Planck PR3 maps, with injected templates.

Assumptions
-----------
* Maps: Planck 2018 (PR3) component-separated temperature maps (field 0), Nside 2048, ``ud_grade`` to 256,
  Kelvin. Mask: the PR3 common intensity mask, ``ud_grade`` to 256; a pixel is kept where the value is at
  least 0.9 (the project's convention).
* Preparation, identical for the real maps and for every simulated sky: monopole and dipole fitted on the
  unmasked pixels and removed; masked pixels filled with a smooth local mean (10 degree Gaussian smoothing
  of the masked map, divided by the same smoothing of the mask) and then relaxed by 200 sweeps of
  neighbour averaging, so that the fill joins the surrounding sky without a step.
* Simulated skies: Gaussian CMB as in ``benchmark_transfer_template.py`` (CAMB lensed TT, 5 arcmin beam,
  pixel window, 30 uK-arcmin white noise). They hold no foreground residuals and no anisotropic noise.
* Tiles: HEALPix Nside 16 pixel centres whose 256 x 256, 13 arcmin patch is at least 90% unmasked.
* Filter statistic of a tile: the largest |SNR| over the bank and over the unmasked pixels within 6 degrees
  of the tile centre. Network score of a tile: the largest mask probability over the patch.
* "Unfilled" variant, networks only, no injections: the masked pixels keep their own values (monopole and
  dipole still removed). For a simulated sky that is the plain Gaussian sky; for a real map it is whatever
  the map holds under the mask. This is how the April full-sky tiles were made.
* "Zeroed" variant, networks only, no injections: masked pixels set to zero. This is what the April real-sky
  training patches (``null_controls_smica_mask090.h5``) hold there.
* Signals: the ramp template of ``collision_template_transfer``; centre within 5 degrees of the tile
  centre; random sign; amplitude = observed peak Delta T / T. One position and sign per (tile, size),
  shared by all amplitudes.
* Injection, fast path: the template is added analytically, to the filter maps (filtering is linear) and to
  the patches (true patch geometry). This treats the signal as if it were visible under the mask as well.
  The exact path (template added to the raw map, then preparation, filtering and projection) is run on a
  sample of tiles to measure the error of the fast path.
* ``--inject-inputs filled,unfilled,zeroed`` repeats the injection test for the networks on the other two
  inputs (default: filled only, the pre-registered design). For the zeroed input the template is not added
  on masked pixels.

Questions
---------
1. With thresholds set at 5% of simulated tiles, what fraction of real-sky tiles exceeds them, and where
   does that fraction rank among the simulated skies (same tiles, same preparation)?
2. What is the detection probability of injected templates on the real sky, at thresholds set at 5% of the
   real sky's own tiles, next to the same quantity on simulated skies with the same mask and tiles?
3. How unusual are the strongest real-sky filter responses among the simulated skies (largest |SNR| over the
   whole searched region, per bank size and over the bank)?

Usage (from the project root):
    python scripts/analysis/benchmark_real_sky.py --out-dir OUT \
        --model name:run_dir:checkpoint [--model name:run_dir:checkpoint:state_dict_file] [--maps smica,nilc]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

for _sub in ("core", "analysis", "ml", "followup"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _sub))

import healpy as hp
import numpy as np
from scipy.stats import spearmanr

import benchmark_transfer_template as bt
import collision_template_transfer as ct
import patch_geometry as pg
from phase2_observing_model import camb_tt_cls
from recompute_candidate_positions import MCEWEN_2012

NSIDE, LMAX = bt.NSIDE, bt.LMAX
NPIX = hp.nside2npix(NSIDE)
MAP_FILES = {
    "smica": "COM_CMB_IQU-smica_2048_R3.00_full.fits",
    "nilc": "COM_CMB_IQU-nilc_2048_R3.00_full.fits",
    "sevem": "COM_CMB_IQU-sevem_2048_R3.00_full.fits",
    "commander": "COM_CMB_IQU-commander_2048_R3.00_full.fits",
}
MASK_FILE = "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
COLD_SPOT = (209.0, -57.0)            # Galactic longitude, latitude (deg)
FILTER = "matched_filter_bank"


class Preparer:
    """Remove monopole and dipole fitted outside the mask; return the map with the masked pixels filled,
    and the same map with the masked pixels left as they are (how the April tiles were made)."""

    def __init__(self, good, vec, fill_fwhm_deg=10.0, relax_sweeps=200, fill_lmax=96):
        self.good = np.asarray(good, dtype=bool)
        self.vec = vec
        self.fwhm = math.radians(fill_fwhm_deg)
        self.fill_lmax = fill_lmax
        self.relax_sweeps = relax_sweeps
        self.bad = np.flatnonzero(~self.good)
        den = hp.smoothing(self.good.astype(np.float64), fwhm=self.fwhm, lmax=fill_lmax, iter=1)
        self.den_bad = den[self.bad]
        nb = hp.get_all_neighbours(NSIDE, self.bad)          # (8, n_bad); -1 where a neighbour does not exist
        self.nb_missing = np.nonzero(nb < 0)
        self.nb = np.where(nb >= 0, nb, 0)
        self.nb_count = np.maximum(np.sum(nb >= 0, axis=0), 1)

    def __call__(self, sky):
        sky = np.asarray(sky, dtype=np.float64)
        work = sky.copy()
        work[self.bad] = hp.UNSEEN
        _, mono, dipole = hp.remove_dipole(work, bad=hp.UNSEEN, fitval=True)       # fitted on the unmasked pixels
        unfilled = sky - float(mono) - np.asarray(dipole, dtype=np.float64) @ self.vec
        work = unfilled.copy()
        work[self.bad] = 0.0
        num = hp.smoothing(work, fwhm=self.fwhm, lmax=self.fill_lmax, iter=1)
        work[self.bad] = num[self.bad] / np.maximum(self.den_bad, 0.05)    # tapers to zero deep inside the mask
        for _ in range(self.relax_sweeps):
            vals = work[self.nb]
            vals[self.nb_missing] = 0.0
            work[self.bad] = vals.sum(axis=0) / self.nb_count
        return work, unfilled


def load_models(specs, device_name, log):
    """Load the networks and return (models, score function). PyTorch and the model code are imported here,
    so the filter-only path (no --model) runs without them."""
    import torch
    import phase3_train_unet as p3
    from phase3_sensitivity_curve import build_model_from_run

    device = torch.device(device_name)
    models = {}
    for text in specs:
        parts = text.split(":")
        model, run_config, ckpt_path, _ = build_model_from_run(Path(parts[1]).resolve(), parts[2], device)
        if len(parts) > 3:
            state = torch.load(parts[3], map_location=device, weights_only=False)
            p3.load_model_state_dict(model, state["model_state_dict"])
            ckpt_path = parts[3]
        model.eval()
        models[parts[0]] = (model, float(run_config["normalization"]["train_mean"]), float(run_config["normalization"]["train_std"]))
        log(f"model {parts[0]}: {ckpt_path}")

    @torch.inference_mode()
    def score_patches(entry, patches, add=None, scale=0.0, batch=256):
        """Largest mask probability per patch; optionally scores patches + scale * add without building the sum."""
        model, mean, std = entry
        out = np.zeros(len(patches), dtype=np.float64)
        for i in range(0, len(patches), batch):
            x = torch.from_numpy(patches[i:i + batch]).to(device)
            if add is not None:
                x = x + float(scale) * torch.from_numpy(add[i:i + batch]).to(device)
            mask_logits, _ = p3.unpack_model_output(model(((x - mean) / std)[:, None]))
            out[i:i + batch] = torch.sigmoid(mask_logits).flatten(1).max(dim=1).values.double().cpu().numpy()
        return out

    return models, score_patches


def rank_p(sim_values, real_value):
    """Fraction of simulated values at least as large as the real one, counting the real one."""
    sim_values = np.asarray(sim_values)
    return float((1 + np.sum(sim_values >= real_value)) / (1 + sim_values.size))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", action="append", default=[], help="name:run_dir:checkpoint or name:run_dir:checkpoint:state_dict_file")
    ap.add_argument("--maps", default="smica,nilc,sevem,commander")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-sims", type=int, default=300)
    ap.add_argument("--subset-tiles", type=int, default=300, help="tiles on which the networks are scored in the simulations")
    ap.add_argument("--sim-injection-skies", type=int, default=8, help="simulated skies that also get injections (reference)")
    ap.add_argument("--tile-nside", type=int, default=16)
    ap.add_argument("--min-unmasked", type=float, default=0.9)
    ap.add_argument("--search-deg", type=float, default=6.0)
    ap.add_argument("--max-offset-deg", type=float, default=5.0)
    ap.add_argument("--sizes-deg", default="5,10,15,20,25")
    ap.add_argument("--amplitudes", default="1e-5,2e-5,3e-5,5e-5,7e-5,1e-4")
    ap.add_argument("--fpr-target", type=float, default=0.05)
    ap.add_argument("--fill-fwhm-deg", type=float, default=10.0)
    ap.add_argument("--relax-sweeps", type=int, default=200)
    ap.add_argument("--inject-inputs", default="filled",
                    help="network inputs that get injections: comma list of filled, unfilled, zeroed")
    ap.add_argument("--exact-checks", type=int, default=24)
    ap.add_argument("--exact-check-amplitude", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sizes = [float(x) for x in args.sizes_deg.split(",")]
    amps = [float(x) for x in args.amplitudes.split(",")]
    data = Path(args.data_dir)
    q = 1.0 - args.fpr_target
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:6.0f} s] {msg}", flush=True)

    # ---- sky model, templates, filter weights: the same as in the Gaussian benchmark -------------------
    cl_tt, _ = camb_tt_cls(lmax=LMAX)
    window = ct.gaussian_beam(LMAX, 5.0) * hp.pixwin(NSIDE, lmax=LMAX)
    pix_area = hp.nside2pixarea(NSIDE)
    sigma_pix = 30.0e-6 * math.radians(1.0 / 60.0) / math.sqrt(pix_area)
    total_power = np.asarray(cl_tt[: LMAX + 1], dtype=np.float64) * window**2 + sigma_pix**2 * pix_area
    templates, resp, fine = bt.build_templates(sizes, window, total_power)
    fwhm_rad = math.radians(5.0 / 60.0)
    vec_all = np.array(hp.pix2vec(NSIDE, np.arange(NPIX)))
    log("templates ready")

    def vec2pix(x, y, z):
        return hp.vec2pix(NSIDE, x, y, z)

    def filter_maps(sky):
        alm = hp.map2alm(sky, lmax=LMAX, iter=1)
        return np.stack([hp.alm2map(hp.almxfl(alm, templates[s]["weights"]), NSIDE, lmax=LMAX) for s in sizes])

    def simulate():
        np.random.seed(int(rng.integers(0, 2**32 - 1)))
        sky = np.asarray(hp.synfast(cl_tt, NSIDE, lmax=LMAX, new=True, pixwin=True, fwhm=fwhm_rad), dtype=np.float64)
        return sky + rng.normal(0.0, sigma_pix, size=NPIX)

    # ---- mask, preparation, tiles --------------------------------------------------------------------------
    good = hp.ud_grade(hp.read_map(str(data / MASK_FILE), field=0, dtype=np.float64), NSIDE) >= 0.9
    prep = Preparer(good, vec_all, args.fill_fwhm_deg, args.relax_sweeps)
    log(f"mask: {good.mean():.4f} of the sky kept")
    index_map = np.arange(NPIX, dtype=np.float64)
    guard_map = rng.normal(size=NPIX)
    disc_size = hp.query_disc(NSIDE, np.array([0.0, 0.0, 1.0]), math.radians(args.search_deg)).size
    tiles, idx_list = [], []
    for pix in range(hp.nside2npix(args.tile_nside)):
        lon, lat = (float(x) for x in hp.pix2ang(args.tile_nside, pix, lonlat=True))
        proj = hp.projector.GnomonicProj(rot=(lon, lat), xsize=pg.PATCH_PIX, ysize=pg.PATCH_PIX, reso=pg.RESO_ARCMIN)
        idx = np.asarray(proj.projmap(index_map, vec2pix)).astype(np.int32)
        frac = float(np.mean(good[idx]))
        if frac < args.min_unmasked:
            continue
        v = hp.ang2vec(lon, lat, lonlat=True)
        foot = hp.query_disc(NSIDE, v, math.radians(args.search_deg))
        foot = foot[good[foot]]
        if foot.size < 0.5 * disc_size:
            continue
        if not tiles:  # guard: the stored pixel indices must reproduce hp.gnomview exactly
            ref = np.asarray(hp.gnomview(guard_map, rot=(lon, lat), reso=pg.RESO_ARCMIN, xsize=pg.PATCH_PIX,
                                         return_projected_map=True, no_plot=True), dtype=np.float64)
            if not np.array_equal(ref, guard_map[idx]):
                raise SystemExit("stored patch indices differ from hp.gnomview; refusing to continue")
            log("check: patch indices reproduce hp.gnomview exactly")
        tiles.append({"lon": lon, "lat": lat, "frac": frac, "foot": foot})
        idx_list.append(idx)
    idx_all = np.stack(idx_list)
    del idx_list
    n_tiles = len(tiles)
    subset = np.sort(rng.choice(n_tiles, size=min(args.subset_tiles, n_tiles), replace=False))
    region = np.zeros(NPIX, dtype=bool)
    for t in tiles:
        region[t["foot"]] = True
    region_pix = np.flatnonzero(region)
    log(f"tiles: {n_tiles} of {hp.nside2npix(args.tile_nside)} eligible (Nside {args.tile_nside}); networks use {subset.size} of them in the "
        f"simulations; searched region {region.mean():.3f} of the sky")

    def tile_stat(absmax):
        return np.array([float(absmax[t["foot"]].max()) for t in tiles])

    def patches_of(sky, tile_ids):
        return np.ascontiguousarray(sky[idx_all[tile_ids]], dtype=np.float32)

    # ---- models ----------------------------------------------------------------------------------------------
    models, score_patches = load_models(args.model, args.device, log) if args.model else ({}, None)
    methods = [FILTER] + list(models)
    c0 = (pg.PATCH_PIX - 1) / 2.0
    max_off_px = args.max_offset_deg * 60.0 / pg.RESO_ARCMIN

    def draw_signal(tile):
        """Random sign and centre within the offset limit; returns sign, patch pixel (cx, cy), unit vector."""
        sign = 1.0 if rng.random() < 0.5 else -1.0
        rr = max_off_px * math.sqrt(rng.random())
        phi = 2 * math.pi * rng.random()
        cx, cy = c0 + rr * math.cos(phi), c0 + rr * math.sin(phi)
        slon, slat = pg.pixel_to_lonlat(tile["lon"], tile["lat"], cx, cy)
        return sign, cx, cy, hp.ang2vec(float(slon), float(slat), lonlat=True)

    inject_suffix = {"filled": "", "unfilled": "_unfilled", "zeroed": "_zeroed"}
    inject_inputs = [x for x in args.inject_inputs.split(",") if x]
    inj_methods = [FILTER] + [f"{name}{inject_suffix[v]}" for v in inject_inputs for name in models]

    def injection_scores(snr, patch_sets, tile_ids):
        """Scores with injected templates, fast path: method -> array [amplitude, size, tile].

        ``patch_sets`` maps an input name (filled, unfilled, zeroed) to the signal-free patches of ``tile_ids``.
        The template is added on every pixel, except for the zeroed input, where masked pixels stay zero.
        """
        res = {m: np.zeros((len(amps), len(sizes), len(tile_ids))) for m in inj_methods}
        unmasked = good[idx_all[tile_ids]].astype(np.float32) if (models and "zeroed" in inject_inputs) else None
        for isz, s in enumerate(sizes):
            tmpl = np.empty((len(tile_ids), pg.PATCH_PIX, pg.PATCH_PIX), dtype=np.float32) if models else None
            resp_foot = []
            for j, i in enumerate(tile_ids):
                sign, cx, cy, sv = draw_signal(tiles[i])
                if models:
                    tmpl[j] = sign * np.interp(pg.angular_distance_grid(cx, cy), fine, templates[s]["profile_k"])
                ang = np.arccos(np.clip(sv @ vec_all[:, tiles[i]["foot"]], -1.0, 1.0))
                resp_foot.append(sign * np.stack([np.interp(ang, fine, resp[(f, s)]) for f in sizes]))
            adds = {v: (tmpl * unmasked if v == "zeroed" else tmpl) for v in inject_inputs} if models else {}
            for ia, a in enumerate(amps):
                res[FILTER][ia, isz] = [float(np.max(np.abs(snr[:, tiles[i]["foot"]] + a * resp_foot[j]))) for j, i in enumerate(tile_ids)]
                for v in inject_inputs:
                    for name in models:
                        res[f"{name}{inject_suffix[v]}"][ia, isz] = score_patches(models[name], patch_sets[v], add=adds[v], scale=a)
        return res

    # ---- simulated skies: thresholds, per-sky trigger fractions, region maxima, injection reference ---------
    # Networks are also scored on patches whose masked pixels are left as they are ("unfilled"): for a
    # simulated sky that is the plain Gaussian sky, for a real map it is the map's own content under the mask.
    # "zeroed": masked pixels set to zero, which is what the April real-sky training patches held there.
    unfilled_names = [f"{name}{suffix}" for name in models for suffix in ("_unfilled", "_zeroed")]
    sim_tile = {FILTER: np.zeros((args.n_sims, n_tiles))}
    for name in list(models) + unfilled_names:
        sim_tile[name] = np.zeros((args.n_sims, subset.size))
    sim_region_max = np.zeros((args.n_sims, len(sizes)))
    sim_region_std = np.zeros((args.n_sims, len(sizes)))
    sim_inj = []
    for isim in range(args.n_sims):
        sky, unfilled = prep(simulate())
        snr = filter_maps(sky)
        sim_tile[FILTER][isim] = tile_stat(np.max(np.abs(snr), axis=0))
        sim_region_max[isim] = np.max(np.abs(snr[:, region_pix]), axis=1)
        sim_region_std[isim] = np.std(snr[:, region_pix], axis=1)
        pt = patches_of(sky, subset) if models else None
        pu = patches_of(unfilled, subset) if models else None
        pz = patches_of(np.where(good, unfilled, 0.0), subset) if models else None
        for name in models:
            sim_tile[name][isim] = score_patches(models[name], pt)
            sim_tile[f"{name}_unfilled"][isim] = score_patches(models[name], pu)
            sim_tile[f"{name}_zeroed"][isim] = score_patches(models[name], pz)
        if isim < args.sim_injection_skies:
            sim_inj.append(injection_scores(snr, {"filled": pt, "unfilled": pu, "zeroed": pz}, subset))
        if (isim + 1) % 20 == 0 or isim + 1 == args.n_sims:
            log(f"simulated sky {isim + 1}/{args.n_sims}")
    null_methods = methods + unfilled_names
    sim_thr = {m: float(np.quantile(sim_tile[m].ravel(), q)) for m in null_methods}
    sim_frac = {m: np.mean(sim_tile[m] > sim_thr[m], axis=1) for m in null_methods}
    sim_bank_max = np.max(sim_region_max, axis=1)
    sim_det = {m: np.mean(np.concatenate([r[m] for r in sim_inj], axis=2) > sim_thr[m], axis=2) for m in inj_methods} if sim_inj else {}
    tile_frac = np.array([t["frac"] for t in tiles])
    clean = tile_frac >= 0.97                       # tiles with little masked area

    print("\n===== simulated skies (Gaussian, same mask, same tiles) =====")
    print("std of the SNR maps over the searched region, per bank size:", [round(float(x), 3) for x in sim_region_std.mean(axis=0)])
    print(f"tiles at least 97% unmasked: {int(clean.sum())} of {n_tiles}")
    print(f"{'method':32s} {'threshold':>10s} {'per-sky trigger fraction: mean':>32s} {'2.5%':>7s} {'16%':>7s} {'84%':>7s} {'97.5%':>7s} {'max':>7s}")
    for m in null_methods:
        f = sim_frac[m]
        print(f"{m:32s} {sim_thr[m]:10.4f} {f.mean():32.3f} " + " ".join(f"{np.quantile(f, x):7.3f}" for x in (0.025, 0.16, 0.84, 0.975)) + f" {f.max():7.3f}")
    if sim_det:
        print(f"detection on {len(sim_inj)} simulated skies x {subset.size} tiles, all sizes; amplitudes {amps}")
        for m in inj_methods:
            print(f"  {m:32s} " + "  ".join(f"{x:.3f}" for x in sim_det[m].mean(axis=1)))
    sys.stdout.flush()

    report = {
        "n_tiles": n_tiles, "subset_tiles": int(subset.size), "n_sims": args.n_sims, "sizes_deg": sizes, "amplitudes": amps,
        "mask_sky_fraction": float(good.mean()), "region_sky_fraction": float(region.mean()), "sim_thresholds": sim_thr,
        "sim_region_snr_std": sim_region_std.mean(axis=0).tolist(),
        "sim_trigger_fraction": {m: {"mean": float(sim_frac[m].mean()), **{f"q{round(1000 * x):04d}": float(np.quantile(sim_frac[m], x)) for x in (0.025, 0.16, 0.5, 0.84, 0.975)},
                                     "max": float(sim_frac[m].max())} for m in null_methods},
        "tiles_at_least_97pct_unmasked": int(clean.sum()),
        "sim_detection_by_amp_size": {m: sim_det[m].tolist() for m in sim_det}, "sim_injection_skies": len(sim_inj),
        "maps": {}, "args": vars(args),
    }
    saved = {"tile_lon": np.array([t["lon"] for t in tiles]), "tile_lat": np.array([t["lat"] for t in tiles]),
             "tile_frac": tile_frac, "subset": subset, "sim_region_max": sim_region_max,
             "sim_region_std": sim_region_std, **{f"sim__{m}": sim_tile[m] for m in null_methods}}
    all_ids = np.arange(n_tiles)

    # ---- real maps ----------------------------------------------------------------------------------------------
    for imap, mname in enumerate(args.maps.split(",")):
        raw = hp.ud_grade(hp.read_map(str(data / MAP_FILES[mname]), field=0, dtype=np.float64), NSIDE)
        odd = ~np.isfinite(raw) | (np.abs(raw) > 1.0)
        if np.any(odd & good):
            raise SystemExit(f"{mname}: unmasked pixels that are not finite or lie outside the Kelvin anisotropy scale")
        sky, unfilled = prep(raw)
        if odd.any():      # only under the mask; the unfilled variant takes the fill there
            unfilled[odd] = sky[odd]
            print(f"{mname}: {int(odd.sum())} masked pixels are not finite or exceed 1 K; the unfilled variant uses the fill there")
        snr = filter_maps(sky)
        absmax = np.max(np.abs(snr), axis=0)
        argsize = np.argmax(np.abs(snr), axis=0)
        peak_pix = np.array([int(t["foot"][int(np.argmax(absmax[t["foot"]]))]) for t in tiles])
        null = {FILTER: absmax[peak_pix]}
        real_patches = patches_of(sky, all_ids) if models else None
        unfilled_patches = patches_of(unfilled, all_ids) if models else None
        zeroed_patches = patches_of(np.where(good, unfilled, 0.0), all_ids) if models else None
        for name in models:
            null[name] = score_patches(models[name], real_patches)
            null[f"{name}_unfilled"] = score_patches(models[name], unfilled_patches)
            null[f"{name}_zeroed"] = score_patches(models[name], zeroed_patches)
        own_thr = {m: float(np.quantile(null[m], q)) for m in null_methods}
        rec = {"rms_outside_mask_uK": float(np.std(sky[good]) * 1e6), "own_thresholds": own_thr, "trigger_fraction": {}, "region": {}}
        print(f"\n===== {mname}: rms outside the mask {rec['rms_outside_mask_uK']:.1f} uK; {n_tiles} tiles =====")

        # question 1: real-sky trigger fraction at the simulation thresholds
        print(f"{'method':32s} {'real tiles above sim threshold':>31s} {'same tiles as the sims':>23s} {'rank among simulated skies':>28s} {'tiles >= 97% unmasked':>22s} {'the rest':>9s}")
        for m in null_methods:
            same = null[m] if m == FILTER else null[m][subset]
            frac_all, frac_same = float(np.mean(null[m] > sim_thr[m])), float(np.mean(same > sim_thr[m]))
            p_hi = rank_p(sim_frac[m], frac_same)
            p_lo = rank_p(-sim_frac[m], -frac_same)
            f_clean, f_rest = float(np.mean(null[m][clean] > sim_thr[m])), float(np.mean(null[m][~clean] > sim_thr[m]))
            rec["trigger_fraction"][m] = {"all_tiles": frac_all, "same_tiles_as_sims": frac_same, "p_at_least_this_high": p_hi, "p_at_least_this_low": p_lo,
                                          "tiles_at_least_97pct_unmasked": f_clean, "other_tiles": f_rest}
            print(f"{m:32s} {frac_all:31.3f} {frac_same:23.3f} {('P(>=) %.3f, P(<=) %.3f' % (p_hi, p_lo)):>28s} {f_clean:22.3f} {f_rest:9.3f}")
        for name in models:
            rho = float(spearmanr(null[name], null[FILTER])[0])
            rec["trigger_fraction"][name]["spearman_with_filter"] = rho
            print(f"rank correlation of {name} scores with the filter statistic over tiles: {rho:.3f}")

        # question 3: strongest filter responses over the searched region
        print(f"{'bank size':>10s} {'SNR std (sims)':>18s} {'largest |SNR|':>14s} {'P(sim >= real)':>15s} {'sign':>5s} {'lon':>7s} {'lat':>7s}")
        for isz, s in enumerate(sizes):
            vals = snr[isz][region_pix]
            j = int(region_pix[int(np.argmax(np.abs(vals)))])
            lon_p, lat_p = (float(x) for x in hp.pix2ang(NSIDE, j, lonlat=True))
            r = {"snr_std": float(np.std(vals)), "max_abs_snr": float(abs(snr[isz][j])), "sign": float(np.sign(snr[isz][j])), "lon": lon_p, "lat": lat_p,
                 "p_global": rank_p(sim_region_max[:, isz], abs(snr[isz][j]))}
            rec["region"][f"{s:g}"] = r
            print(f"{s:10.0f} {r['snr_std']:8.3f} ({sim_region_std[:, isz].mean():.3f}) {r['max_abs_snr']:14.2f} {r['p_global']:15.3f} {r['sign']:+5.0f} {lon_p:7.1f} {lat_p:7.1f}")
        bank_real = float(np.max(absmax[region_pix]))
        rec["region"]["bank"] = {"max_abs_snr": bank_real, "p_global": rank_p(sim_bank_max, bank_real)}
        print(f"whole bank: largest |SNR| {bank_real:.2f}; P(sim >= real) = {rec['region']['bank']['p_global']:.3f} over {args.n_sims} simulated skies")
        top = []
        for i in np.argsort(-null[FILTER]):
            lon_p, lat_p = (float(x) for x in hp.pix2ang(NSIDE, int(peak_pix[i]), lonlat=True))
            if any(float(pg.angular_separation_deg(lon_p, lat_p, c["lon"], c["lat"])) < 10.0 for c in top):
                continue
            isz = int(argsize[peak_pix[i]])
            near = min(MCEWEN_2012, key=lambda mc: float(pg.angular_separation_deg(lon_p, lat_p, mc[3], 90.0 - mc[2])))
            top.append({"lon": lon_p, "lat": lat_p, "snr": float(snr[isz][peak_pix[i]]), "size_deg": sizes[isz],
                        "p_tile": rank_p(sim_tile[FILTER][:, i], null[FILTER][i]),
                        "nearest_mcewen": int(near[0]), "nearest_mcewen_size_deg": float(near[4]),
                        "nearest_mcewen_sep_deg": float(pg.angular_separation_deg(lon_p, lat_p, near[3], 90.0 - near[2])),
                        "cold_spot_sep_deg": float(pg.angular_separation_deg(lon_p, lat_p, *COLD_SPOT))})
            if len(top) == 8:
                break
        rec["top_filter_peaks"] = top
        print("strongest filter peaks (at least 10 deg apart): lon, lat, SNR, bank size, P for that tile, nearest McEwen 2012 entry, distance to the Cold Spot")
        for c in top:
            print(f"  ({c['lon']:6.1f},{c['lat']:6.1f})  {c['snr']:+6.2f}  {c['size_deg']:4.0f} deg  P {c['p_tile']:.3f}   #{c['nearest_mcewen']} "
                  f"({c['nearest_mcewen_size_deg']:g} deg) at {c['nearest_mcewen_sep_deg']:.1f} deg   Cold Spot at {c['cold_spot_sep_deg']:.1f} deg")

        # question 2: injected templates
        inj = injection_scores(snr, {"filled": real_patches, "unfilled": unfilled_patches, "zeroed": zeroed_patches}, all_ids)
        del unfilled_patches, zeroed_patches
        rec["detection"] = {}
        print(f"detection of injected templates, all sizes; amplitudes {amps}")
        for m in inj_methods:
            own = np.mean(inj[m] > own_thr[m], axis=2)
            sim = np.mean(inj[m] > sim_thr[m], axis=2)
            rec["detection"][m] = {"own_threshold_by_amp_size": own.tolist(), "sim_threshold_by_amp_size": sim.tolist()}
            print(f"  {m:32s} own 5% threshold: " + "  ".join(f"{x:.3f}" for x in own.mean(axis=1)) + "   | simulation threshold: " + "  ".join(f"{x:.3f}" for x in sim.mean(axis=1)))
            if sim_det:
                print(f"  {'':32s} simulated skies : " + "  ".join(f"{x:.3f}" for x in sim_det[m].mean(axis=1)))

        # exact path against the fast path, first map only
        if imap == 0 and args.exact_checks > 0:
            a = args.exact_check_amplitude
            diffs = {m: [] for m in methods}
            patch_rms = []
            for i in rng.choice(n_tiles, size=min(args.exact_checks, n_tiles), replace=False):
                tile = tiles[i]
                s = sizes[int(rng.integers(len(sizes)))]
                sign, cx, cy, sv = draw_signal(tile)
                ang_all = np.arccos(np.clip(sv @ vec_all, -1.0, 1.0))
                sky_x, _ = prep(raw + sign * a * np.interp(ang_all, fine, templates[s]["profile_k"]))
                snr_x = filter_maps(sky_x)
                fast = max(float(np.max(np.abs(snr[k][tile["foot"]] + sign * a * np.interp(ang_all[tile["foot"]], fine, resp[(f, s)])))) for k, f in enumerate(sizes))
                diffs[FILTER].append((float(np.max(np.abs(snr_x[:, tile["foot"]]))), fast))
                p_exact = patches_of(sky_x, np.array([i]))
                p_fast = patches_of(sky, np.array([i])) + (sign * a * np.interp(pg.angular_distance_grid(cx, cy), fine, templates[s]["profile_k"])).astype(np.float32)[None]
                patch_rms.append(float(np.sqrt(np.mean((p_exact - p_fast) ** 2)) * 1e6))
                for name in models:
                    diffs[name].append((float(score_patches(models[name], p_exact)[0]), float(score_patches(models[name], p_fast)[0])))
            rec["exact_check"] = {"amplitude": a, "n": len(patch_rms), "patch_rms_difference_uK_mean": float(np.mean(patch_rms)), "methods": {}}
            print(f"exact injection against the fast path, {len(patch_rms)} tiles at amplitude {a:g}: patch rms difference {np.mean(patch_rms):.2f} uK (mean)")
            for m in methods:
                d = np.asarray(diffs[m])
                e = {"mean_exact": float(d[:, 0].mean()), "mean_fast": float(d[:, 1].mean()), "mean_abs_difference": float(np.mean(np.abs(d[:, 0] - d[:, 1]))),
                     "max_abs_difference": float(np.max(np.abs(d[:, 0] - d[:, 1]))),
                     "detected_exact": float(np.mean(d[:, 0] > own_thr[m])), "detected_fast": float(np.mean(d[:, 1] > own_thr[m]))}
                rec["exact_check"]["methods"][m] = e
                print(f"  {m:24s} mean exact {e['mean_exact']:.4f}, mean fast {e['mean_fast']:.4f}, mean |difference| {e['mean_abs_difference']:.4f}, "
                      f"largest {e['max_abs_difference']:.4f}; above own threshold: exact {e['detected_exact']:.2f}, fast {e['detected_fast']:.2f}")

        report["maps"][mname] = rec
        saved[f"peak_pix__{mname}"] = peak_pix
        for m in null_methods:
            saved[f"null__{mname}__{m}"] = null[m]
        for m in inj_methods:
            saved[f"inj__{mname}__{m}"] = inj[m].astype(np.float32)
        (out / "real_sky_report.json").write_text(json.dumps(report, indent=1))
        log(f"{mname} done")
    np.savez_compressed(out / "real_sky_scores.npz", **saved)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
