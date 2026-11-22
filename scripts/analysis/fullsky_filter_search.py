"""Full-sky optimal-filter search for bubble-collision candidates, after McEwen et al. (2012).

Procedure (McEwen, Feeney, Johnson, Peiris, arXiv:1202.2861, Secs. III C and IV):
  1. The masked sky is filtered with the matched filter of every candidate size (their 33 sizes, 1 to 90 deg).
  2. Significance S_R(n) = |w_R(n)| / sigma_R(n), sigma per pixel from Gaussian simulations that went through
     the same mask and preparation. (They also subtract a per-pixel mean. The preparation here is linear and
     the simulations have zero mean, so the mean is zero by construction; it is measured and reported.)
  3. S below the threshold N_sigma(R) is set to zero. Every connected region that is left is a potential
     detection, placed at its pixel of largest S.
  4. A potential detection is dropped if a region on an adjacent size (|R' - R| <= 5 deg) overlaps it, has
     the same sign and a larger filtered-field amplitude. What is left are the candidates.
  5. Thresholds, per size: the published table of the same group, Feeney et al. (2013, arXiv:1210.2725,
     Table 1): 4.25 at 1 to 2 deg, 4.00 at 3 to 8, 3.75 at 9 to 16, 3.50 at 18 to 26, 3.25 at 28 to 75, 3.00 at
     80 to 90 deg. They were set on one end-to-end simulation of WMAP so that at most three detections per
     size survive step 4. (``--thresholds calibrated`` instead uses, per size, the lowest value for which a
     simulated sky has on average at most two regions before step 4; stage calib.)

Assumptions and choices that are this project's, not theirs
----------------------------------------------------------
* Two templates, each with its own filters, thresholds and candidates:
  "cosine"   their template with z_crit = 0: (cos t - cos t_c) / (1 - cos t_c) inside the disc;
  "transfer" the ramp through the CAMB transfer functions (``collision_template_transfer``).
  Both carry the beam and the pixel window; monopole and dipole are dropped; unit amplitude = unit observed
  peak Delta T / T.
* Filter: w(n) = sum_lm (t_l / D_l) a_lm Y_lm(n) / sqrt(norm), norm = sum_l (2l+1)/(4 pi) t_l^2 / D_l, D_l the
  total power of the simulated sky. The amplitude estimate is w / sqrt(norm).
* Preparation of a sky, real or simulated: monopole and dipole fitted on the unmasked pixels and removed,
  masked pixels set to zero. The search runs over the unmasked pixels only.
* A potential detection is one connected region (8 HEALPix neighbours) of pixels at or above the threshold.
  Step 4 compares the largest |amplitude| inside the two overlapping regions (my reading of their text).
* Data sets:
  "planck"  PR3 SMICA, NILC, SEVEM, Commander at Nside 256 (ud_grade), common mask >= 0.9, 5 arcmin beam,
            l <= 256 as in their search, simulations as in ``benchmark_real_sky.py`` (white noise 30 uK-arcmin);
  "planck767"  the same with l <= 767, the limit of the Nside 256 maps;
  "wmap7w"  WMAP 7-year foreground-reduced W band, the map of their search, at Nside 256, extended
            temperature analysis mask (KQ75y7) >= 0.9, 13.2 arcmin beam, l <= 256 as in their paper, noise
            sigma_0 / sqrt(N_obs) with sigma_0 = 6.549 mK, WMAP 7-year parameters.
* Simulated skies are Gaussian; they hold no foreground residuals.

Stages: selftest, stats (sigma maps), calib (only with ``--thresholds calibrated``), null (the algorithm on
simulated skies), data (the real maps). ``--stage all`` runs them in order.

Usage (from the project root):
    python scripts/analysis/fullsky_filter_search.py --dataset planck --out-dir OUT --workers 20
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _sub in ("core", "analysis", "ml", "followup"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / _sub))

import healpy as hp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import benchmark_transfer_template as bt
import collision_template_transfer as ct
from phase2_observing_model import camb_tt_cls
from recompute_candidate_positions import MCEWEN_2012

SIZES = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0,
         30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
TEMPLATES = ("cosine", "transfer")
NS = len(SIZES)
NF = NS * len(TEMPLATES)                      # filter index f = template_index * NS + size_index
THETA_ADJ = 5.0
THRESHOLDS = np.round(np.arange(2.0, 6.0001, 0.05), 2)
TARGET_PER_SIZE = 2.0
PUBLISHED_THRESHOLDS = [(2.0, 4.25), (8.0, 4.00), (16.0, 3.75), (26.0, 3.50), (75.0, 3.25), (90.0, 3.00)]   # up to size, N_sigma


def published_threshold(size):
    return next(t for upto, t in PUBLISHED_THRESHOLDS if size <= upto)
NSIDE = 256
NPIX = hp.nside2npix(NSIDE)
LMAX_SIM = 3 * NSIDE - 1
WMAP7 = {"H0": 70.4, "ombh2": 0.02260, "omch2": 0.1123, "tau": 0.087, "As": 2.441e-9, "ns": 0.963}
WMAP_W_SIGMA0_K = 6.549e-3
PLANCK_MAPS = {
    "smica": "COM_CMB_IQU-smica_2048_R3.00_full.fits",
    "nilc": "COM_CMB_IQU-nilc_2048_R3.00_full.fits",
    "sevem": "COM_CMB_IQU-sevem_2048_R3.00_full.fits",
    "commander": "COM_CMB_IQU-commander_2048_R3.00_full.fits",
}
PLANCK_MASK = "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
WMAP_FILES = {"wmap7_w_forered": ("wmap7/wmap_band_forered_iqumap_r9_7yr_W_v4.fits", 0),
              "wmap7_w_raw": ("wmap7/wmap_band_imap_r9_7yr_W_v4.fits", 0)}
WMAP_NOBS = ("wmap7/wmap_band_imap_r9_7yr_W_v4.fits", 1)
WMAP_MASK = "wmap7/wmap_ext_temperature_analysis_mask_r9_7yr_v4.fits"
STAGE_CODE = {"stats": 1, "calib": 2, "null": 3, "inject": 4, "selftest": 5}

G = {}          # state shared with forked workers (set once in the parent, read-only afterwards)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f} s] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- data sets
def load_dataset(name, data_dir, seed):
    pix_area = hp.nside2pixarea(NSIDE)
    if name in ("planck", "planck767"):
        lmax, fwhm_arcmin, transfer_params = (256 if name == "planck" else LMAX_SIM), 5.0, None
        cl, _ = camb_tt_cls(lmax=LMAX_SIM)
        good = hp.ud_grade(hp.read_map(str(data_dir / PLANCK_MASK), field=0, dtype=np.float64), NSIDE) >= 0.9
        noise_sigma = np.full(NPIX, 30.0e-6 * math.radians(1.0 / 60.0) / math.sqrt(pix_area))
        maps = {k: (data_dir / v, 0, 1.0) for k, v in PLANCK_MAPS.items()}
    elif name == "wmap7w":
        lmax, fwhm_arcmin, transfer_params = 256, 13.2, WMAP7
        cl, _ = camb_tt_cls(lmax=LMAX_SIM, params=WMAP7)
        good = hp.ud_grade(hp.read_map(str(data_dir / WMAP_MASK), field=0, dtype=np.float64), NSIDE) >= 0.9
        nobs = hp.read_map(str(data_dir / WMAP_NOBS[0]), field=WMAP_NOBS[1], dtype=np.float64)
        var512 = WMAP_W_SIGMA0_K**2 / np.maximum(nobs, 1.0)
        noise_sigma = np.sqrt(hp.ud_grade(var512, NSIDE) / 4.0)       # mean of four Nside 512 pixels
        maps = {k: (data_dir / v[0], v[1], 1.0e-3) for k, v in WMAP_FILES.items()}       # mK -> K
    else:
        raise ValueError(name)
    cl = np.asarray(cl[: LMAX_SIM + 1], dtype=np.float64)
    window = ct.gaussian_beam(lmax, fwhm_arcmin) * hp.pixwin(NSIDE, lmax=lmax)
    noise_power = float(np.mean(noise_sigma[good] ** 2)) * pix_area
    total_power = cl[: lmax + 1] * window**2 + noise_power
    return {"name": name, "lmax": lmax, "fwhm_rad": math.radians(fwhm_arcmin / 60.0), "cl": cl, "window": window,
            "total_power": total_power, "good": good, "noise_sigma": noise_sigma, "maps": maps,
            "transfer_params": transfer_params, "seed": int(seed), "vec": np.array(hp.pix2vec(NSIDE, np.arange(NPIX))),
            "noise_power": noise_power}


# ---------------------------------------------------------------------------------------------- templates
def cosine_coefficients(theta_c, lmax):
    """t_l = 2 pi * integral_{mu_c}^{1} (mu - mu_c) / (1 - mu_c) P_l(mu) dmu. Gauss-Legendre on [mu_c, 1] is
    exact here: the integrand is a polynomial of degree l + 1."""
    mu_c = math.cos(theta_c)
    x, w = np.polynomial.legendre.leggauss(lmax // 2 + 8)
    mu = 0.5 * (1.0 - mu_c) * x + 0.5 * (1.0 + mu_c)
    f = (mu - mu_c) / (1.0 - mu_c)
    return 2.0 * math.pi * ((0.5 * (1.0 - mu_c) * w * f) @ np.polynomial.legendre.legvander(mu, lmax))


def finish_template(b, window, total_power, fine):
    """From raw coefficients b_l to: t_l in Kelvin per unit observed peak amplitude, filter weights, norm.

    Sign convention: a positive amplitude is a disc that is hot at its centre. ``peak_value`` is the largest
    |value| of the windowed profile without monopole and dipole, for the raw coefficients as given; for the
    cosine template (raw central amplitude 1) the central amplitude z0 of McEwen et al. is amplitude / peak_value.
    """
    lmax = b.size - 1
    ells = np.arange(lmax + 1)
    coeff = (2 * ells + 1) / (4 * math.pi) * b * window
    coeff[:2] = 0.0
    prof = np.polynomial.legendre.legval(np.cos(fine), coeff)
    peak = int(np.argmax(np.abs(prof)))
    centre_sign = 1.0 if prof[0] >= 0.0 else -1.0
    t_l = b * window * centre_sign / abs(prof[peak]) * bt.T_CMB_K
    t_l[:2] = 0.0
    norm = float(np.sum((2 * ells[2:] + 1) / (4 * math.pi) * t_l[2:] ** 2 / total_power[2:]))
    weights = np.where(ells >= 2, t_l / np.maximum(total_power, 1e-300), 0.0) / math.sqrt(norm)
    return t_l, weights, norm, float(np.degrees(fine[peak])), float(abs(prof[peak]))


def build_bank(ds):
    lmax = ds["lmax"]
    fine = np.radians(np.linspace(0.0, 180.0, 7201))
    tr = ct.camb_transfer(lmax=lmax, params=ds["transfer_params"])
    t_l = np.zeros((NF, lmax + 1))
    weights = np.zeros((NF, lmax + 1))
    norm = np.zeros(NF)
    peak_angle = np.zeros(NF)
    z0_factor = np.ones(NF)          # amplitude -> central amplitude z0 of the full cosine template
    for si, size in enumerate(SIZES):
        tc = math.radians(size)
        raw = {"cosine": cosine_coefficients(tc, lmax), "transfer": np.zeros(lmax + 1)}
        raw["transfer"][tr.ells] = ct.b_ell(tr.ells, tr.k, tr.delta_t, tr.d_star * math.cos(tc), a_ramp=1.0 / tr.d_star)
        for ti, tname in enumerate(TEMPLATES):
            f = ti * NS + si
            t_l[f], weights[f], norm[f], peak_angle[f], peak_value = finish_template(raw[tname], ds["window"], ds["total_power"], fine)
            if tname == "cosine":
                z0_factor[f] = 1.0 / peak_value
    return {"t_l": t_l, "weights": weights, "norm": norm, "peak_angle": peak_angle, "z0_factor": z0_factor, "d_star": tr.d_star}


# ---------------------------------------------------------------------------------------------- sky handling
def simulate(isim, stage):
    ds = G["ds"]
    ss = np.random.SeedSequence([ds["seed"], STAGE_CODE[stage], int(isim)])
    np.random.seed(int(ss.generate_state(1)[0]))
    sky = np.asarray(hp.synfast(ds["cl"], NSIDE, lmax=LMAX_SIM, new=True, pixwin=True, fwhm=ds["fwhm_rad"]), dtype=np.float64)
    rng = np.random.default_rng(ss.spawn(1)[0])
    return sky + rng.normal(size=NPIX) * ds["noise_sigma"]


def prepare(sky):
    ds = G["ds"]
    work = np.where(ds["good"], sky, hp.UNSEEN)
    _, mono, dipole = hp.remove_dipole(work, bad=hp.UNSEEN, fitval=True)
    return np.where(ds["good"], sky - float(mono) - np.asarray(dipole, dtype=np.float64) @ ds["vec"], 0.0)


def filtered(prepared):
    """Generator over the NF filtered maps (float64, SNR units) of one prepared sky."""
    ds, bank = G["ds"], G["bank"]
    alm = hp.map2alm(prepared, lmax=ds["lmax"], iter=1)
    for f in range(NF):
        yield f, hp.alm2map(hp.almxfl(alm, bank["weights"][f]), NSIDE, lmax=ds["lmax"])


# ---------------------------------------------------------------------------------------------- regions
def _edges(S, t0):
    """Pixels with S >= t0 and the neighbour pairs among them (local indices), each pair once."""
    idx = np.flatnonzero(S >= t0)
    n0 = idx.size
    if n0 == 0:
        return idx, None, None, None
    lookup = G["lookup"]
    lookup[idx] = np.arange(n0, dtype=np.int32)
    q = G["nb"][:, idx]
    j = np.where(q >= 0, lookup[np.where(q >= 0, q, 0)], -1)
    rows = np.broadcast_to(np.arange(n0, dtype=np.int32), (8, n0))
    keep = j > rows
    a, b = rows[keep], j[keep]
    lookup[idx] = -1
    s_nodes = S[idx]
    return idx, a, b, s_nodes


def region_counts(S, thresholds):
    """Number of connected regions of {S >= t} for every t in the ascending list ``thresholds``."""
    counts = np.zeros(len(thresholds), dtype=np.int32)
    idx, a, b, s_nodes = _edges(S, thresholds[0])
    n0 = idx.size
    if n0 == 0:
        return counts
    ew = np.minimum(s_nodes[a], s_nodes[b])
    for k, t in enumerate(thresholds):
        n_t = int(np.count_nonzero(s_nodes >= t))
        if n_t == 0:
            break
        e = ew >= t
        g = coo_matrix((np.ones(int(e.sum()), dtype=np.int8), (a[e], b[e])), shape=(n0, n0))
        ncomp = connected_components(g, directed=False, return_labels=False)
        counts[k] = ncomp - (n0 - n_t)          # pixels below t are isolated nodes of g
    return counts


def regions_at(S, t):
    """Connected regions of {S >= t}: list of sorted pixel-index arrays."""
    idx, a, b, _ = _edges(S, t)
    n0 = idx.size
    if n0 == 0:
        return []
    g = coo_matrix((np.ones(a.size, dtype=np.int8), (a, b)), shape=(n0, n0))
    ncomp, labels = connected_components(g, directed=False)
    order = np.argsort(labels, kind="stable")
    cuts = np.flatnonzero(np.diff(labels[order])) + 1
    return [np.sort(idx[part]) for part in np.split(order, cuts)]


# ---------------------------------------------------------------------------------------------- the algorithm
def significance(f, w):
    S = np.abs(w) / G["sigma"][f]
    S[~G["ds"]["good"]] = 0.0
    return S


def run_algorithm(w_all):
    """McEwen et al. steps 2 to 5 on the NF filtered maps of one sky. Returns candidates, dropped potential
    detections, the largest S per filter and the number of potential detections per filter."""
    norm, thr = G["bank"]["norm"], G["thr"]
    max_s = np.zeros(NF)
    n_pot = np.zeros(NF, dtype=np.int32)
    kept, dropped = [], []
    for ti in range(len(TEMPLATES)):
        pot = []
        for si, size in enumerate(SIZES):
            f = ti * NS + si
            w = np.asarray(w_all[f], dtype=np.float64)
            S = significance(f, w)
            max_s[f] = float(S.max())
            regs = regions_at(S, thr[f])
            n_pot[f] = len(regs)
            for pix in regs:
                amp = w[pix] / math.sqrt(norm[f])
                ip = int(pix[np.argmax(S[pix])])
                im = int(np.argmax(np.abs(amp)))
                pot.append({"template": TEMPLATES[ti], "f": f, "size": size, "pix": pix, "peak": ip, "S": float(S[ip]),
                            "amp": float(w[ip] / math.sqrt(norm[f])), "amp_max": float(amp[im]), "n_pix": int(pix.size)})
        for d in pot:
            loser = False
            for e in pot:
                if e is d or e["size"] == d["size"] or abs(e["size"] - d["size"]) > THETA_ADJ:
                    continue
                if e["amp_max"] * d["amp_max"] <= 0.0 or abs(e["amp_max"]) <= abs(d["amp_max"]):
                    continue
                if np.intersect1d(e["pix"], d["pix"], assume_unique=True).size:
                    loser = True
                    break
            (dropped if loser else kept).append(d)
    return kept, dropped, max_s, n_pot


def describe(d):
    lon, lat = (float(x) for x in hp.pix2ang(NSIDE, d["peak"], lonlat=True))
    good = G["ds"]["good"]
    disc = hp.query_disc(NSIDE, hp.pix2vec(NSIDE, d["peak"]), math.radians(d["size"]))
    return {"template": d["template"], "size_deg": d["size"], "lon": lon, "lat": lat, "S": d["S"], "amplitude": d["amp"],
            "z0_mK": d["amp"] * bt.T_CMB_K * 1e3 * float(G["bank"]["z0_factor"][d["f"]]), "amp_max": d["amp_max"],
            "n_pix": d["n_pix"], "disc_unmasked_fraction": float(np.mean(good[disc]))}


# ---------------------------------------------------------------------------------------------- workers
def _stats_worker(task):
    wid, sims = task
    acc = np.zeros((2, NF, NPIX), dtype=np.float32)
    t0 = time.time()
    for isim in sims:
        for f, w in filtered(prepare(simulate(isim, "stats"))):
            acc[0, f] += w
            acc[1, f] += w * w
    return wid, len(sims), time.time() - t0, acc


def _calib_worker(task):
    sims = task
    counts = np.zeros((len(sims), NF, len(THRESHOLDS)), dtype=np.int32)
    max_s = np.zeros((len(sims), NF))
    for i, isim in enumerate(sims):
        for f, w in filtered(prepare(simulate(isim, "calib"))):
            S = significance(f, w)
            max_s[i, f] = S.max()
            counts[i, f] = region_counts(S, THRESHOLDS)
    return counts, max_s


def _null_worker(task):
    sims = task
    out = []
    for isim in sims:
        w_all = np.empty((NF, NPIX), dtype=np.float32)
        for f, w in filtered(prepare(simulate(isim, "null"))):
            w_all[f] = w
        kept, dropped, max_s, n_pot = run_algorithm(w_all)
        out.append({"isim": int(isim), "max_s": max_s, "n_pot": n_pot,
                    "n_candidates": [sum(1 for d in kept if d["template"] == t) for t in TEMPLATES],
                    "candidates": [(d["f"], d["peak"], d["S"], d["amp"]) for d in kept]})
    return out


def split(n, parts, start=0):
    return [list(range(start + i, start + n, parts)) for i in range(parts) if i < n]


# ---------------------------------------------------------------------------------------------- stages
def stage_selftest(out):
    ds, bank = G["ds"], G["bank"]
    rep = {}
    # 1. the cosine coefficients against a direct quadrature of the real-space ramp
    lmax = ds["lmax"]
    worst = 0.0
    for size in (5.0, 30.0, 90.0):
        got = cosine_coefficients(math.radians(size), lmax)
        mu, wts = np.polynomial.legendre.leggauss(3000)
        mu_c = math.cos(math.radians(size))
        g = np.where(mu > mu_c, (mu - mu_c) / (1.0 - mu_c), 0.0)
        ref = 2.0 * math.pi * ((wts * g) @ np.polynomial.legendre.legvander(mu, 60))
        worst = max(worst, float(np.max(np.abs(got[:61] - ref)) / np.max(np.abs(ref))))
    rep["cosine_coefficients_vs_quadrature"] = worst
    assert worst < 1e-4, worst
    # 2. region finder: three caps, one over a pole, one over the longitude seam, one plain; and two touching
    S = np.zeros(NPIX)
    caps = [((0.0, 90.0), 6.0), ((0.0, 0.0), 6.0), ((120.0, -40.0), 6.0)]
    for (lon, lat), rad in caps:
        S[hp.query_disc(NSIDE, hp.ang2vec(lon, lat, lonlat=True), math.radians(rad))] = 5.0
    regs = regions_at(S, 4.0)
    sizes = sorted(r.size for r in regs)
    want = sorted(hp.query_disc(NSIDE, hp.ang2vec(lon, lat, lonlat=True), math.radians(rad)).size for (lon, lat), rad in caps)
    rep["region_finder_three_caps"] = [sizes, want]
    assert sizes == want, (sizes, want)
    S[hp.query_disc(NSIDE, hp.ang2vec(120.0, -29.0, lonlat=True), math.radians(6.0))] = 3.0
    counts = region_counts(S, np.array([2.5, 3.5, 4.5, 5.5]))
    rep["region_counts_nested"] = counts.tolist()
    assert counts.tolist() == [3, 3, 3, 0], counts
    # 3. filter normalisation and unbiased amplitude on an unmasked sky
    sky = simulate(0, "selftest")
    keep_good = ds["good"]
    ds["good"] = np.ones(NPIX, dtype=bool)
    try:
        base = prepare(sky)
        lon0, lat0, amp0 = 75.0, 35.0, 1.0e-4
        v0 = hp.ang2vec(lon0, lat0, lonlat=True)
        ang = np.arccos(np.clip(v0 @ ds["vec"], -1.0, 1.0))
        ells = np.arange(lmax + 1)
        checks = {}
        for size in (5.0, 20.0, 60.0):
            si = SIZES.index(size)
            for ti, tname in enumerate(TEMPLATES):
                f = ti * NS + si
                coeff = (2 * ells + 1) / (4 * math.pi) * bank["t_l"][f]
                signal = amp0 * np.polynomial.legendre.legval(np.cos(ang), coeff)
                alm0 = hp.map2alm(base, lmax=lmax, iter=1)
                alm1 = hp.map2alm(prepare(sky + signal), lmax=lmax, iter=1)
                w0 = hp.alm2map(hp.almxfl(alm0, bank["weights"][f]), NSIDE, lmax=lmax)
                w1 = hp.alm2map(hp.almxfl(alm1, bank["weights"][f]), NSIDE, lmax=lmax)
                p0 = hp.vec2pix(NSIDE, *v0)
                rec = float((w1[p0] - w0[p0]) / math.sqrt(bank["norm"][f]) / amp0)
                checks[f"{tname}_{size:g}"] = {"recovered_over_injected": rec, "std_of_filtered_sky": float(np.std(w0)),
                                               "peak_of_response_offset_deg": float(np.degrees(ang[int(np.argmax(np.abs(w1 - w0)))]))}
                assert abs(rec - 1.0) < 0.03, (tname, size, rec)
        rep["injection_unmasked"] = checks
    finally:
        ds["good"] = keep_good
    # 4. timing of one sky
    t0 = time.time()
    n = sum(1 for _ in filtered(prepare(simulate(1, "selftest"))))
    rep["seconds_per_sky"] = time.time() - t0
    rep["n_filters"] = n
    log("selftest: " + json.dumps(rep)[:1500])
    with open(out / "selftest.json", "w") as fh:
        json.dump(rep, fh, indent=1)


def stage_stats(out, n_sims, workers):
    # each task is a short batch, so that the sums travel back in pieces and no float32 sum grows large
    batches = [b for sims in split(n_sims, max(workers, n_sims // 25)) for b in [sims] if b]
    tasks = [(i, sims) for i, sims in enumerate(batches)]
    total = np.zeros((2, NF, NPIX), dtype=np.float64)
    n = 0
    t_start = time.time()
    with Pool(workers) as pool:
        for wid, count, secs, part in pool.imap_unordered(_stats_worker, tasks):
            total += part
            n += count
            del part
            if wid % 10 == 0:
                log(f"stats: batch {wid} done, {count} skies in {secs:.0f} s ({secs / max(count, 1):.1f} s per sky); "
                    f"{n} of {n_sims} skies after {time.time() - t_start:.0f} s")
    mean = total[0] / n
    sigma = np.sqrt(total[1] / n)
    good = G["ds"]["good"]
    z = np.abs(mean[:, good]) / (sigma[:, good] / math.sqrt(n))
    rep = {"n_sims": n, "mean_over_error_rms_per_filter": np.sqrt(np.mean(z**2, axis=1)).tolist(),
           "sigma_median_unmasked": np.median(sigma[:, good], axis=1).tolist(),
           "sigma_min_unmasked": sigma[:, good].min(axis=1).tolist(), "sigma_max_unmasked": sigma[:, good].max(axis=1).tolist()}
    log("stats: rms of mean / (sigma / sqrt(N)) over unmasked pixels, per filter, min %.3f max %.3f (1 expected); median sigma %.3f to %.3f"
        % (min(rep["mean_over_error_rms_per_filter"]), max(rep["mean_over_error_rms_per_filter"]),
           min(rep["sigma_median_unmasked"]), max(rep["sigma_median_unmasked"])))
    np.save(out / "sigma.npy", sigma.astype(np.float32))
    with open(out / "stats.json", "w") as fh:
        json.dump(rep, fh, indent=1)


def stage_calib(out, n_sims, workers):
    with Pool(workers) as pool:
        parts = pool.map(_calib_worker, split(n_sims, workers))
    counts = np.concatenate([p[0] for p in parts], axis=0)
    max_s = np.concatenate([p[1] for p in parts], axis=0)
    mean_counts = counts.mean(axis=0)                           # (NF, NT)
    thr = np.zeros(NF)
    for f in range(NF):
        # the lowest grid value above which the mean count stays at or below the target (regions merge at low
        # thresholds, so the count is not monotonic there)
        above = np.flatnonzero(mean_counts[f] > TARGET_PER_SIZE)
        thr[f] = THRESHOLDS[0] if not above.size else THRESHOLDS[min(above[-1] + 1, len(THRESHOLDS) - 1)]
    rep = {"n_sims": int(counts.shape[0]), "thresholds_grid": THRESHOLDS.tolist(), "thresholds": thr.tolist(),
           "mean_counts_at_threshold": [float(mean_counts[f, int(np.argmin(np.abs(THRESHOLDS - thr[f])))]) for f in range(NF)],
           "mean_counts": mean_counts.tolist(), "max_s_median": np.median(max_s, axis=0).tolist()}
    for ti, tname in enumerate(TEMPLATES):
        log("calib: thresholds %s: " % tname + " ".join("%g:%.2f" % (SIZES[si], thr[ti * NS + si]) for si in range(NS)))
    with open(out / "thresholds.json", "w") as fh:
        json.dump(rep, fh, indent=1)


def stage_null(out, n_sims, workers):
    with Pool(workers) as pool:
        parts = pool.map(_null_worker, split(n_sims, workers))
    rows = sorted((r for p in parts for r in p), key=lambda r: r["isim"])
    max_s = np.array([r["max_s"] for r in rows])
    n_pot = np.array([r["n_pot"] for r in rows])
    n_cand = np.array([r["n_candidates"] for r in rows])
    cand = np.array([(r["isim"],) + c for r in rows for c in r["candidates"]], dtype=np.float64).reshape(-1, 5)
    np.savez_compressed(out / "null.npz", max_s=max_s, n_pot=n_pot, n_candidates=n_cand, candidates=cand)
    for ti, tname in enumerate(TEMPLATES):
        c = n_cand[:, ti]
        log("null: %s candidates per simulated sky: mean %.2f, median %.0f, 5%%-95%% %d-%d, max %d; potential detections per size: mean %.2f"
            % (tname, c.mean(), np.median(c), np.percentile(c, 5), np.percentile(c, 95), c.max(), n_pot[:, ti * NS:(ti + 1) * NS].mean()))


def rank_p(sim_values, value):
    sim_values = np.asarray(sim_values)
    return float((1 + np.count_nonzero(sim_values >= value)) / (sim_values.size + 1))


def stage_data(out):
    ds = G["ds"]
    null = np.load(out / "null.npz")
    report = {"dataset": ds["name"], "mask_fraction": float(ds["good"].mean()), "sizes_deg": SIZES, "templates": list(TEMPLATES),
              "thresholds": G["thr"].tolist(), "thresholds_source": G["thr_source"], "lmax": ds["lmax"],
              "n_null": int(null["max_s"].shape[0]), "maps": {},
              "null_potential_per_size": null["n_pot"].mean(axis=0).tolist(),
              "null_candidates": {t: {"mean": float(null["n_candidates"][:, ti].mean()),
                                      "percentiles_5_50_95": np.percentile(null["n_candidates"][:, ti], [5, 50, 95]).tolist()}
                                  for ti, t in enumerate(TEMPLATES)}}
    for mname, (path, field, scale) in ds["maps"].items():
        raw = hp.ud_grade(hp.read_map(str(path), field=field, dtype=np.float64), NSIDE) * scale
        odd = ~np.isfinite(raw) | (np.abs(raw) > 1.0)
        raw[odd] = 0.0
        w_all = np.empty((NF, NPIX), dtype=np.float32)
        for f, w in filtered(prepare(raw)):
            w_all[f] = w
        np.save(out / f"filtered_{mname}.npy", w_all)
        kept, dropped, max_s, n_pot = run_algorithm(w_all)
        res = {"odd_pixels_unmasked": int(np.count_nonzero(odd & ds["good"])), "candidates": {}, "dropped": {}, "n_candidates": {},
               "p_n_candidates": {}, "max_s": max_s.tolist(), "n_potential": n_pot.tolist(),
               "p_max_s_per_filter": [rank_p(null["max_s"][:, f], max_s[f]) for f in range(NF)], "largest": {}, "mcewen": []}
        for ti, tname in enumerate(TEMPLATES):
            mine = sorted((describe(d) for d in kept if d["template"] == tname), key=lambda c: -c["S"])
            res["candidates"][tname] = mine
            res["dropped"][tname] = sorted((describe(d) for d in dropped if d["template"] == tname), key=lambda c: -c["S"])
            res["n_candidates"][tname] = len(mine)
            res["p_n_candidates"][tname] = rank_p(null["n_candidates"][:, ti], len(mine))
            sl = slice(ti * NS, (ti + 1) * NS)
            res["largest"][tname] = {"max_s": float(max_s[sl].max()), "size_deg": SIZES[int(np.argmax(max_s[sl]))],
                                     "p": rank_p(null["max_s"][:, sl].max(axis=1), float(max_s[sl].max()))}
            log(f"{mname} / {tname}: {len(mine)} candidates (simulated skies: mean {null['n_candidates'][:, ti].mean():.2f}, "
                f"p = {res['p_n_candidates'][tname]:.3f}); largest S {res['largest'][tname]['max_s']:.2f} at "
                f"{res['largest'][tname]['size_deg']:g} deg (p = {res['largest'][tname]['p']:.3f})")
            for c in mine:
                log("    %5.1f deg  (l, b) = (%6.1f, %+5.1f)  S = %.2f  z0 = %+.3f mK  disc unmasked %.2f"
                    % (c["size_deg"], c["lon"], c["lat"], c["S"], c["z0_mK"], c["disc_unmasked_fraction"]))
        # the published candidates of McEwen et al. (2012): our significance at their position and size
        for label, z0_mk, colat, lon, size in MCEWEN_2012:
            lat = 90.0 - colat
            si = SIZES.index(float(size))
            v = hp.ang2vec(lon, lat, lonlat=True)
            p0 = int(hp.vec2pix(NSIDE, *v))
            near = hp.query_disc(NSIDE, v, math.radians(max(1.5, 0.25 * size)))
            near = near[ds["good"][near]]
            row = {"label": label, "z0_mK": z0_mk, "lon": lon, "lat": lat, "size_deg": size, "masked_here": bool(not ds["good"][p0])}
            for ti, tname in enumerate(TEMPLATES):
                f = ti * NS + si
                S = np.abs(w_all[f].astype(np.float64)) / G["sigma"][f]
                best = int(near[np.argmax(S[near])]) if near.size else p0
                tol = max(3.0, 0.25 * size)
                match = [c for c in res["candidates"][tname] if abs(c["size_deg"] - size) <= THETA_ADJ and c["z0_mK"] * z0_mk > 0
                         and np.degrees(np.arccos(np.clip(hp.ang2vec(c["lon"], c["lat"], lonlat=True) @ v, -1, 1))) <= tol]
                row[tname] = {"S_at_position": float(S[p0]), "S_nearby_max": float(S[best]),
                              "z0_mK_at_position": float(w_all[f][p0] / math.sqrt(G["bank"]["norm"][f]) * bt.T_CMB_K * 1e3
                                                         * G["bank"]["z0_factor"][f]),
                              "threshold": float(G["thr"][f]), "recovered": bool(match),
                              "matched_candidate": match[0] if match else None}
            res["mcewen"].append(row)
        log(f"{mname}: McEwen et al. candidates recovered (cosine template): "
            f"{sum(r['cosine']['recovered'] for r in res['mcewen'])} of {len(res['mcewen'])}; "
            f"(transfer template): {sum(r['transfer']['recovered'] for r in res['mcewen'])}")
        for r in res["mcewen"]:
            log("    no. %2d  %5.1f deg  S here %.2f (nearby max %.2f, threshold %.2f)  z0 here %+.3f mK (published %+.2f)  %s%s"
                % (r["label"], r["size_deg"], r["cosine"]["S_at_position"], r["cosine"]["S_nearby_max"], r["cosine"]["threshold"],
                   r["cosine"]["z0_mK_at_position"], r["z0_mK"], "recovered" if r["cosine"]["recovered"] else "not recovered",
                   " (position masked here)" if r["masked_here"] else ""))
        report["maps"][mname] = res
    with open(out / "fullsky_filter_search_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    log(f"wrote {out / 'fullsky_filter_search_report.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("planck", "planck767", "wmap7w"))
    ap.add_argument("--thresholds", default="published", choices=("published", "calibrated"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--stage", default="all", help="comma list of selftest, stats, calib, null, data; or all")
    ap.add_argument("--maps", default="", help="comma list of map names to search (default: all of the data set)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--n-stats", type=int, default=3000)
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261122)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stages = ["selftest", "stats", "calib", "null", "data"] if args.stage == "all" else args.stage.split(",")
    if args.thresholds == "published":
        stages = [s for s in stages if s != "calib"]

    G["ds"] = load_dataset(args.dataset, Path(args.data_dir), args.seed)
    log(f"{args.dataset}: mask keeps {G['ds']['good'].mean():.4f} of the sky; lmax {G['ds']['lmax']}; "
        f"noise power / CMB power at l = 200: {G['ds']['noise_power'] / (G['ds']['cl'][200] * G['ds']['window'][200] ** 2):.3f}")
    G["bank"] = build_bank(G["ds"])
    log("filters ready; ideal SNR per 1e-5 of amplitude, cosine: " + " ".join(
        "%g:%.2f" % (SIZES[si], 1e-5 * math.sqrt(G["bank"]["norm"][si])) for si in range(0, NS, 4))
        + " | transfer: " + " ".join("%g:%.2f" % (SIZES[si], 1e-5 * math.sqrt(G["bank"]["norm"][NS + si])) for si in range(0, NS, 4)))
    G["nb"] = hp.get_all_neighbours(NSIDE, np.arange(NPIX)).astype(np.int32)
    G["lookup"] = np.full(NPIX, -1, dtype=np.int32)

    if "selftest" in stages:
        stage_selftest(out)
    if "stats" in stages:
        stage_stats(out, args.n_stats, args.workers)
    if any(s in stages for s in ("calib", "null", "data")):
        G["sigma"] = np.load(out / "sigma.npy").astype(np.float64)
        G["sigma"][:, ~G["ds"]["good"]] = np.inf
    if "calib" in stages:
        stage_calib(out, args.n_calib, args.workers)
    if any(s in stages for s in ("null", "data")):
        if args.thresholds == "published":
            G["thr"] = np.array([published_threshold(size) for _ in TEMPLATES for size in SIZES])
        else:
            G["thr"] = np.array(json.load(open(out / "thresholds.json"))["thresholds"])
        G["thr_source"] = args.thresholds
    if args.maps:
        G["ds"]["maps"] = {k: v for k, v in G["ds"]["maps"].items() if k in args.maps.split(",")}
    if "null" in stages:
        stage_null(out, args.n_null, args.workers)
    if "data" in stages:
        stage_data(out)
    log("done")


if __name__ == "__main__":
    main()
