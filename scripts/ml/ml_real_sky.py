"""Real-sky test of the network: training with the Planck mask, a scan of the four Planck maps and of
simulated skies prepared the same way, with the published WMAP candidates as named test positions.

Pre-registration: ``work/notebook_ml_ceiling.md`` entry (M), written and pushed before any run of this
script other than its self-test and smoke checks.

Assumptions
-----------
* Everything not listed here is as in ``ml_ceiling_study.py``: Gaussian CAMB skies at Nside 256 with a
  5 arcmin beam, the pixel window and 30 uK-arcmin white noise; 256 x 256 patches at 13 arcmin; signal
  centre within 5 degrees of the patch centre; random sign; strength log-uniform in ideal matched-filter
  SNR between 1 and 12; the training recipe of arm N (random initialisation, 30 epochs, batch 64, AdamW
  3e-4 with a cosine schedule, rotations and flips, last epoch used).
* Sizes 2 to 25 degrees in steps of 1 (the ceiling study used 5 to 25): two of the four strongest published
  spots are 3 and 4 degrees across.
* Two templates, one network each:
    T  the curvature ramp through the CAMB transfer functions (``collision_template_transfer``), the
       physical template of this project, as arm N of the ceiling study;
    C  the cosine cap with z_crit = 0, Delta T / T = z0 (cos t - cos t_c) / (1 - cos t_c) inside t_c, the
       template the text of McEwen et al. (2012) describes (coefficients as
       ``fullsky_filter_search.cosine_coefficients``).
  Both get the beam and pixel window, lose their monopole and dipole, and are scaled to unit observed peak,
  exactly as ``World.template``.
* Mask: Planck PR3 common intensity mask, ``ud_grade`` to Nside 256, a pixel kept where the value is at
  least 0.9 (the project's convention).
* Preparation, identical for training skies, simulated skies and Planck maps: monopole and dipole fitted by
  least squares on the kept pixels and removed from the whole map, then masked pixels set to zero. The
  network input has two channels: the prepared patch divided by the input scale, and the patch of the mask
  (1 kept, 0 masked). Patches are read by nearest-pixel lookup, exactly as ``ml_ceiling_study.project``.
* Training patches: centres drawn uniformly over the kept pixels, 8 per sky, at least 45 degrees apart.
  Signals are added to the patch (as in the ceiling study) and multiplied by the patch's mask. Only in
  training, the change a signal would make to the fitted monopole and dipole is left out.
* Planck maps: PR3 component-separated temperature maps (SMICA, NILC, SEVEM, Commander), field 0,
  Nside 2048, ``ud_grade`` to 256, Kelvin.
* Tiles: HEALPix Nside 16 pixel centres whose Nside 256 pixel is kept.
* Tile scores: the image-head logit (primary) and the largest mask logit, each standardised per tile by
  its mean and standard deviation over the simulated skies.
* Spot score at a sky position: the largest standardised score over the tiles whose centres lie within
  5 degrees of it (the prior's largest offset). Its p value is its rank among the simulated skies' spot
  scores at the same position, (1 + number at least as large) / (1 + number of simulated skies). A
  position is "caught" when p < 0.01.
* Injections in the scan are made on the sphere: the template is evaluated at every pixel, prepared as
  above, and added to the prepared map's tiles near the position (exact, because the preparation is
  linear); only those tiles are scored again.

Modes
-----
    python scripts/ml/ml_real_sky.py selftest --data-dir DATA
    python scripts/ml/ml_real_sky.py train --out-dir RUN --data-dir DATA --template T|C --seed 1 [--test-dir T0]
    python scripts/ml/ml_real_sky.py scan --out-dir SCAN --data-dir DATA --model T:ckpt --model C:ckpt

``--smoke`` shrinks train and scan to a few minutes.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "followup"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ml_ceiling_study as cs  # noqa: E402
import patch_geometry as pg  # noqa: E402
from recompute_candidate_positions import MCEWEN_2012  # noqa: E402  (no, z0 in mK, colatitude, longitude, size)

NSIDE = cs.NSIDE
NPIX = 12 * NSIDE * NSIDE
LMAX = cs.LMAX
P = cs.P
MAP_FILES = {
    "smica": "COM_CMB_IQU-smica_2048_R3.00_full.fits",
    "nilc": "COM_CMB_IQU-nilc_2048_R3.00_full.fits",
    "sevem": "COM_CMB_IQU-sevem_2048_R3.00_full.fits",
    "commander": "COM_CMB_IQU-commander_2048_R3.00_full.fits",
}
MASK_FILE = "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
SIZES_DEFAULT = ",".join(str(x) for x in range(2, 26))
SPOT_RADIUS_DEG = 5.0
CAUGHT_P = 0.01
SCORES = ("image_head", "mask_max")

# Feeney et al. (2011, arXiv:1012.3667), Table VIII: feature -> (colatitude, longitude, theta_crit, sign of z0,
# ln of the local evidence ratio). Feature 1 has no fit (its size range lies beyond their 11 degree patches);
# its position is the second blob of Table VII. Their text singles out 2, 3, 7 and 10, whose evidence ratios
# are above those of their end-to-end simulation; feature 2 is the Cold Spot.
FEENEY_2011 = {
    1: (140.9, 174.4, None, +1, None), 2: (147.3, 208.0, 6.4, -1, -4.6), 3: (123.0, 320.8, 6.5, +1, -4.1),
    4: (145.3, 32.8, 3.2, -1, -7.3), 5: (32.6, 74.3, 6.9, +1, -9.2), 6: (128.6, 91.8, 7.9, +1, -9.7),
    7: (169.0, 185.7, 4.6, -1, -5.4), 8: (58.0, 115.7, 4.4, +1, -8.6), 9: (152.1, 241.9, 2.3, -1, -9.0),
    10: (167.2, 269.1, 4.2, -1, -3.8), 11: (114.9, 22.4, 2.5, -1, -8.1),
}
FEENEY_TOP = (2, 3, 7, 10)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def published_spots():
    spots = []
    for no, z0, colat, lon, size in MCEWEN_2012:
        spots.append({"label": f"M{no}", "lon": float(lon), "lat": 90.0 - float(colat), "size_deg": float(size),
                      "sign": 1 if z0 > 0 else -1, "source": "McEwen et al. 2012, Table I"})
    for k, (colat, lon, th, sg, lnb) in FEENEY_2011.items():
        spots.append({"label": f"F{k}", "lon": float(lon), "lat": 90.0 - float(colat), "size_deg": th, "sign": sg,
                      "ln_evidence_ratio": lnb, "top_four": k in FEENEY_TOP, "source": "Feeney et al. 2011, Table VIII"})
    return spots


# --------------------------------------------------------------------------------------------------------
# templates
# --------------------------------------------------------------------------------------------------------
def cosine_coefficients(theta_c, lmax):
    """t_l = 2 pi * integral_{mu_c}^{1} (mu - mu_c) / (1 - mu_c) P_l(mu) dmu (copied from
    ``fullsky_filter_search.py`` on main; Gauss-Legendre on [mu_c, 1] is exact for this polynomial)."""
    mu_c = math.cos(theta_c)
    x, w = np.polynomial.legendre.leggauss(lmax // 2 + 8)
    mu = 0.5 * (1.0 - mu_c) * x + 0.5 * (1.0 + mu_c)
    f = (mu - mu_c) / (1.0 - mu_c)
    return 2.0 * math.pi * ((0.5 * (1.0 - mu_c) * w * f) @ np.polynomial.legendre.legvander(mu, lmax))


def finish_template(world, b, size, kind):
    """The body of ``World.template`` for raw harmonic coefficients ``b``: beam and pixel window, monopole and
    dipole removed, unit observed peak."""
    ells = world.ells
    coeff = (2 * ells + 1) / (4 * math.pi) * b * world.window
    coeff[:2] = 0.0
    prof = np.polynomial.legendre.legval(np.cos(world.fine), coeff)
    peak = int(np.argmax(np.abs(prof)))
    scale = 1.0 / prof[peak]
    prof = prof * scale
    t_l = b * world.window * scale * cs.T_CMB_K
    t_l[:2] = 0.0
    norm = float(np.sum((2 * ells[2:] + 1) / (4 * math.pi) * t_l[2:] ** 2 / world.total_power[2:]))
    below = np.nonzero((np.abs(prof) < 0.1) & (np.arange(cs.N_FINE) > peak))[0]
    return {"size": float(size), "kind": kind, "t_l": t_l, "profile_k": prof * cs.T_CMB_K, "norm": norm,
            "mask_radius": float(world.fine[below[0]]) if below.size else math.radians(size),
            "peak_angle_deg": float(np.degrees(world.fine[peak]))}


def make_template(world, kind, size):
    if kind == "T":
        t = dict(world.template(size, 0.0))
        t["kind"] = "T"
        return t
    if kind == "C":
        return finish_template(world, cosine_coefficients(math.radians(size), LMAX), size, "C")
    raise ValueError(kind)


# --------------------------------------------------------------------------------------------------------
# mask, preparation, tiles
# --------------------------------------------------------------------------------------------------------
IDX_MAP = np.arange(NPIX, dtype=np.float64)


def patch_index(lon, lat):
    """HEALPix pixel of every patch pixel; ``sky[patch_index(lon, lat)]`` equals ``cs.project(sky, lon, lat)``."""
    return cs.project(IDX_MAP, lon, lat).astype(np.int64)


class SkyPrep:
    """The mask, the monopole and dipole fit on the kept pixels, and the scan tiles."""

    def __init__(self, mask_path, tile_nside=16, min_mask=0.9, with_tiles=True):
        import healpy as hp

        raw = hp.read_map(str(mask_path), field=0, dtype=np.float64)
        self.good = hp.ud_grade(raw, NSIDE) >= min_mask
        self.pix_vec = np.array(hp.pix2vec(NSIDE, np.arange(NPIX)))                 # (3, NPIX)
        self.design = np.vstack([np.ones(NPIX), self.pix_vec])                     # (4, NPIX)
        dg = self.design[:, self.good]
        self.fit = np.linalg.solve(dg @ dg.T, dg)                                   # coefficients = fit @ m[good]
        if not with_tiles:
            return
        lon, lat = hp.pix2ang(tile_nside, np.arange(12 * tile_nside * tile_nside), lonlat=True)
        keep = self.good[hp.ang2pix(NSIDE, lon, lat, lonlat=True)]
        self.tiles = np.stack([lon[keep], lat[keep]], axis=1)
        self.tile_vec = np.asarray(hp.ang2vec(self.tiles[:, 0], self.tiles[:, 1], lonlat=True))   # (n, 3)
        t0 = time.time()
        self.tile_idx = np.stack([patch_index(a, b) for a, b in self.tiles]).astype(np.int32)  # (n, P, P)
        self.tile_mask = self.good[self.tile_idx]                                   # bool (n, P, P)
        self.n_tiles = len(self.tiles)
        log(f"{self.n_tiles} tiles (Nside {tile_nside} centres on kept pixels); index tables in {time.time() - t0:.0f} s")

    def prepare(self, m):
        coef = self.fit @ m[self.good]
        out = m - coef @ self.design
        out[~self.good] = 0.0
        return out

    def near(self, vec, radius_deg=SPOT_RADIUS_DEG):
        return np.nonzero(self.tile_vec @ np.asarray(vec, dtype=np.float64) >= math.cos(math.radians(radius_deg)))[0]


def masked_positions(rng, good, n, min_sep_deg, max_tries=20000):
    """Up to ``n`` directions on kept pixels, at least ``min_sep_deg`` apart; (lon, lat) in degrees."""
    import healpy as hp

    cmin = math.cos(math.radians(min_sep_deg))
    pts, tries = [], 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        if not good[hp.vec2pix(NSIDE, v[0], v[1], v[2])]:
            continue
        if all(float(v @ q) <= cmin for q in pts):
            pts.append(v)
    out = []
    for v in pts:
        lon, lat = hp.vec2ang(v, lonlat=True)
        out.append((float(np.ravel(lon)[0]), float(np.ravel(lat)[0])))
    return out


def masked_backgrounds(world, prep, n_skies, per_sky, rng, label, min_sep_deg=45.0):
    """Signal-free prepared patches and their mask patches."""
    xs = np.zeros((n_skies * per_sky, P, P), dtype=np.float32)
    ms = np.zeros((n_skies * per_sky, P, P), dtype=np.uint8)
    row, t0 = 0, time.time()
    for isky in range(n_skies):
        sky = prep.prepare(world.sky(rng))
        for lon, lat in masked_positions(rng, prep.good, per_sky, min_sep_deg):
            idx = patch_index(lon, lat)
            xs[row] = sky[idx]
            ms[row] = prep.good[idx]
            row += 1
        if (isky + 1) % 250 == 0 or isky == n_skies - 1:
            log(f"  {label}: skies {isky + 1}/{n_skies}, {row} patches, {time.time() - t0:.0f} s")
    return xs[:row], ms[:row]


# --------------------------------------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------------------------------------
class TemplateInjector(cs.Injector):
    """``ml_ceiling_study.Injector`` for an explicit list of templates (its ``__call__`` is reused unchanged)."""

    def __init__(self, templates, max_offset_deg, device):  # noqa: D107 (the parent's __init__ builds from World)
        import torch

        self.torch = torch
        self.profiles = torch.tensor(np.stack([t["profile_k"] for t in templates]), dtype=torch.float32, device=device)
        self.mask_radius = torch.tensor([t["mask_radius"] for t in templates], dtype=torch.float32, device=device)
        self.root_norm = torch.tensor([math.sqrt(t["norm"]) for t in templates], dtype=torch.float32, device=device)
        c0 = (P - 1) / 2.0
        self.c0 = c0
        self.dx = math.radians(pg.RESO_ARCMIN / 60.0)
        iy, ix = torch.meshgrid(torch.arange(P, device=device, dtype=torch.float32), torch.arange(P, device=device, dtype=torch.float32), indexing="ij")
        v = torch.stack([-(ix - c0) * self.dx, (iy - c0) * self.dx, torch.ones_like(ix)], dim=0)
        self.v = (v / v.norm(dim=0, keepdim=True)).reshape(3, -1)
        self.max_off_px = max_offset_deg * 60.0 / pg.RESO_ARCMIN
        self.step = math.pi / (cs.N_FINE - 1)
        self.device = device


def build_model(init, device):
    import segmentation_models_pytorch as smp

    weights = "imagenet" if init == "imagenet" else None
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=weights, in_channels=2, classes=1,
                     aux_params={"classes": 1, "dropout": 0.2})
    return model.to(device)


def score_patches(model, patches, masks, std, device, batch=256):
    """Largest mask logit and image logit per patch; ``masks`` None means fully kept patches."""
    import torch

    model.eval()
    n = len(patches)
    seg = np.zeros(n, dtype=np.float64)
    img = np.zeros(n, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, n, batch):
            x = torch.from_numpy(np.array(patches[i:i + batch], dtype=np.float32)).to(device) / std
            if masks is None:
                m = torch.ones_like(x)
            else:
                m = torch.from_numpy(np.array(masks[i:i + batch], dtype=np.float32)).to(device)
            mask_logits, image_logits = model(torch.stack([x, m], dim=1))
            seg[i:i + batch] = mask_logits.flatten(1).max(dim=1).values.double().cpu().numpy()
            img[i:i + batch] = image_logits.reshape(-1).double().cpu().numpy()
    return seg, img


# --------------------------------------------------------------------------------------------------------
# self-test (no GPU)
# --------------------------------------------------------------------------------------------------------
def cmd_selftest(args):
    import healpy as hp

    rep = {}
    prep = SkyPrep(Path(args.data_dir) / MASK_FILE, with_tiles=False)
    rep["mask_kept_fraction"] = float(prep.good.mean())
    log(f"mask keeps {rep['mask_kept_fraction']:.4f} of the sky")
    world = cs.World()
    rng = np.random.default_rng(7)
    sky = world.sky(rng)

    # 1. the fast monopole and dipole removal equals healpy's on the kept pixels
    w = sky.copy()
    w[~prep.good] = hp.UNSEEN
    ref = hp.remove_dipole(w, bad=hp.UNSEEN)
    mine = prep.prepare(sky)
    d = float(np.max(np.abs(ref[prep.good] - mine[prep.good])) / np.std(sky))
    rep["dipole_fit_max_rel_diff"] = d
    assert d < 1e-8, d
    assert np.all(mine[~prep.good] == 0.0)

    # 2. index lookup equals the projection of the ceiling study, pixel for pixel
    for lon, lat in [(0.0, 60.0), (208.0, -57.3), (321.3, -33.2), (100.0, 10.0)]:
        a = mine[patch_index(lon, lat)]
        b = cs.project(mine, lon, lat)
        assert np.array_equal(a, b), (lon, lat)
    rep["index_lookup_equals_projection"] = True

    # 3. cosine coefficients against a plain quadrature, and the beam-free profile inside the cap
    for size in (3.0, 10.0, 25.0):
        tc = math.radians(size)
        t_l = cosine_coefficients(tc, LMAX)
        mu = np.linspace(math.cos(tc), 1.0, 200001)
        f = (mu - math.cos(tc)) / (1.0 - math.cos(tc))
        for ell in (0, 1, 2, 10, 100, 500):
            pl = np.polynomial.legendre.legval(mu, np.eye(ell + 1)[ell])
            direct = 2.0 * math.pi * np.trapezoid(f * pl, mu) if hasattr(np, "trapezoid") else 2.0 * math.pi * np.trapz(f * pl, mu)
            assert abs(direct - t_l[ell]) <= 1e-6 * max(abs(t_l[0]), 1e-30), (size, ell, direct, t_l[ell])
        t = make_template(world, "C", size)
        rep[f"cosine_{size:g}_mask_radius_deg"] = math.degrees(t["mask_radius"])
        rep[f"cosine_{size:g}_peak_angle_deg"] = t["peak_angle_deg"]
        assert t["peak_angle_deg"] < 0.2 * size, t["peak_angle_deg"]
    rep["cosine_coefficients_ok"] = True

    # 4. linearity: preparing sky + signal equals preparing each and adding
    t = make_template(world, "T", 10.0)
    v = np.asarray(hp.ang2vec(208.0, -57.3, lonlat=True))
    sig = 3e-5 * np.interp(np.arccos(np.clip(v @ prep.pix_vec, -1, 1)), world.fine, t["profile_k"])
    both = prep.prepare(sky + sig)
    split = prep.prepare(sky) + prep.prepare(sig)
    d = float(np.max(np.abs(both - split)) / np.std(sky))
    rep["linearity_max_rel_diff"] = d
    assert d < 1e-10, d
    out = Path(args.out_json) if args.out_json else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1))
    log("self-test passed: " + json.dumps(rep))


# --------------------------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------------------------
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
    if args.smoke:
        sizes, args.train_skies, args.val_skies, args.epochs = [3.0, 10.0, 25.0], 12, 4, 1
    world = cs.World()
    prep = SkyPrep(Path(args.data_dir) / MASK_FILE, with_tiles=False)
    templates = [make_template(world, args.template, s) for s in sizes]
    inject = TemplateInjector(templates, args.max_offset_deg, device)
    log(f"template {args.template}: {len(sizes)} sizes {sizes[0]:g} to {sizes[-1]:g} deg; init {args.init}; seed {args.seed}; "
        f"mask keeps {prep.good.mean():.4f}")

    xtr, mtr = masked_backgrounds(world, prep, args.train_skies, args.positions_per_sky, np.random.default_rng(5_000_000 + args.seed), "train")
    xva, mva = masked_backgrounds(world, prep, args.val_skies, args.positions_per_sky, np.random.default_rng(6_000_000 + args.seed), "validation")
    k = min(len(xtr), 2000)
    std = float(np.std(xtr[:k][mtr[:k].astype(bool)]))
    log(f"input scale {std:.3e} K (kept pixels); {len(xtr)} training and {len(xva)} validation backgrounds; "
        f"kept fraction of training patches {float(mtr[:k].mean()):.3f}")

    vgen = torch.Generator(device=device)
    vgen.manual_seed(777 + args.seed)
    xv, yv, av = [], [], []
    for i in range(0, len(xva), 256):
        mb = torch.from_numpy(mva[i:i + 256].astype(np.float32)).to(device)
        xb, _, yb, ab = inject(torch.from_numpy(xva[i:i + 256]).to(device), vgen)
        xv.append((xb * mb).cpu().numpy())
        yv.append(yb.cpu().numpy())
        av.append(ab.cpu().numpy())
    xv, yv, av = np.concatenate(xv), np.concatenate(yv), np.concatenate(av)
    mid = (yv == 1) & (av >= 2.0) & (av <= 5.0)

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
            mb = torch.from_numpy(mtr[idx].astype(np.float32)).to(device)
            x, m, y, _ = inject(torch.from_numpy(xtr[idx]).to(device), gen)
            x = torch.stack([(x * mb) / std, mb], dim=1)
            m = m[:, None]
            kk = int(rng.integers(0, 4))
            if kk:
                x, m = torch.rot90(x, kk, dims=(2, 3)), torch.rot90(m, kk, dims=(2, 3))
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
            if epoch == 1 and nb == 50:
                log(f"  first 50 steps: {time.time() - t0:.0f} s ({steps_per_epoch} steps per epoch, {args.epochs} epochs)")
        segv, imgv = score_patches(model, xv, mva, std, device, batch=args.eval_batch)
        rec = {"epoch": epoch, "train_loss": tot / nb, "seconds": time.time() - t0}
        for name, s in (("mask_max", segv), ("image_head", imgv)):
            thr = cs.null_threshold(s[yv == 0], 0.05)
            rec[f"val_detect_mid_{name}"] = float(np.mean(s[mid] > thr)) if mid.any() else float("nan")
        history.append(rec)
        log(f"epoch {epoch:2d}  loss {rec['train_loss']:.4f}  validation detection (ideal SNR 2-5) at 5%: "
            f"mask max {rec['val_detect_mid_mask_max']:.3f}, image head {rec['val_detect_mid_image_head']:.3f}  ({rec['seconds']:.0f} s)")

    args_record = {k2: v for k2, v in vars(args).items() if k2 != "func"}
    args_record["sizes_used"] = sizes
    torch.save({"model_state_dict": model.state_dict(), "input_std": std, "args": args_record, "in_channels": 2}, out / "checkpoint_last.pt")
    report = {"args": args_record, "history": history, "input_std": std, "tests": {}}
    if device.type == "cuda":
        report["max_gpu_memory_gib_training"] = torch.cuda.max_memory_allocated() / 2 ** 30
        torch.cuda.empty_cache()
    for test_dir in args.test_dir:   # unmasked test sets of the ceiling study, scored with a fully kept mask channel
        td = Path(test_dir)
        cfg = json.loads((td / "test_config.json").read_text())
        meta = dict(np.load(td / "meta.npz"))
        patches = np.load(td / "patches.npy", mmap_mode="r")
        seg, img = score_patches(model, patches, None, std, device, batch=args.eval_batch)
        scores = {"labels": meta["labels"], "level": meta["level"], "size_deg": meta["size_deg"],
                  "net__mask_max": seg, "net__image_head": img}
        scores.update({k2: v for k2, v in meta.items() if k2.startswith("ref")})
        table = cs.detection_table(scores, cfg["levels"], cfg["test_sizes_deg"], [0.05, 0.01])
        report["tests"][td.name] = {"mix": cfg["mix"], "level_kind": cfg["level_kind"], "table": table}
        np.savez_compressed(out / f"scores__{td.name}.npz", mask_max=seg, image_head=img)
        print(f"\ntest set {td.name} (mix {cfg['mix']:g}, levels are {cfg['level_kind']}), mask channel all kept")
        cs.print_table(table, cfg["levels"], 0.05)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    log(f"wrote {out}")


# --------------------------------------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------------------------------------
def rank_p(null_values, value):
    """(1 + number of simulated values at least as large) / (1 + number of simulated values)."""
    null_values = np.asarray(null_values)
    return float((1 + np.sum(null_values >= value)) / (1 + null_values.size))


def cmd_scan(args):
    import healpy as hp
    import torch

    t_start = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    data = Path(args.data_dir)
    maps = [m for m in args.maps.split(",") if m]
    n_null = args.n_null
    inj_sizes = [float(x) for x in args.inject_sizes.split(",")]
    inj_snrs = [float(x) for x in args.inject_snr.split(",")]
    n_inj_pos = args.inject_positions
    n_null_inj = min(args.n_null_inject, n_null)
    if args.smoke:
        n_null, n_inj_pos, n_null_inj, inj_sizes, inj_snrs = 6, 3, 2, [5.0], [6.0]
        args.n_random = 20

    prep = SkyPrep(data / MASK_FILE, tile_nside=args.tile_nside)
    world = cs.World()
    models = []
    for spec in args.model:
        name, path = spec.split(":", 1)
        ck = torch.load(path, map_location=device, weights_only=False)
        model = build_model("random", device)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        models.append({"name": name, "model": model, "std": float(ck["input_std"]), "kind": ck["args"]["template"],
                       "sizes": ck["args"].get("sizes_used"), "checkpoint": path})
        log(f"model {name}: template {ck['args']['template']}, input scale {float(ck['input_std']):.3e} K")

    # positions: the published spots, then random kept positions (the first ``n_inj_pos`` also carry injections)
    spots = published_spots()
    rng = np.random.default_rng(args.seed)
    rand = masked_positions(rng, prep.good, args.n_random, 0.0)
    pos = [(s["lon"], s["lat"]) for s in spots] + rand
    pvec = np.asarray(hp.ang2vec(np.array([p[0] for p in pos]), np.array([p[1] for p in pos]), lonlat=True))
    near = [prep.near(v) for v in pvec]
    n_pos, n_spots = len(pos), len(spots)
    log(f"{n_spots} published positions, {len(rand)} random positions; tiles within {SPOT_RADIUS_DEG:g} deg: "
        f"median {int(np.median([len(x) for x in near]))}")

    injections = []
    inj_positions = [e for e in range(n_spots, n_pos) if len(near[e])][:n_inj_pos]
    for e in inj_positions:
        for size in inj_sizes:
            for snr in inj_snrs:
                injections.append({"pos": e, "size_deg": size, "snr": snr, "sign": int(rng.choice([-1, 1]))})
    inj_tiles = np.concatenate([near[q["pos"]] for q in injections]) if injections else np.zeros(0, int)
    inj_off = np.cumsum([0] + [len(near[q["pos"]]) for q in injections])
    for m in models:
        if not injections:
            m["inj_sig"] = np.zeros((0, P, P), np.float32)
            continue
        tmpl = {s: make_template(world, m["kind"], s) for s in inj_sizes}
        sig = np.zeros((len(inj_tiles), P, P), dtype=np.float32)
        for i, q in enumerate(injections):
            t = tmpl[q["size_deg"]]
            amp = q["snr"] / math.sqrt(t["norm"])
            ang = np.arccos(np.clip(pvec[q["pos"]] @ prep.pix_vec, -1.0, 1.0))
            smap = prep.prepare(q["sign"] * amp * np.interp(ang, world.fine, t["profile_k"]))
            sig[inj_off[i]:inj_off[i + 1]] = smap[prep.tile_idx[near[q["pos"]]]]
        m["inj_sig"] = sig
    log(f"{len(injections)} injections per model, {len(inj_tiles)} tile patches each")

    def score_tiles(m, patches, tiles=None):
        masks = prep.tile_mask if tiles is None else prep.tile_mask[tiles]
        seg, img = score_patches(m["model"], patches, masks, m["std"], device, batch=args.eval_batch)
        return {"image_head": img.astype(np.float32), "mask_max": seg.astype(np.float32)}

    def score_injections(m, patches):
        res = {s: np.zeros(len(inj_tiles), np.float32) for s in SCORES}
        for c in range(0, len(inj_tiles), 2048):
            sl = slice(c, c + 2048)
            x = patches[inj_tiles[sl]] + m["inj_sig"][sl]
            r = score_tiles(m, x, inj_tiles[sl])
            for s in SCORES:
                res[s][sl] = r[s]
        return res

    # simulated skies
    null = {m["name"]: {s: np.zeros((n_null, prep.n_tiles), np.float32) for s in SCORES} for m in models}
    null_inj = {m["name"]: {s: np.zeros((n_null_inj, len(inj_tiles)), np.float32) for s in SCORES} for m in models}
    t0 = time.time()
    for j in range(n_null):
        sky = prep.prepare(world.sky(np.random.default_rng([args.seed, 7, j]))).astype(np.float32)
        patches = sky[prep.tile_idx]
        for m in models:
            r = score_tiles(m, patches)
            for s in SCORES:
                null[m["name"]][s][j] = r[s]
            if j < n_null_inj:
                ri = score_injections(m, patches)
                for s in SCORES:
                    null_inj[m["name"]][s][j] = ri[s]
        if (j + 1) % 25 == 0 or j == n_null - 1:
            log(f"  simulated skies {j + 1}/{n_null}  {time.time() - t0:.0f} s")

    # Planck maps
    real, real_inj = {}, {}
    for mname in maps:
        t1 = time.time()
        raw = hp.ud_grade(hp.read_map(str(data / MAP_FILES[mname]), field=0, dtype=np.float64), NSIDE)
        sky = prep.prepare(raw).astype(np.float32)
        patches = sky[prep.tile_idx]
        real[mname], real_inj[mname] = {}, {}
        for m in models:
            real[mname][m["name"]] = score_tiles(m, patches)
            real_inj[mname][m["name"]] = score_injections(m, patches)
        log(f"  map {mname}: {time.time() - t1:.0f} s")

    # statistics
    summary = {"config": {"n_tiles": prep.n_tiles, "tile_nside": args.tile_nside, "mask_kept_fraction": float(prep.good.mean()),
                          "n_null": n_null, "n_null_inject": n_null_inj, "maps": maps, "spot_radius_deg": SPOT_RADIUS_DEG,
                          "caught_p": CAUGHT_P, "n_random_positions": len(rand), "inject_positions": len(inj_positions),
                          "inject_sizes_deg": inj_sizes, "inject_snr": inj_snrs, "seed": args.seed,
                          "models": [{k: m[k] for k in ("name", "kind", "std", "sizes", "checkpoint")} for m in models]},
               "spots": spots, "models": {}}
    arrays = {}
    for m in models:
        name = m["name"]
        msum = {}
        for s in SCORES:
            N = null[name][s].astype(np.float64)
            mu, sd = N.mean(axis=0), N.std(axis=0, ddof=1)
            sd = np.where(sd > 0, sd, 1.0)
            Z = (N - mu) / sd
            spot_null = np.full((n_null, n_pos), np.nan)
            for e in range(n_pos):
                if len(near[e]):
                    spot_null[:, e] = Z[:, near[e]].max(axis=1)
            # per-tile p of every simulated sky against the others, for the count statistic
            srt = np.sort(N, axis=0)
            ge = np.empty_like(N)
            for t in range(prep.n_tiles):
                ge[:, t] = n_null - np.searchsorted(srt[:, t], N[:, t], side="left")
            k_null = np.sum(ge / n_null < CAUGHT_P, axis=1)
            m_null = Z.max(axis=1)
            rec = {"null_max_z_quantiles_5_50_95": [float(x) for x in np.percentile(m_null, [5, 50, 95])],
                   "null_count_quantiles_5_50_95": [float(x) for x in np.percentile(k_null, [5, 50, 95])],
                   "null_count_mean": float(k_null.mean()), "maps": {}}
            # injections on simulated skies (detected when the spot p value is below CAUGHT_P)
            def inj_table(raw_scores_rows):
                cells = {}
                for i, q in enumerate(injections):
                    e = q["pos"]
                    tl = near[e]
                    zi = (raw_scores_rows[:, inj_off[i]:inj_off[i + 1]] - mu[tl]) / sd[tl]
                    spot = zi.max(axis=1)
                    det = [(1 + np.sum(spot_null[:, e] >= v)) / (1 + n_null) < CAUGHT_P for v in spot]
                    key = f"{q['size_deg']:g}|{q['snr']:g}"
                    c = cells.setdefault(key, [0, 0])
                    c[0] += int(np.sum(det))
                    c[1] += len(det)
                return {k2: {"k": v[0], "n": v[1], "p": v[0] / max(v[1], 1)} for k2, v in cells.items()}
            if injections and n_null_inj:
                rec["null_injections"] = inj_table(null_inj[name][s].astype(np.float64))
            for mname in maps:
                r = real[mname][name][s].astype(np.float64)
                z = (r - mu) / sd
                p_tile = np.array([(1 + np.sum(N[:, t] >= r[t])) / (1 + n_null) for t in range(prep.n_tiles)])
                k_real = int(np.sum(p_tile < CAUGHT_P))
                m_real = float(z.max())
                spot_rows = []
                for e in range(n_spots):
                    if not len(near[e]):
                        spot_rows.append({"label": spots[e]["label"], "n_tiles": 0})
                        continue
                    zz = z[near[e]]
                    v = float(zz.max())
                    pe = rank_p(spot_null[:, e], v)
                    best = int(near[e][int(np.argmax(zz))])
                    spot_rows.append({"label": spots[e]["label"], "n_tiles": int(len(near[e])), "z": v, "p": pe,
                                      "caught": bool(pe < CAUGHT_P), "best_tile_lon": float(prep.tiles[best, 0]),
                                      "best_tile_lat": float(prep.tiles[best, 1])})
                rand_p = [rank_p(spot_null[:, e], float(z[near[e]].max())) for e in range(n_spots, n_pos) if len(near[e])]
                order = np.lexsort((-z, p_tile))[:25]
                top = []
                for t in order:
                    tv = prep.tile_vec[t]
                    dist = [(math.degrees(math.acos(float(np.clip(tv @ pvec[e], -1, 1)))), spots[e]["label"]) for e in range(n_spots)]
                    dmin = min(dist)
                    top.append({"lon": float(prep.tiles[t, 0]), "lat": float(prep.tiles[t, 1]), "z": float(z[t]), "p_tile": float(p_tile[t]),
                                "nearest_published": dmin[1], "nearest_published_deg": dmin[0]})
                mrec = {"max_z": m_real, "p_max_z": rank_p(m_null, m_real),
                        "count_p_tile_below": k_real, "p_count_high": rank_p(k_null, k_real),
                        "p_count_low": float((1 + np.sum(k_null <= k_real)) / (1 + n_null)),
                        "random_positions_caught_fraction": float(np.mean(np.array(rand_p) < CAUGHT_P)) if rand_p else None,
                        "spots": spot_rows, "top_tiles": top}
                if injections:
                    mrec["injections"] = inj_table(real_inj[mname][name][s][None, :].astype(np.float64))
                rec["maps"][mname] = mrec
                arrays[f"{name}__{s}__{mname}__tiles"] = r.astype(np.float32)
            msum[s] = rec
            arrays[f"{name}__{s}__null_tiles"] = N.astype(np.float32)
            arrays[f"{name}__{s}__spot_null"] = spot_null.astype(np.float32)
        summary["models"][name] = msum
    arrays["tiles_lonlat"] = prep.tiles.astype(np.float32)
    arrays["positions_lonlat"] = np.array(pos, dtype=np.float32)
    np.savez_compressed(out / "scan_arrays.npz", **arrays)
    summary["seconds"] = time.time() - t_start
    (out / "scan_summary.json").write_text(json.dumps(summary, indent=1))

    # short print-out
    for name, msum in summary["models"].items():
        rec = msum["image_head"]
        print(f"\n===== model {name} ({[m['kind'] for m in models if m['name'] == name][0]}), image head =====")
        print(f"simulated skies: largest z 5/50/95% {rec['null_max_z_quantiles_5_50_95']}, count 5/50/95% {rec['null_count_quantiles_5_50_95']}")
        for mname, mrec in rec["maps"].items():
            caught = [r["label"] for r in mrec["spots"] if r.get("caught")]
            print(f"{mname:10s} largest z {mrec['max_z']:.2f} (p {mrec['p_max_z']:.3f}); count {mrec['count_p_tile_below']} "
                  f"(p high {mrec['p_count_high']:.3f}, low {mrec['p_count_low']:.3f}); caught: {' '.join(caught) or 'none'}")
    log(f"wrote {out}; {summary['seconds']:.0f} s in all")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest")
    st.add_argument("--data-dir", required=True)
    st.add_argument("--out-json", default=None)
    st.set_defaults(func=cmd_selftest)

    r = sub.add_parser("train")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--data-dir", required=True)
    r.add_argument("--template", choices=["T", "C"], required=True)
    r.add_argument("--test-dir", action="append", default=[])
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--init", choices=["imagenet", "random"], default="random")
    r.add_argument("--sizes", default=SIZES_DEFAULT)
    r.add_argument("--max-offset-deg", type=float, default=5.0)
    r.add_argument("--train-skies", type=int, default=4000)
    r.add_argument("--val-skies", type=int, default=300)
    r.add_argument("--positions-per-sky", type=int, default=8)
    r.add_argument("--epochs", type=int, default=30)
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--lr", type=float, default=3e-4)
    r.add_argument("--image-loss-weight", type=float, default=1.0)
    r.add_argument("--device", default="cuda")
    r.add_argument("--eval-batch", type=int, default=256)
    r.add_argument("--smoke", action="store_true")
    r.set_defaults(func=cmd_train)

    s = sub.add_parser("scan")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--data-dir", required=True)
    s.add_argument("--model", action="append", default=[], help="name:checkpoint")
    s.add_argument("--maps", default="smica,nilc,sevem,commander")
    s.add_argument("--n-null", type=int, default=400)
    s.add_argument("--n-null-inject", type=int, default=20)
    s.add_argument("--n-random", type=int, default=300)
    s.add_argument("--inject-positions", type=int, default=100)
    s.add_argument("--inject-sizes", default="3,5,10,20")
    s.add_argument("--inject-snr", default="4,6,8")
    s.add_argument("--tile-nside", type=int, default=16)
    s.add_argument("--seed", type=int, default=20261123)
    s.add_argument("--device", default="cuda")
    s.add_argument("--eval-batch", type=int, default=256)
    s.add_argument("--smoke", action="store_true")
    s.set_defaults(func=cmd_scan)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
