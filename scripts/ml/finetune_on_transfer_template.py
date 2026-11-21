"""Fine-tune a saved U-Net on transfer-function collision templates and test it on the fixed benchmark.

Assumptions
-----------
* Training data are made in memory with the same recipe as
  ``scripts/analysis/benchmark_transfer_template.py`` (Gaussian CMB skies, ramp templates through CAMB
  transfer functions, correct patch geometry, signal centre within 5 degrees of the patch centre) but with a
  different random seed, 21 template sizes from 5 to 25 degrees, and amplitudes log-uniform between 1e-5 and
  1.2e-4 in observed peak Delta T / T.
* Segmentation target: the disc where the template exceeds 10% of its peak.
* Loss as in the April recipe: binary cross-entropy plus soft Dice on the mask, plus 0.2 times binary
  cross-entropy on the image-level head. AdamW, cosine schedule, dihedral augmentation.
* The test set is the benchmark's ``benchmark_data.h5`` (seed 20261121), never used for training or model
  selection. The last epoch is used; there is no checkpoint selection on the test set.
* Score: maximum mask probability over the patch (as in April); the image-level head is reported as well.

Usage (from the project root):
    python scripts/ml/finetune_on_transfer_template.py --run-dir RUN --benchmark-dir BENCH --out-dir OUT
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
import torch
import torch.nn.functional as F

import benchmark_transfer_template as bt
import collision_template_transfer as ct
import patch_geometry as pg
import phase3_train_unet as p3
from phase2_observing_model import camb_tt_cls
from phase3_sensitivity_curve import build_model_from_run
from phase3_thresholds import conformal_threshold_from_scores

NSIDE, LMAX = bt.NSIDE, bt.LMAX


def make_patches(n_skies, per_sky, rng, cl_tt, sigma_pix, fwhm_rad, templates, sizes, fine, mask_radius, max_off_px, amp_lo, amp_hi):
    n = n_skies * per_sky
    patches = np.zeros((n, pg.PATCH_PIX, pg.PATCH_PIX), dtype=np.float32)
    masks = np.zeros((n, pg.PATCH_PIX, pg.PATCH_PIX), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.float32)
    amps = np.zeros(n)
    c0 = (pg.PATCH_PIX - 1) / 2.0
    npix = hp.nside2npix(NSIDE)

    def vec2pix(x, y, z):
        return hp.vec2pix(NSIDE, x, y, z)

    row = 0
    t0 = time.time()
    for isky in range(n_skies):
        np.random.seed(int(rng.integers(0, 2**32 - 1)))
        sky = np.asarray(hp.synfast(cl_tt, NSIDE, lmax=LMAX, new=True, pixwin=True, fwhm=fwhm_rad), dtype=np.float64)
        sky = sky + rng.normal(0.0, sigma_pix, size=npix)
        for v in bt.random_positions(rng, per_sky, 50.0):
            lon, lat = hp.vec2ang(v, lonlat=True)
            proj = hp.projector.GnomonicProj(rot=(float(lon[0]), float(lat[0])), xsize=pg.PATCH_PIX, ysize=pg.PATCH_PIX, reso=pg.RESO_ARCMIN)
            patch = np.asarray(proj.projmap(sky, vec2pix), dtype=np.float64)
            if rng.random() < 0.5:
                s = sizes[int(rng.integers(0, len(sizes)))]
                a = math.exp(rng.uniform(math.log(amp_lo), math.log(amp_hi)))
                sign = 1.0 if rng.random() < 0.5 else -1.0
                rr = max_off_px * math.sqrt(rng.random())
                phi = 2 * math.pi * rng.random()
                ang = pg.angular_distance_grid(c0 + rr * math.cos(phi), c0 + rr * math.sin(phi))
                patch = patch + sign * a * np.interp(ang, fine, templates[s]["profile_k"])
                masks[row] = (ang <= mask_radius[s]).astype(np.uint8)
                labels[row], amps[row] = 1.0, a
            patches[row] = patch.astype(np.float32)
            row += 1
        if (isky + 1) % 250 == 0 or isky == n_skies - 1:
            print(f"  skies {isky + 1}/{n_skies}  {time.time() - t0:.0f} s", flush=True)
    return patches, masks, labels, amps


@torch.no_grad()
def score(model, patches, mean, std, device, batch=256):
    model.eval()
    seg = np.zeros(len(patches), dtype=np.float64)
    aux = np.full(len(patches), np.nan)
    for i in range(0, len(patches), batch):
        x = torch.from_numpy(patches[i:i + batch]).to(device)[:, None]
        x = (x - mean) / std
        mask_logits, aux_logits = p3.unpack_model_output(model(x))
        seg[i:i + batch] = torch.sigmoid(mask_logits).flatten(1).max(dim=1).values.double().cpu().numpy()
        image_logits, _ = p3.split_aux_logits(aux_logits, use_image_aux=True, radius_bin_count=0)
        if image_logits is not None:
            aux[i:i + batch] = torch.sigmoid(image_logits.reshape(-1)).double().cpu().numpy()
    return seg, aux


def detection(scores, labels, sel, fpr):
    thr = float(conformal_threshold_from_scores(scores, labels, fpr)["threshold"])
    k, n = int(np.sum(scores[sel] > thr)), int(np.sum(sel))
    return k / max(n, 1), bt.wilson(k, n), thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="run folder with run_config.json and the checkpoint to start from")
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--benchmark-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-skies", type=int, default=3000)
    ap.add_argument("--val-skies", type=int, default=300)
    ap.add_argument("--positions-per-sky", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--fpr-target", type=float, default=0.05)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ---- templates and sky model, as in the benchmark -------------------------------------------------
    cl_tt, _ = camb_tt_cls(lmax=LMAX)
    window = ct.gaussian_beam(LMAX, 5.0) * hp.pixwin(NSIDE, lmax=LMAX)
    pix_area = hp.nside2pixarea(NSIDE)
    sigma_pix = 30.0 * 1e-6 * math.radians(1.0 / 60.0) / math.sqrt(pix_area)
    sizes = [float(x) for x in range(5, 26)]
    # only the real-space profiles are needed here (the benchmark's build_templates also makes filter responses)
    tr = ct.camb_transfer(lmax=LMAX)
    ells = np.arange(LMAX + 1)
    fine = np.radians(np.linspace(0.0, 180.0, 18001))
    templates, mask_radius = {}, {}
    for s in sizes:
        b = np.zeros(LMAX + 1)
        b[tr.ells] = ct.b_ell(tr.ells, tr.k, tr.delta_t, tr.d_star * math.cos(math.radians(s)), a_ramp=1.0 / tr.d_star)
        coeff = (2 * ells + 1) / (4 * math.pi) * b * window
        coeff[:2] = 0.0
        prof = np.polynomial.legendre.legval(np.cos(fine), coeff)
        prof = prof / prof[int(np.argmax(np.abs(prof)))]
        templates[s] = {"profile_k": prof * bt.T_CMB_K}
        below = np.nonzero(np.abs(prof) < 0.1)[0]
        mask_radius[s] = float(fine[below[0]]) if below.size else math.radians(s)
    print("mask radius (deg) for 5, 10, 15, 20, 25 deg templates:", [round(math.degrees(mask_radius[s]), 1) for s in (5.0, 10.0, 15.0, 20.0, 25.0)], flush=True)
    fwhm_rad = math.radians(5.0 / 60.0)
    max_off_px = 5.0 * 60.0 / pg.RESO_ARCMIN

    print("generating training patches", flush=True)
    xtr, mtr, ytr, atr = make_patches(args.train_skies, args.positions_per_sky, rng, cl_tt, sigma_pix, fwhm_rad, templates, sizes, fine, mask_radius, max_off_px, 1e-5, 1.2e-4)
    print("generating validation patches", flush=True)
    xva, mva, yva, ava = make_patches(args.val_skies, args.positions_per_sky, rng, cl_tt, sigma_pix, fwhm_rad, templates, sizes, fine, mask_radius, max_off_px, 1e-5, 1.2e-4)

    # ---- test set: the fixed benchmark --------------------------------------------------------------------
    bench = Path(args.benchmark_dir)
    with h5py.File(bench / "benchmark_data.h5", "r") as h5:
        xte = np.asarray(h5["patches"][:], dtype=np.float32)
        yte = np.asarray(h5["labels"][:], dtype=np.uint8)
        ate = np.asarray(h5["truth/amplitude"][:], dtype=np.float64)
        ste = np.asarray(h5["truth/theta_crit_deg"][:], dtype=np.float64)
    ref = dict(np.load(bench / "benchmark_scores.npz"))
    amps_grid = sorted(set(np.round(ate[yte == 1], 12).tolist()))

    model, run_config, ckpt_path, _ = build_model_from_run(Path(args.run_dir).resolve(), args.checkpoint, device)
    mean = float(run_config["normalization"]["train_mean"])
    std = float(run_config["normalization"]["train_std"])
    print(f"start checkpoint {ckpt_path}; input mean {mean:.3e} K, std {std:.3e} K", flush=True)

    seg0, aux0 = score(model, xte, mean, std, device)
    key = "score__imagenet_b64_aux"
    if key in ref:
        d = float(np.max(np.abs(seg0 - ref[key])))
        print(f"check: starting model through this code path against the benchmark's saved scores: max |diff| {d:.2e}", flush=True)

    # ---- fine-tune ------------------------------------------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(xtr) / args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history = []
    mid_va = (yva == 1) & (ava >= 2e-5) & (ava <= 5e-5)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(xtr))
        tot, nb = 0.0, 0
        t0 = time.time()
        for i in range(0, len(order), args.batch_size):
            idx = np.sort(order[i:i + args.batch_size])
            x = torch.from_numpy(xtr[idx]).to(device)[:, None]
            m = torch.from_numpy(mtr[idx]).to(device)[:, None].float()
            y = torch.from_numpy(ytr[idx]).to(device)
            x = (x - mean) / std
            k = int(rng.integers(0, 4))
            if k:
                x, m = torch.rot90(x, k, dims=(2, 3)), torch.rot90(m, k, dims=(2, 3))
            if rng.random() < 0.5:
                x, m = torch.flip(x, dims=(3,)), torch.flip(m, dims=(3,))
            mask_logits, aux_logits = p3.unpack_model_output(model(x))
            prob = torch.sigmoid(mask_logits)
            inter = (prob * m).flatten(1).sum(1)
            dice = 1.0 - (2.0 * inter + 1.0) / (prob.flatten(1).sum(1) + m.flatten(1).sum(1) + 1.0)
            loss = F.binary_cross_entropy_with_logits(mask_logits, m) + dice.mean()
            image_logits, _ = p3.split_aux_logits(aux_logits, use_image_aux=True, radius_bin_count=0)
            if image_logits is not None:
                loss = loss + 0.2 * F.binary_cross_entropy_with_logits(image_logits.reshape(-1), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        segv, auxv = score(model, xva, mean, std, device)
        pv, _, _ = detection(segv, yva.astype(np.uint8), mid_va, args.fpr_target)
        pa, _, _ = detection(auxv, yva.astype(np.uint8), mid_va, args.fpr_target) if np.all(np.isfinite(auxv)) else (float("nan"), None, None)
        history.append({"epoch": epoch, "train_loss": tot / nb, "val_detect_mid_segmax": pv, "val_detect_mid_aux": pa, "seconds": time.time() - t0})
        print(f"epoch {epoch:2d}  loss {tot / nb:.4f}  validation detection 2e-5..5e-5: mask max {pv:.3f}, image head {pa:.3f}  ({time.time() - t0:.0f} s)", flush=True)

    torch.save({"model_state_dict": model.state_dict(), "start_checkpoint": str(ckpt_path), "args": vars(args)}, out / "finetuned_checkpoint.pt")
    seg1, aux1 = score(model, xte, mean, std, device)

    # ---- test table -------------------------------------------------------------------------------------------
    methods = {k[len("score__"):]: np.asarray(v, dtype=np.float64) for k, v in ref.items() if k.startswith("score__")}
    methods["finetuned_mask_max"] = seg1
    if np.all(np.isfinite(aux1)):
        methods["finetuned_image_head"] = aux1
        methods["april_imagenet_image_head"] = aux0
    report = {"history": history, "methods": {}, "amplitudes": amps_grid}
    print(f"\ntest: fixed benchmark, {len(yte)} patches, detection probability at {args.fpr_target:.0%} false positives, all sizes")
    print(f"{'method':34s}" + "".join(f"{a:>10.0e}" for a in amps_grid))
    for name, s in methods.items():
        row = []
        for a in amps_grid:
            sel = (yte == 1) & (np.round(ate, 12) == a)
            p, ci, thr = detection(s, yte, sel, args.fpr_target)
            row.append({"amplitude": a, "p": p, "ci95": ci})
        by_size = []
        for size in sorted(set(ste[yte == 1].tolist())):
            sel = (yte == 1) & (np.round(ate, 12) == 3e-5) & (ste == size)
            p, ci, _ = detection(s, yte, sel, args.fpr_target)
            by_size.append({"size_deg": size, "p_at_3e-5": p})
        report["methods"][name] = {"by_amplitude": row, "by_size_at_3e-5": by_size}
        print(f"{name:34s}" + "".join(f"{r['p']:10.3f}" for r in row))
    np.savez_compressed(out / "finetune_test_scores.npz", labels=yte, amplitude=ate, size_deg=ste, score__finetuned_mask_max=seg1, score__finetuned_image_head=aux1)
    (out / "finetune_report.json").write_text(json.dumps(report, indent=1))
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
