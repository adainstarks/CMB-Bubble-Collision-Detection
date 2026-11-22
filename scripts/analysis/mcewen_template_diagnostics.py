"""Two diagnostics for notebook entry (ak) (the template of McEwen et al. carries an edge).

masked-positions
    The search of ``fullsky_filter_search.py`` runs over unmasked pixels only. Two of the 16 published
    candidates of McEwen et al. (2012) are centred on masked pixels, so their search did not leave those out.
    This command reads the filtered map and the sigma map of finished runs and gives the significance at the
    published positions, and the largest one nearby, with masked pixels included.

edge-response
    What an edge in a real signal would do to the sensitivity numbers of entry (ag): the response of this
    project's filters (cosine with z_crit = 0, and transfer) to a collision that carries an edge, relative to
    one with the same z0 and no edge. Two signals: the template of their code (0.2 z0 at the edge, smooth
    cut-off out to 1.1 times the size) and a plain jump of 0.5 z0. Also the share of the best possible
    significance (that of the signal's own matched filter) which the z_crit = 0 filter reaches.

Usage (from the project root):
    python scripts/analysis/mcewen_template_diagnostics.py masked-positions --runs runs/fullsky_search/wmap7w_v4 \\
        runs/fullsky_search/wmap7w_v6 --out report.json
    python scripts/analysis/mcewen_template_diagnostics.py edge-response --dataset planck --out report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import healpy as hp
import numpy as np

import fullsky_filter_search as fs

# significance column of Table I of McEwen et al. (arXiv:1202.2861), candidates 0 to 15
PUBLISHED_S = [4.25, 4.15, 4.08, 4.05, 4.26, 4.02, 4.56, 4.67, 4.43, 4.39, 3.94, 3.07, 3.33, 3.07, 3.03, 3.11]


def masked_positions(runs, map_name):
    out = {}
    for run in runs:
        sigma = np.load(Path(run) / "sigma.npy", mmap_mode="r")
        w = np.load(Path(run) / ("filtered_%s.npy" % map_name), mmap_mode="r")
        rows = []
        for label, z0_mk, colat, lon, size in fs.MCEWEN_2012:
            lat = 90.0 - colat
            f = fs.SIZES.index(float(size))                # cosine template: filter index = size index
            v = hp.ang2vec(lon, lat, lonlat=True)
            p0 = int(hp.vec2pix(fs.NSIDE, *v))
            near = hp.query_disc(fs.NSIDE, v, math.radians(max(1.5, 0.25 * size)))
            S = np.abs(np.asarray(w[f], dtype=np.float64)) / np.asarray(sigma[f], dtype=np.float64)
            rows.append({"label": int(label), "size_deg": float(size), "lon": float(lon), "lat": float(lat),
                         "published_S": PUBLISHED_S[int(label)], "published_z0_mK": float(z0_mk),
                         "S_at_position": float(S[p0]), "S_nearby_max": float(S[near].max())})
        n_near = sum(abs(r["S_nearby_max"] - r["published_S"]) <= 0.5 for r in rows)
        n_at = sum(abs(r["S_at_position"] - r["published_S"]) <= 0.5 for r in rows)
        out[Path(run).name] = {"map": map_name, "within_0p5_nearby_max": int(n_near), "within_0p5_at_position": int(n_at), "candidates": rows}
        print("%s, %s, masked pixels included: S within 0.5 of the published value for %d of 16 (largest nearby), %d of 16 (at the position)" % (
            Path(run).name, map_name, n_near, n_at))
        print("   no.: at position / nearby (published): " + "  ".join(
            "%d: %.2f / %.2f (%.2f)" % (r["label"], r["S_at_position"], r["S_nearby_max"], r["published_S"]) for r in rows))
    return out


def edge_response(dataset, data_dir):
    ds = fs.load_dataset(dataset, Path(data_dir), 1)
    lmax, window, power = ds["lmax"], ds["window"], ds["total_power"]
    ells = np.arange(lmax + 1)
    wt = (2 * ells + 1) / (4 * math.pi)
    tr = fs.ct.camb_transfer(lmax=lmax, params=ds["transfer_params"])
    fine = np.radians(np.linspace(0.0, 180.0, 7201))

    def clean(t):
        t = t * window
        t[:2] = 0.0
        return t

    def peak(t):
        return float(np.max(np.abs(np.polynomial.legendre.legval(np.cos(fine), wt * t))))

    def resp(tf, tx):          # significance per unit z0 of signal x in the filter matched to f
        return float(np.sum(wt * tf * tx / power) / math.sqrt(np.sum(wt * tf * tf / power)))

    rows = []
    for size in fs.INJECT_SIZES:
        tc = math.radians(size)
        mu_c = math.cos(tc)
        plain = clean(fs.cosine_coefficients(tc, lmax))
        code = clean(fs.cosine_edge_coefficients(tc, lmax, 0.2, (1.0, 1.1, "schwartz")))
        x, w = np.polynomial.legendre.leggauss(lmax // 2 + 8)
        mu = 0.5 * (1.0 - mu_c) * x + 0.5 * (1.0 + mu_c)
        jump = clean(2.0 * math.pi * ((0.5 * (1.0 - mu_c) * w * (1.0 - 0.5 * (1.0 - mu) / (1.0 - mu_c))) @ np.polynomial.legendre.legvander(mu, lmax)))
        transfer = np.zeros(lmax + 1)
        transfer[tr.ells] = fs.ct.b_ell(tr.ells, tr.k, tr.delta_t, tr.d_star * mu_c, a_ramp=1.0 / tr.d_star)
        transfer = clean(transfer)
        row = {"size_deg": size, "observed_peak_per_unit_z0": {"z_crit_0": peak(plain), "code_template": peak(code), "jump_0p5": peak(jump)}}
        for name, sig in (("code_template", code), ("jump_0p5", jump)):
            row[name] = {"cosine_filter_same_z0": resp(plain, sig) / resp(plain, plain),
                         "transfer_filter_same_z0": resp(transfer, sig) / resp(transfer, plain),
                         "share_of_best_possible": resp(plain, sig) / resp(sig, sig),
                         "own_matched_filter_same_z0": resp(sig, sig) / resp(plain, plain)}
        rows.append(row)
        print("%4.0f deg | signal with the edge of their code: S in the z_crit = 0 filter %.2f of the no-edge signal, in the transfer filter %.2f, "
              "share of the best possible %.2f, its own filter %.2f | plain jump of 0.5 z0: %.2f, %.2f, %.2f, %.2f" % (
                  size, *[row["code_template"][k] for k in ("cosine_filter_same_z0", "transfer_filter_same_z0", "share_of_best_possible", "own_matched_filter_same_z0")],
                  *[row["jump_0p5"][k] for k in ("cosine_filter_same_z0", "transfer_filter_same_z0", "share_of_best_possible", "own_matched_filter_same_z0")]))
    return {"dataset": dataset, "lmax": int(lmax), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("masked-positions")
    a.add_argument("--runs", nargs="+", required=True)
    a.add_argument("--map", default="wmap7_w_forered")
    a.add_argument("--out", required=True)
    b = sub.add_parser("edge-response")
    b.add_argument("--dataset", default="planck")
    b.add_argument("--data-dir", default="data")
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    report = masked_positions(args.runs, args.map) if args.command == "masked-positions" else edge_response(args.dataset, args.data_dir)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
