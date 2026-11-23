"""Diagnostics for notebook entries (ak) to (am) (the template of McEwen et al. carries an edge).

large-sizes
    For finished runs: the cosine candidates of 45 degrees or more on the real maps against the simulated
    skies of the same run, the largest S among those sizes, and the candidates per size (entry (al), item 3).

size-profile
    For chosen published candidates: the largest S near the published position at every filter size of the
    bank, cosine and transfer filters, masked pixels included (entry (ar)). Shows whether a candidate that
    is missing at its published size under another template is there at another size.

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
    python scripts/analysis/mcewen_template_diagnostics.py large-sizes --runs runs/fullsky_search/planck_v2 \\
        runs/fullsky_search/planck_v3 --out report.json
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


def large_sizes(runs, min_size):
    """Cosine template: candidates of ``min_size`` degrees or more on the real maps against the simulated skies
    of the same run (their stored candidate lists), the largest S among those sizes, and candidates per size."""
    out = {}
    for run in runs:
        rep = json.load(open(Path(run) / "fullsky_filter_search_report.json"))
        null = np.load(Path(run) / "null.npz")
        sizes = np.array(rep["sizes_deg"])
        ns = sizes.size
        big = np.flatnonzero(sizes >= min_size)                 # cosine filters: filter index = size index
        n_sims = int(null["max_s"].shape[0])
        cand = null["candidates"]                               # rows: simulated sky, filter, pixel, S, amplitude
        f = cand[:, 1].astype(int)
        cos = f < ns
        counts = np.bincount(cand[cos & np.isin(f, big), 0].astype(int), minlength=n_sims)[:n_sims]
        sim_max = null["max_s"][:, big].max(axis=1)
        per_size_sim = np.bincount(f[cos], minlength=ns)[:ns] / n_sims
        res = {"min_size_deg": float(min_size), "n_simulated_skies": n_sims,
               "simulated": {"candidates_mean": float(counts.mean()), "candidates_percentiles_5_50_95": np.percentile(counts, [5, 50, 95]).tolist(),
                             "share_with_none": float(np.mean(counts == 0)),
                             "largest_S_percentiles_5_50_95": np.percentile(sim_max, [5, 50, 95]).tolist()},
               "sizes_deg": sizes.tolist(), "simulated_candidates_per_size": per_size_sim.tolist(), "maps": {}}
        print("%s: simulated skies, cosine candidates of %g deg or more: mean %.2f, 5/50/95%%: %s, none in %.1f%% of skies; largest S there 5/50/95%%: %s" % (
            Path(run).name, min_size, counts.mean(), np.percentile(counts, [5, 50, 95]).tolist(), 100 * np.mean(counts == 0),
            " ".join("%.2f" % v for v in np.percentile(sim_max, [5, 50, 95]))))
        for mname, m in rep["maps"].items():
            mine = sorted((c for c in m["candidates"]["cosine"] if c["size_deg"] >= min_size), key=lambda c: -c["S"])
            n = len(mine)
            s = float(max(m["max_s"][i] for i in big))
            per_size = [sum(1 for c in m["candidates"]["cosine"] if c["size_deg"] == sz) for sz in sizes]
            res["maps"][mname] = {"n_candidates": n, "p_n_candidates": float((1 + np.count_nonzero(counts >= n)) / (n_sims + 1)),
                                  "largest_S": s, "p_largest_S": float((1 + np.count_nonzero(sim_max >= s)) / (n_sims + 1)),
                                  "candidates": [{k: c[k] for k in ("size_deg", "lon", "lat", "S", "z0_mK")} for c in mine],
                                  "candidates_per_size": per_size}
            print("   %-10s %d candidates (p = %.3f), largest S %.2f (p = %.3f): %s" % (
                mname, n, res["maps"][mname]["p_n_candidates"], s, res["maps"][mname]["p_largest_S"],
                "; ".join("%g deg (%.1f, %+.1f) S %.2f z0 %+.3f mK" % (c["size_deg"], c["lon"], c["lat"], c["S"], c["z0_mK"]) for c in mine)))
        out[Path(run).name] = res
    return out


def size_profile(runs, map_name, labels):
    """For published candidates: the largest S near the published position at every filter size of the bank,
    for the cosine and the transfer filters of each run, masked pixels included (entry (ar))."""
    out = {}
    ns = len(fs.SIZES)
    for run in runs:
        sigma = np.load(Path(run) / "sigma.npy", mmap_mode="r")
        w = np.load(Path(run) / ("filtered_%s.npy" % map_name), mmap_mode="r")
        res = {}
        for label, z0_mk, colat, lon, size in fs.MCEWEN_2012:
            if int(label) not in labels:
                continue
            v = hp.ang2vec(lon, 90.0 - colat, lonlat=True)
            rows = {"cosine": [], "transfer": []}
            for ti, tname in enumerate(("cosine", "transfer")):
                for si, fsize in enumerate(fs.SIZES):
                    near = hp.query_disc(fs.NSIDE, v, math.radians(max(3.0, 0.25 * fsize)))
                    f = ti * ns + si
                    wf = np.asarray(w[f][near], dtype=np.float64)
                    sg = np.asarray(sigma[f][near], dtype=np.float64)
                    same_sign = wf * z0_mk > 0
                    rows[tname].append(float(np.max(np.abs(wf[same_sign]) / sg[same_sign])) if same_sign.any() else 0.0)
            res[int(label)] = {"published_size_deg": float(size), "published_S": PUBLISHED_S[int(label)], **rows}
            for tname in ("cosine", "transfer"):
                best = int(np.argmax(rows[tname]))
                print("%s no. %d (published %g deg, S %.2f), %s filters: largest S %.2f at %g deg; at 60, 70, 80, 85, 90 deg: %s" % (
                    Path(run).name, int(label), size, PUBLISHED_S[int(label)], tname, rows[tname][best], fs.SIZES[best],
                    " ".join("%.2f" % rows[tname][fs.SIZES.index(x)] for x in (60.0, 70.0, 80.0, 85.0, 90.0))))
        out[Path(run).name] = {"map": map_name, "sizes_deg": list(fs.SIZES), "candidates": res}
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
    c = sub.add_parser("large-sizes")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--min-size", type=float, default=45.0)
    c.add_argument("--out", required=True)
    d = sub.add_parser("size-profile")
    d.add_argument("--runs", nargs="+", required=True)
    d.add_argument("--map", default="wmap7_w_forered")
    d.add_argument("--labels", default="12,13,14,15")
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.command == "masked-positions":
        report = masked_positions(args.runs, args.map)
    elif args.command == "size-profile":
        report = size_profile(args.runs, args.map, [int(x) for x in args.labels.split(",")])
    elif args.command == "large-sizes":
        report = large_sizes(args.runs, args.min_size)
    else:
        report = edge_response(args.dataset, args.data_dir)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
