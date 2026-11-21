"""Recompute the frozen candidates' sky positions with the corrected geometry, recluster, cross-match.

Assumptions
-----------
* Input scores, peaks and tiles are the April ones, unchanged. Only the pixel-to-sky conversion changes
  (``patch_geometry.pixel_to_lonlat`` instead of ``phase3_fullsky_tile.peak_sky_coord``, bug B3).
* Clustering: greedy by ``policy_margin``, pooling the four component-separated maps; a candidate joins the
  first (highest-score) cluster whose seed lies within the radius, else it seeds a new cluster.
* Published comparison list: McEwen, Feeney, Johnson, Peiris (2012), Table I, WMAP 7-year W band,
  (colatitude, longitude) in Galactic coordinates. A match is a separation of at most
  max(5 deg, theta_crit / 2). Only labels with theta_crit between 5 and 26 deg enter the chance test.
* Chance test (pre-registered, notebook entry (h)): random locations drawn from eligible tile centres.

Usage (from the project root):
    python scripts/followup/recompute_candidate_positions.py --out-dir OUT [--tile-features NPZ] [--skip-mc]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import numpy as np

import patch_geometry as pg

# label, z0 (mK), colatitude, longitude, theta_crit (deg); McEwen et al. 2012, Table I
MCEWEN_2012 = [
    (0, 0.24, 119.0, 304.5, 1.5), (1, 0.20, 78.3, 176.5, 2), (2, 0.20, 112.3, 264.4, 2), (3, -0.19, 145.1, 33.0, 2),
    (4, -0.17, 169.0, 187.5, 3), (5, 0.17, 72.4, 150.8, 3), (6, -0.16, 167.2, 268.7, 4), (7, -0.16, 147.4, 207.1, 5),
    (8, 0.15, 123.2, 321.3, 5), (9, 0.14, 62.7, 220.4, 7), (10, 0.11, 136.6, 172.6, 20), (11, -0.09, 127.2, 216.9, 26),
    (12, 0.09, 116.3, 31.6, 35), (13, 0.10, 136.6, 282.0, 40), (14, 0.15, 69.6, 62.6, 85), (15, -0.16, 88.5, 277.7, 90),
]
IN_RANGE = (5.0, 26.0)
DEFAULT_CANDIDATES = "runs/phase3_unet/remediated_v1_tile_constrained_candidates/candidate_records.jsonl"
DEFAULT_TILES = "runs/phase3_unet/batch6_fullsky_nside32_smica/tile_features_smica_nside32.npz"


def cluster(recs, radius_deg):
    order = sorted(range(len(recs)), key=lambda i: -recs[i]["score"])
    clusters = []
    for i in order:
        r = recs[i]
        for c in clusters:
            if float(pg.angular_separation_deg(r["lon"], r["lat"], c["lon"], c["lat"])) <= radius_deg:
                c["members"].append(i)
                break
        else:
            clusters.append({"lon": r["lon"], "lat": r["lat"], "score": r["score"], "members": [i]})
    out = []
    for c in clusters:
        m = [recs[i] for i in c["members"]]
        out.append({
            "lon": c["lon"], "lat": c["lat"], "max_score": c["score"], "n_members": len(m),
            "maps": sorted({x["map"] for x in m}), "tiles": sorted({int(x["patch_index"]) for x in m}),
            "scatter_deg": float(max(float(pg.angular_separation_deg(x["lon"], x["lat"], c["lon"], c["lat"])) for x in m)),
        })
    return out


def matches(lons, lats):
    """Set of in-range published labels matched by at least one location."""
    hit = set()
    for label, _z0, colat, lon, tc in MCEWEN_2012:
        if not (IN_RANGE[0] <= tc <= IN_RANGE[1]):
            continue
        sep = pg.angular_separation_deg(np.asarray(lons), np.asarray(lats), lon, 90.0 - colat)
        if np.any(sep <= max(5.0, tc / 2.0)):
            hit.add(label)
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--tile-features", default=DEFAULT_TILES)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-mc", action="store_true")
    ap.add_argument("--draws", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()

    recs = []
    worst_repro = 0.0
    for line in open(args.candidates):
        r = json.loads(line)
        lon, lat = pg.pixel_to_lonlat(r["patch_center_glon_deg"], r["patch_center_glat_deg"], r["peak_pixel_j"], r["peak_pixel_i"])
        # the April conversion, reproduced: columns taken as east, row 0 taken as north
        c0 = (pg.PATCH_PIX - 1) / 2.0
        old_lon, old_lat = pg.pixel_to_lonlat(r["patch_center_glon_deg"], r["patch_center_glat_deg"], 2 * c0 - r["peak_pixel_j"], 2 * c0 - r["peak_pixel_i"])
        worst_repro = max(worst_repro, float(pg.angular_separation_deg(old_lon, old_lat, r["peak_glon_deg"], r["peak_glat_deg"])))
        recs.append({"map": r["map"], "patch_index": r["patch_index"], "score": float(r["policy_margin"]),
                     "lon": float(lon), "lat": float(lat), "reported_lon": r["peak_glon_deg"], "reported_lat": r["peak_glat_deg"],
                     "peak_pixel_i": r["peak_pixel_i"], "peak_pixel_j": r["peak_pixel_j"]})
    print(f"{len(recs)} tile candidates; April positions reproduced by the reflected convention to {worst_repro:.4f} deg")

    out = {"n_candidates": len(recs), "april_reproduction_max_deg": worst_repro, "clusterings": {}}
    for radius in (5.0, 10.0, 15.0):
        cl = cluster(recs, radius)
        out["clusterings"][str(radius)] = cl
        print(f"\n== clusters at {radius:.0f} deg: {len(cl)} ==")
        print(f"{'(l, b)':>16s} {'score':>6s} {'n':>3s} {'scatter':>8s}  {'maps':28s} nearest published (label, theta_crit, z0 mK, separation)")
        for c in cl:
            best = min(MCEWEN_2012, key=lambda m: float(pg.angular_separation_deg(c["lon"], c["lat"], m[3], 90.0 - m[2])))
            sep = float(pg.angular_separation_deg(c["lon"], c["lat"], best[3], 90.0 - best[2]))
            c["nearest_mcewen"] = {"label": best[0], "theta_crit_deg": best[4], "z0_mK": best[1], "separation_deg": sep,
                                   "is_match": bool(sep <= max(5.0, best[4] / 2.0))}
            print(f"({c['lon']:6.1f},{c['lat']:6.1f}) {c['max_score']:6.3f} {c['n_members']:3d} {c['scatter_deg']:8.1f}  {','.join(c['maps']):28s} "
                  f"#{best[0]} th={best[4]} z0={best[1]:+.2f} sep={sep:.1f}{' MATCH' if c['nearest_mcewen']['is_match'] else ''}")

    primary = out["clusterings"]["5.0"]
    lons = [c["lon"] for c in primary]
    lats = [c["lat"] for c in primary]
    observed = matches(lons, lats)
    out["primary_radius_deg"] = 5.0
    out["observed_matched_labels"] = sorted(observed)
    print(f"\nprimary clusters: {len(primary)}; in-range published candidates matched: {sorted(observed)}")

    if not args.skip_mc:
        with np.load(args.tile_features) as z:
            keys = list(z.files)
            lon_key = next(k for k in keys if "glon" in k)
            lat_key = next(k for k in keys if "glat" in k)
            mask_key = next(k for k in keys if "mask" in k and "frac" in k)
            tlon, tlat, mfrac = np.asarray(z[lon_key]), np.asarray(z[lat_key]), np.asarray(z[mask_key])
        elig = mfrac >= 0.9
        print(f"tile features: {tlon.size} tiles, {int(elig.sum())} eligible (keys {lon_key}, {lat_key}, {mask_key})")
        rng = np.random.default_rng(args.seed)
        results = {}
        for name, sel in (("all eligible tiles", elig), ("eligible tiles, southern Galactic hemisphere", elig & (tlat < 0))):
            pool_lon, pool_lat = tlon[sel], tlat[sel]
            n = len(primary)
            counts = np.zeros(6, dtype=np.int64)
            for _ in range(args.draws):
                idx = rng.choice(pool_lon.size, size=n, replace=False)
                counts[len(matches(pool_lon[idx], pool_lat[idx]))] += 1
            p = float(counts[len(observed):].sum() / args.draws)
            results[name] = {"pool": int(pool_lon.size), "draws": args.draws, "counts_by_number_matched": counts.tolist(), "p_at_least_observed": p}
            print(f"chance test, {name} ({pool_lon.size} tiles, {n} locations, {args.draws} draws): "
                  f"P(matched >= {len(observed)}) = {p:.4f}; distribution {counts.tolist()}")
        out["chance_test"] = results

    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "candidates_corrected.json").write_text(json.dumps({"summary": out, "candidates": recs}, indent=1))
    print("wrote", od / "candidates_corrected.json")


if __name__ == "__main__":
    main()
