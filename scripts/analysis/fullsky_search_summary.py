"""Tables from the reports of ``fullsky_filter_search.py`` (notebook entry (w)).

For the WMAP run: the two pre-registered criteria against the table of McEwen et al. (2012).
For every run: candidates per map and template next to the simulated skies, the published candidates that are
recovered, and (for runs with several maps) the candidates found in all maps.

Usage: python scripts/analysis/fullsky_search_summary.py REPORT.json [REPORT.json ...] [--out SUMMARY.json]
"""
import json
import math
import sys

# significance column of Table I of McEwen et al. (arXiv:1202.2861), candidates 0 to 15
PUBLISHED_S = [4.25, 4.15, 4.08, 4.05, 4.26, 4.02, 4.56, 4.67, 4.43, 4.39, 3.94, 3.07, 3.33, 3.07, 3.03, 3.11]
PUBLISHED_THRESHOLDS = [(2.0, 4.25), (8.0, 4.00), (16.0, 3.75), (26.0, 3.50), (75.0, 3.25), (90.0, 3.00)]


def threshold(size):
    return next(t for upto, t in PUBLISHED_THRESHOLDS if size <= upto)


def separation(a, b):
    la, ba, lb, bb = (math.radians(x) for x in (a["lon"], a["lat"], b["lon"], b["lat"]))
    c = math.sin(ba) * math.sin(bb) + math.cos(ba) * math.cos(bb) * math.cos(la - lb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def same_feature(a, b):
    """The match rule of entry (w): same sign, size within 5 deg, centres within max(3 deg, size / 4)."""
    size = max(a["size_deg"], b["size_deg"])
    return a["z0_mK"] * b["z0_mK"] > 0 and abs(a["size_deg"] - b["size_deg"]) <= 5.0 and separation(a, b) <= max(3.0, 0.25 * size)


def main():
    args = sys.argv[1:]
    out_path = None
    if "--out" in args:
        i = args.index("--out")
        out_path = args[i + 1]
        del args[i:i + 2]
    summary = {}
    for path in args:
        rep = json.load(open(path))
        ds = rep["dataset"]
        print("=" * 110)
        print("%s: mask keeps %.4f of the sky, l <= %d, thresholds %s, %d simulated skies" % (
            ds, rep["mask_fraction"], rep["lmax"], rep["thresholds_source"], rep["n_null"]))
        summary[ds] = {"mask_fraction": rep["mask_fraction"], "lmax": rep["lmax"], "n_null": rep["n_null"],
                       "null_candidates": rep["null_candidates"], "maps": {}}
        for t in rep["templates"]:
            nc = rep["null_candidates"][t]
            print("  simulated skies, %-8s: candidates per sky mean %.2f, 5%% / median / 95%%: %s" % (t, nc["mean"], nc["percentiles_5_50_95"]))
        for mname, res in rep["maps"].items():
            ms = summary[ds]["maps"][mname] = {}
            print("-" * 110)
            for t in rep["templates"]:
                lg = res["largest"][t]
                print("%s / %s: %d candidates (p = %.3f); largest S %.2f at %g deg (p = %.3f)" % (
                    mname, t, res["n_candidates"][t], res["p_n_candidates"][t], lg["max_s"], lg["size_deg"], lg["p"]))
                ms[t] = {"n_candidates": res["n_candidates"][t], "p_n_candidates": res["p_n_candidates"][t], "largest": lg,
                         "candidates": [{k: c[k] for k in ("size_deg", "lon", "lat", "S", "z0_mK", "disc_unmasked_fraction")}
                                        for c in res["candidates"][t]]}
            rows = res["mcewen"]
            near = [abs(r["cosine"]["S_nearby_max"] - PUBLISHED_S[r["label"]]) <= 0.5 for r in rows]
            eligible = [r for r in rows if PUBLISHED_S[r["label"]] >= threshold(r["size_deg"])]
            got = [r["label"] for r in eligible if r["cosine"]["recovered"]]
            print("%s, published candidates (cosine template): nearby S within 0.5 of the published S for %d of 16; "
                  "recovered %d of the %d that pass the thresholds on their published S (%s); recovered of all 16: %d (cosine), %d (transfer)" % (
                      mname, sum(near), len(got), len(eligible), got, sum(r["cosine"]["recovered"] for r in rows),
                      sum(r["transfer"]["recovered"] for r in rows)))
            print("   no. size  published S | S at position  nearby max  threshold  z0 here (published) | cosine  transfer  masked")
            for r in rows:
                c, tr = r["cosine"], r["transfer"]
                print("   %2d  %4g   %.2f        | %5.2f          %5.2f       %.2f       %+.3f (%+.2f)       | %-7s %-8s %s" % (
                    r["label"], r["size_deg"], PUBLISHED_S[r["label"]], c["S_at_position"], c["S_nearby_max"], c["threshold"],
                    c["z0_mK_at_position"], r["z0_mK"], "yes" if c["recovered"] else "no", "yes" if tr["recovered"] else "no",
                    "masked" if r["masked_here"] else ""))
            ms["mcewen"] = {"nearby_within_0p5": int(sum(near)), "eligible": [r["label"] for r in eligible], "eligible_recovered": got,
                            "recovered_cosine": [r["label"] for r in rows if r["cosine"]["recovered"]],
                            "recovered_transfer": [r["label"] for r in rows if r["transfer"]["recovered"]],
                            "rows": [{"label": r["label"], "size_deg": r["size_deg"], "published_S": PUBLISHED_S[r["label"]],
                                      "S_at_position": r["cosine"]["S_at_position"], "S_nearby_max": r["cosine"]["S_nearby_max"],
                                      "z0_mK_here": r["cosine"]["z0_mK_at_position"], "z0_mK_published": r["z0_mK"],
                                      "transfer_S_nearby_max": r["transfer"]["S_nearby_max"], "masked_here": r["masked_here"]} for r in rows]}
        # candidates common to all maps of the run
        names = list(rep["maps"])
        if len(names) > 1:
            print("-" * 110)
            summary[ds]["in_all_maps"] = {}
            for t in rep["templates"]:
                first = rep["maps"][names[0]]["candidates"][t]
                common = []
                for c in first:
                    others = [next((o for o in rep["maps"][n]["candidates"][t] if same_feature(c, o)), None) for n in names[1:]]
                    if all(o is not None for o in others):
                        common.append({"size_deg": c["size_deg"], "lon": c["lon"], "lat": c["lat"], "z0_mK": c["z0_mK"],
                                       "S": [c["S"]] + [o["S"] for o in others], "disc_unmasked_fraction": c["disc_unmasked_fraction"]})
                summary[ds]["in_all_maps"][t] = common
                print("%s: %d of the %d %s candidates are found in all %d maps" % (t, len(common), len(first), names[0], len(names)))
                for c in common:
                    print("    %5.1f deg  (l, b) = (%6.1f, %+5.1f)  z0 %+.3f mK  S %s  disc unmasked %.2f" % (
                        c["size_deg"], c["lon"], c["lat"], c["z0_mK"], " ".join("%.2f" % s for s in c["S"]), c["disc_unmasked_fraction"]))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=1)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
