"""Compare two runs of the full-sky filter search from their summaries (``fullsky_search_summary.py --out``).

Written for notebook entries (ah) and (ai): the same search before and after step 4 was changed to compare
z0 instead of the peak in the map. For every data set, map and template it lists the number of candidates in
both runs, the candidates that are in only one of them, the simulated-sky statistics and, where present, the
two criteria of the WMAP check.

Usage: python scripts/analysis/fullsky_compare_runs.py OLD.json NEW.json [OLD2.json NEW2.json ...] --out REPORT.json
"""
import json
import sys


def key(c):
    return (c["size_deg"], round(c["lon"], 1), round(c["lat"], 1))


def brief(c):
    return {"size_deg": c["size_deg"], "lon": round(c["lon"], 1), "lat": round(c["lat"], 1), "S": round(c["S"], 2)}


def main():
    args = sys.argv[1:]
    out_path = None
    if "--out" in args:
        i = args.index("--out")
        out_path = args[i + 1]
        del args[i:i + 2]
    report = {}
    for old_path, new_path in zip(args[0::2], args[1::2]):
        old, new = json.load(open(old_path)), json.load(open(new_path))
        for ds in old:
            a, b = old[ds], new[ds]
            entry = {"old": old_path.replace("\\", "/").split("/")[-1], "new": new_path.replace("\\", "/").split("/")[-1],
                     "simulated_skies": {t: {"old": a["null_candidates"][t], "new": b["null_candidates"][t]} for t in a["null_candidates"]},
                     "maps": {}}
            print("%s: simulated skies, cosine candidates per sky: mean %.2f -> %.2f" % (
                ds, a["null_candidates"]["cosine"]["mean"], b["null_candidates"]["cosine"]["mean"]))
            for m in a["maps"]:
                row = {}
                for t in ("cosine", "transfer"):
                    ca = {key(c): c for c in a["maps"][m][t].get("candidates", [])}
                    cb = {key(c): c for c in b["maps"][m][t].get("candidates", [])}
                    row[t] = {"n_old": len(ca), "n_new": len(cb), "p_n_new": b["maps"][m][t]["p_n_candidates"],
                              "largest_S_new": b["maps"][m][t]["largest"]["max_s"], "p_largest_S_new": b["maps"][m][t]["largest"]["p"],
                              "only_in_old": [brief(ca[k]) for k in sorted(set(ca) - set(cb))],
                              "only_in_new": [brief(cb[k]) for k in sorted(set(cb) - set(ca))],
                              "sizes_45_deg_or_more_new": sorted(k[0] for k in cb if k[0] >= 45.0)}
                    print("  %-16s %-8s %3d -> %3d candidates (p = %.3f); only in old: %s; only in new: %s" % (
                        m, t, len(ca), len(cb), row[t]["p_n_new"], row[t]["only_in_old"], row[t]["only_in_new"]))
                mc_a, mc_b = a["maps"][m].get("mcewen"), b["maps"][m].get("mcewen")
                if mc_a and mc_b:
                    row["published_candidates"] = {k: {"old": mc_a.get(k), "new": mc_b.get(k)}
                                                   for k in ("nearby_within_0p5", "eligible_recovered", "recovered_cosine")}
                    print("  %-16s published candidates: within 0.5 in S %s -> %s; eligible found %s -> %s" % (
                        m, mc_a.get("nearby_within_0p5"), mc_b.get("nearby_within_0p5"),
                        len(mc_a.get("eligible_recovered", [])), len(mc_b.get("eligible_recovered", []))))
                entry["maps"][m] = row
            report[ds] = entry
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=1)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
