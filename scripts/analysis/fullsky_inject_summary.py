"""Table and figure from the sensitivity measurement of the full-sky filter search.

Reads ``inject_report.json`` (``fullsky_filter_search.py --stage inject``) and, if present,
``inject_check.json`` (``fullsky_inject_check.py``). Writes a compact summary JSON, a markdown table and
one figure.

Amplitude everywhere is the observed peak dT/T of the disc: the largest |value| of the template as it sits
in the map, after beam and pixel window, without monopole and dipole. Two derived columns translate it:
  * cosine template: z0 = amplitude / peak_value, the central amplitude of the literature's template;
  * transfer template: the z0 of the Sachs-Wolfe ramp with the same primordial gradient,
    (amplitude / peak_value) * (1 - cos(size)) / 5. For large discs the two templates agree, so the two
    columns are the same quantity there. Below about 30 deg the transfer template is brighter than the
    Sachs-Wolfe ramp for the same primordial gradient, so the same search reaches a smaller gradient.

Usage:
    python scripts/analysis/fullsky_inject_summary.py --report inject_report.json --out-dir OUT [--check inject_check.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # slots 1 to 3 of the reference palette (validated together)
# Criterion (a) is "S reaches the candidate threshold near the true position". The end-to-end check of notebook
# entry (ag) found a matching candidate in 89% of such cases, below the 90% fixed beforehand, so the curves are
# labelled "reach the threshold" and not "become candidates".
CURVES = [("candidate", "a50", "half reach the candidate threshold"),
          ("candidate", "a95", "95% reach the candidate threshold"),
          ("stands_out", "a95", "95% would be the strongest feature of a whole sky")]


def e5(v):
    return "none" if v is None else "%.2f" % (v * 1e5)


def conversions(report, template, sizes):
    pv = np.array(report["peak_value"][template])
    if template == "cosine":
        return 1.0 / pv
    return (1.0 - np.cos(np.radians(sizes))) / 5.0 / pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--check", default="")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = json.load(open(args.report))
    sizes = np.array(rep["sizes_deg"])
    maps = rep["real_maps"]
    summary = {"n_skies": rep["n_skies"], "n_positions": rep["n_positions"], "s_global_95": rep["s_global_95"],
               "sizes_deg": rep["sizes_deg"], "amplitudes": rep["amplitudes"], "peak_value": rep["peak_value"],
               "peak_angle_deg": rep["peak_angle_deg"], "thresholds": rep["thresholds"], "pairs": []}
    lines = []
    for pair in rep["pairs"]:
        inj, srch = pair["injected"], pair["searched"]
        conv = conversions(rep, inj, sizes)
        lines += ["", "%s template searched with the %s filters. Amplitudes in units of 1e-5 (observed peak dT/T)." % (inj, srch), "",
                  "| size (deg) | threshold 50% | threshold 95% | stands out 50% | stands out 95% | chance rate, threshold | chance rate, stands out |"
                  " real maps, stands out 95% | stands out 95% as z0 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        rows = []
        for i, row in enumerate(pair["by_size"]):
            c, s = row["candidate"], row["stands_out"]
            real = [s[m]["a95"] for m in maps]
            real_txt = "none" if any(v is None for v in real) or not real else "%.2f to %.2f" % (min(real) * 1e5, max(real) * 1e5)
            z0 = None if s["a95"] is None else s["a95"] * conv[i]
            lines.append("| %g | %s | %s | %s | %s | %.3f | %.4f | %s | %s |" % (
                row["size_deg"], e5(c["a50"]), e5(c["a95"]), e5(s["a50"]), e5(s["a95"]), c["fraction"][0], s["fraction"][0], real_txt, e5(z0)))
            rows.append({"size_deg": row["size_deg"], "candidate_a50": c["a50"], "candidate_a95": c["a95"],
                         "stands_out_a50": s["a50"], "stands_out_a95": s["a95"], "candidate_fraction": c["fraction"],
                         "stands_out_fraction": s["fraction"], "to_z0": float(conv[i]),
                         "real_maps": {m: {"candidate_a50": c[m]["a50"], "candidate_a95": c[m]["a95"],
                                           "stands_out_a50": s[m]["a50"], "stands_out_a95": s[m]["a95"]} for m in maps}})
        summary["pairs"].append({"injected": inj, "searched": srch, "by_size": rows})
    if args.check:
        chk = json.load(open(args.check))
        summary["end_to_end_check"] = {k: chk[k] for k in ("n_cases", "largest_S_difference", "criterion_a_same", "table")}
        summary["end_to_end_check"]["step4_amplitude"] = chk.get("step4_amplitude", "peak")
        lines += ["", "End-to-end check: %d cases; largest |S shortcut - S long way| %.1e; criterion (a) the same in %d; %s" % (
            chk["n_cases"], chk["largest_S_difference"], chk["criterion_a_same"], json.dumps(chk["table"]))]
    text = "\n".join(lines).strip() + "\n"
    print(text)
    (out / "fullsky_inject_summary.md").write_text(text, encoding="utf-8")
    with open(out / "fullsky_inject_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": INK2, "ytick.labelcolor": INK2})
    panels = [p for p in rep["pairs"] if p["injected"] == p["searched"]]
    titles = {"transfer": "Ramp through the transfer functions\n(this project's template and filters)",
              "cosine": "Cosine ramp\n(the template and filters of the published searches)"}
    fig, axes = plt.subplots(1, len(panels), figsize=(11.0, 5.0), sharey=True, facecolor=SURFACE)
    ymax = 0.0
    for ax, pair in zip(np.atleast_1d(axes), panels):
        ax.set_facecolor(SURFACE)
        for (crit, key, label), colour in zip(CURVES, SERIES):
            y = np.array([np.nan if r[crit][key] is None else r[crit][key] * 1e5 for r in pair["by_size"]])
            ax.plot(sizes, y, color=colour, lw=2.0, marker="o", ms=6.5, mec=SURFACE, mew=1.5, label=label, zorder=3)
            ymax = max(ymax, float(np.nanmax(y)))
        ax.set_xscale("log")
        ax.set_xticks([2, 3, 5, 7, 10, 14, 20, 30, 40, 60, 90])
        ax.set_xticklabels(["2", "3", "5", "7", "10", "14", "20", "30", "40", "60", "90"])
        ax.minorticks_off()
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Size of the disc (degrees)")
        ax.set_title(titles[pair["injected"]], fontsize=10.5, color=INK, loc="left")
        ax.tick_params(length=3)
    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("Peak of the disc in the map, ΔT/T × 10⁵")
    top = math.ceil(ymax * 1.08)
    axes[0].set_ylim(0.0, top)
    # direct labels to the right of the last panel, at the end of each line, kept apart from each other
    last = panels[-1]["by_size"][-1]
    ends = [(last[crit][key] * 1e5, short) for (crit, key, _), short in zip(CURVES, ("half reach the threshold", "95% reach the threshold", "95% stand out"))
            if last[crit][key] is not None]
    placed = []
    for y, short in sorted(ends):
        y_text = y if not placed else max(y, placed[-1] + 0.06 * top)
        placed.append(y_text)
        axes[-1].annotate(short, xy=(sizes[-1], y), xytext=(sizes[-1] * 1.12, y_text), textcoords="data", fontsize=9, color=INK2,
                          va="center", annotation_clip=False, zorder=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("How bright a collision must be for the full-sky search to catch it (Planck, 77% of the sky, multipoles up to 256)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.905, "%d simulated skies × %d positions per size. Lower is more sensitive." % (rep["n_skies"], rep["n_positions"]),
             fontsize=9.5, color=INK2, ha="left")
    fig.subplots_adjust(left=0.06, right=0.875, top=0.80, bottom=0.2, wspace=0.06)
    fig.savefig(out / "fullsky_inject_sensitivity.png", dpi=160, facecolor=SURFACE)
    print("wrote", out / "fullsky_inject_sensitivity.png")


if __name__ == "__main__":
    main()
