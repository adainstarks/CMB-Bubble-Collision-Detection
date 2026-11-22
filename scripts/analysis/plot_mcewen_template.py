"""Figure for notebook entry (ak): the template behind the figures of McEwen et al. (2012).

Left: their drawn template (Fig. 1a, z0 = 100 uK, 10 deg) with the template of their text (z_crit = 0) and the
one of their public code (z_crit = 0.2 z0, smooth cut-off out to 1.1 times the size). Right: their published
signal-to-noise curve of the matched filter (Fig. 4a), recomputed with either template, as computed / published.

Usage: python scripts/analysis/plot_mcewen_template.py check_mcewen_template_report.json mcewen2012_fig1a_profile.csv OUT.png
"""
import csv
import json
import math
import sys

import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834"]          # slots 1 and 2 of the reference palette (validated together)


def text_template(theta_deg, size_deg=10.0):
    mu_c = math.cos(math.radians(size_deg))
    th = np.radians(theta_deg)
    return np.where(theta_deg <= size_deg, (np.cos(th) - mu_c) / (1.0 - mu_c), 0.0)


def code_template(theta_deg, size_deg=10.0, edge=0.2, outer=1.1):
    """comb_tmplmap_bubble with its defaults."""
    mu_c = math.cos(math.radians(size_deg))
    th = np.radians(theta_deg)
    s = np.clip((theta_deg - size_deg) / (outer * size_deg - size_deg), 0.0, 1.0)
    k = np.zeros_like(s)
    k[s < 1.0] = np.exp(1.0 - 1.0 / (1.0 - s[s < 1.0] ** 2))
    return (1.0 - (1.0 - edge) * (1.0 - np.cos(th)) / (1.0 - mu_c)) * k


def main():
    rep = json.load(open(sys.argv[1]))
    drawn = np.array([(float(r["theta_deg"]), float(r["delta_t_uK"])) for r in csv.DictReader(open(sys.argv[2]))])
    runs = rep["fig4a_signal_to_noise"]["runs"]
    pub = {int(k): v for k, v in rep["fig4a_signal_to_noise"]["published"]["matched_filter"].items()}
    spectrum = "their file" if any(r["spectrum"] == "their file" for r in runs.values()) else "CAMB"
    pick = {"text": "text, l <= 256, %s spectrum" % spectrum, "code": "code, l <= 512, %s spectrum" % spectrum}
    sizes = np.array([t for t in sorted(pub) if t >= 2])
    ratio = {k: np.array([runs[v]["matched"][str(t)] / pub[t] for t in sizes]) for k, v in pick.items()}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": INK2, "ytick.labelcolor": INK2})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 5.0), facecolor=SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=3)

    th = np.linspace(0.0, 14.0, 2801)
    a.plot(th, 100.0 * text_template(th), color=SERIES[0], lw=2.0, zorder=3)
    a.plot(th, 100.0 * code_template(th), color=SERIES[1], lw=2.0, zorder=3)
    a.plot(drawn[::3, 0], drawn[::3, 1], ls="none", marker="o", ms=3.6, mfc=INK2, mec=SURFACE, mew=0.6, zorder=4)
    a.axvline(10.0, color=AXIS, lw=0.8, zorder=1)
    a.set_xlim(0.0, 14.0)
    a.set_ylim(0.0, 105.0)
    a.set_xlabel("Angle from the centre of the disc (degrees)")
    a.set_ylabel("Temperature of the template (µK)")
    a.set_title("Their drawn template: 100 µK at the centre, size 10°", fontsize=10.5, color=INK, loc="left")
    a.annotate("edge of the disc", xy=(10.0, 98.0), xytext=(10.2, 98.0), fontsize=9, color=INK2, va="center")
    a.annotate("their text: 0 µK at the edge", xy=(7.2, 100.0 * float(text_template(np.array([7.2]))[0])), xytext=(1.0, 30.0),
               fontsize=9, color=INK2, va="center", arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8})
    a.annotate("their code and figure:\n20 µK at the edge,\nthen a smooth fall\nto zero at 11°", xy=(10.0, 20.0), xytext=(10.55, 54.0),
               fontsize=9, color=INK2, va="center", arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8})

    b.axhline(1.0, color=AXIS, lw=0.8, zorder=1)
    b.plot(sizes, ratio["text"], color=SERIES[0], lw=2.0, zorder=3)
    b.plot(sizes, ratio["code"], color=SERIES[1], lw=2.0, zorder=3)
    b.set_xscale("log")
    b.set_xticks([2, 3, 5, 7, 10, 14, 20, 30, 45, 60, 90])
    b.set_xticklabels(["2", "3", "5", "7", "10", "14", "20", "30", "45", "60", "90"])
    b.minorticks_off()
    b.set_xlim(2.0, 90.0)
    b.set_ylim(0.88, 1.04)
    b.set_xlabel("Size of the disc (degrees)")
    b.set_ylabel("Signal-to-noise: recomputed / published")
    b.set_title("Their published filter signal-to-noise curve, recomputed", fontsize=10.5, color=INK, loc="left")
    b.annotate("with the template of their code:\nwithin 2% everywhere, 0.4% on average", xy=(20.0, float(np.interp(20.0, sizes, ratio["code"]))),
               xytext=(6.2, 1.026), fontsize=9, color=INK2, va="center", arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8})
    b.annotate("with the template of their text:\n3 to 10% too low below 60°", xy=(20.0, float(np.interp(20.0, sizes, ratio["text"]))),
               xytext=(9.0, 0.915), fontsize=9, color=INK2, va="center", arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8})

    b.annotate("the small jitter is the 0.01 step in which their figure can be read", xy=(2.1, 0.886), fontsize=8.5, color=MUTED, va="bottom")

    handles = [Line2D([], [], ls="none", marker="o", ms=4.5, mfc=INK2, mec=SURFACE, mew=0.6),
               Line2D([], [], color=SERIES[0], lw=2.0), Line2D([], [], color=SERIES[1], lw=2.0)]
    labels = ["curve drawn in their Fig. 1a", "template of their text, z_crit = 0 (this project's cosine template)",
              "template of their code, z_crit = 0.2 z0"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The template behind McEwen et al.'s figures carries 20% of its central amplitude at the edge",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.905, "Their paper says z_crit ~ 0. Their public code sets z_crit = 0.2 z0, and their published curves follow the code.",
             fontsize=9.5, color=INK2, ha="left")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.2, wspace=0.2)
    fig.savefig(sys.argv[3], dpi=160, facecolor=SURFACE)
    print("wrote", sys.argv[3], "| mean and largest deviation:",
          {k: (round(float(np.mean(v)), 4), round(float(np.max(np.abs(v - 1.0))), 4)) for k, v in ratio.items()})


if __name__ == "__main__":
    main()
