"""Numbers computed from a finished real-sky run after its results were seen (notebook entries (r), (s)).

None of this was pre-registered. It reads the two files ``benchmark_real_sky.py`` writes
(``real_sky_report.json`` and ``real_sky_scores.npz``) and needs only numpy.

1. The April situation: a network threshold taken from simulated patches without masked pixels, applied
   to tiles whose masked pixels are zero or filled. April's real-sky patches had such holes and its
   simulated patches did not.
2. Where the null triggers sit: how much of all false triggers, over all simulated skies, comes from the
   5% most-triggering tiles, and how the per-tile trigger rate follows the masked fraction.
3. Saturation: share of signal-free tiles with a network score of exactly 1.
4. Networks against the filter on the real sky, each at its own 5% threshold, with a block bootstrap over
   30 x 30 degree sky cells. Tiles overlap heavily, so tiles are not independent trials; whole cells are
   resampled and every threshold is recomputed in each resample.

Usage:
    python scripts/analysis/real_sky_post_hoc.py RUN_DIR [--bootstrap 1000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FILTER = "matched_filter_bank"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    run = Path(args.run_dir)
    z = np.load(run / "real_sky_scores.npz")
    rep = json.loads((run / "real_sky_report.json").read_text())
    q = 1.0 - float(rep["args"]["fpr_target"])
    frac, subset = z["tile_frac"], z["subset"]
    clean = frac >= 0.97
    maps = list(rep["maps"])
    nets = sorted({k[len("sim__"):-len("_unfilled")] for k in z.files if k.startswith("sim__") and k.endswith("_unfilled")})
    print(f"tiles {frac.size} ({int(clean.sum())} at least 97% unmasked); networks: {nets}; maps: {maps}")

    print("\n1. threshold from simulated patches without holes, applied to tiles with holes")
    for net in nets:
        thr = float(np.quantile(z[f"sim__{net}_unfilled"].ravel(), q))
        print(f"{net}: threshold {thr:.4f}")
        for suffix, label in (("_zeroed", "holes zeroed"), ("", "holes filled")):
            sim = z[f"sim__{net}{suffix}"]
            print(f"   simulated skies, {label}: {np.mean(sim > thr):.3f} "
                  f"(tiles >=97% unmasked {np.mean(sim[:, clean[subset]] > thr):.3f}, the rest {np.mean(sim[:, ~clean[subset]] > thr):.3f})")
            for m in maps:
                real = z[f"null__{m}__{net}{suffix}"]
                print(f"   {m:9s} real sky, {label}: {np.mean(real > thr):.3f} "
                      f"(tiles >=97% unmasked {np.mean(real[clean] > thr):.3f}, the rest {np.mean(real[~clean] > thr):.3f})")

    print("\n2. where the false triggers sit, over all simulated skies (each method at its own simulation threshold)")
    for name in [FILTER] + [f"{net}{s}" for net in nets for s in ("", "_zeroed", "_unfilled")]:
        s = z[f"sim__{name}"]
        hit = s > np.quantile(s.ravel(), q)
        f = frac[subset] if s.shape[1] == subset.size else frac
        rate = hit.mean(axis=0)
        top = np.argsort(-rate)[: max(1, int(round(0.05 * rate.size)))]
        print(f"   {name:32s} tiles that never trigger {np.mean(rate == 0):.2f}; the 5% most-triggering tiles carry "
              f"{hit[:, top].sum() / max(hit.sum(), 1):.2f} of all triggers; correlation with the masked fraction {np.corrcoef(rate, 1.0 - f)[0, 1]:+.2f}")

    print("\n3. signal-free tiles with a score of exactly 1 (and of 0.999 or more)")
    for net in nets:
        for suffix in ("", "_zeroed", "_unfilled"):
            s, r = z[f"sim__{net}{suffix}"], z[f"null__{maps[0]}__{net}{suffix}"]
            print(f"   {net + suffix:32s} simulated {np.mean(s >= 1.0):.3f} ({np.mean(s >= 0.999):.3f}); {maps[0]} {np.mean(r >= 1.0):.3f} ({np.mean(r >= 0.999):.3f})")

    variants = [f"{net}_unfilled" for net in nets if f"inj__{maps[0]}__{net}_unfilled" in z.files]
    if not variants:
        print("\n4. skipped: this run has no injections on the unfilled input (--inject-inputs)")
        return
    print(f"\n4. detection at each method's own 5% threshold, unfilled network input; difference to the filter with a 95% interval "
          f"({args.bootstrap} resamples of 30 x 30 degree sky cells)")
    cell = np.floor(z["tile_lon"] / 30.0).astype(int) * 6 + np.floor((z["tile_lat"] + 90.0) / 30.0).astype(int)
    members = [np.flatnonzero(cell == c) for c in np.unique(cell)]
    rng = np.random.default_rng(args.seed)
    amps = rep["amplitudes"]

    def detection(null, inj, idx, ia):
        return float(np.mean(inj[ia][:, idx] > np.quantile(null[idx], q)))

    everything = np.arange(frac.size)
    for amp in (2e-5, 3e-5, 5e-5):
        if amp not in amps:
            continue
        ia = amps.index(amp)
        print(f"   amplitude {amp:g}")
        for m in maps:
            data = {name: (z[f"null__{m}__{name}"], z[f"inj__{m}__{name}"]) for name in [FILTER] + variants}
            full = {name: detection(*data[name], everything, ia) for name in data}
            boot = {name: np.empty(args.bootstrap) for name in data}
            for b in range(args.bootstrap):
                idx = np.concatenate([members[j] for j in rng.integers(0, len(members), size=len(members))])
                for name in data:
                    boot[name][b] = detection(*data[name], idx, ia)
            line = f"      {m:9s} filter {full[FILTER]:.3f}"
            for name in variants:
                lo, hi = np.quantile(boot[name] - boot[FILTER], [0.025, 0.975])
                line += f" | {name} {full[name]:.3f}, difference {full[name] - full[FILTER]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            print(line)


if __name__ == "__main__":
    main()
