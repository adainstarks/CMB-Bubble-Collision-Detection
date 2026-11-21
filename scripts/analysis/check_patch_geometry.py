"""Numerical test of the patch geometry conventions against healpy's gnomonic projection.

Assumptions
-----------
* Backgrounds in every data set come from ``hp.gnomview(..., reso=13 arcmin, xsize=256,
  return_projected_map=True)``. Truth for "where on the sky is patch pixel (iy, ix)" is obtained by
  projecting maps of the three Cartesian components of the HEALPix pixel directions through that same call.
* Tested code: ``phase_dataset_utils.make_angular_distance_grid`` (used to draw signals and masks on the
  patch) and ``phase2_observing_model.offset_signal_vector`` (used to place signals on the sphere).

Questions
---------
B1. Does the code's angular-distance grid equal the true angular distance between patch pixels?
B2. Does ``offset_signal_vector(cx, cy)`` point at the sky direction of patch pixel (iy=cy, ix=cx)?

Usage (from the project root):  python scripts/analysis/check_patch_geometry.py --out-json OUT
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import healpy as hp
import numpy as np

from phase_dataset_utils import make_angular_distance_grid
from phase2_observing_model import offset_signal_vector

PATCH_PIX = 256
RESO_ARCMIN = 13.0


def gnom(hp_map, lon, lat):
    return np.asarray(
        hp.gnomview(hp_map, rot=(lon, lat), reso=RESO_ARCMIN, xsize=PATCH_PIX, return_projected_map=True, no_plot=True),
        dtype=np.float64,
    )


def true_directions(nside, lon, lat):
    vx, vy, vz = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
    d = np.stack([gnom(vx, lon, lat), gnom(vy, lon, lat), gnom(vz, lon, lat)], axis=0)
    return d / np.linalg.norm(d, axis=0, keepdims=True)


def linear_plane_grid(cx, cy):
    """Angular distance if pixel index is linear in tangent-plane coordinate (true gnomonic geometry)."""
    c = (PATCH_PIX - 1) / 2.0
    dx = np.radians(RESO_ARCMIN / 60.0)
    iy, ix = np.mgrid[0:PATCH_PIX, 0:PATCH_PIX]
    v = np.stack([(ix - c) * dx, (iy - c) * dx, np.ones((PATCH_PIX, PATCH_PIX))], axis=0)
    v /= np.linalg.norm(v, axis=0, keepdims=True)
    cv = np.array([(cx - c) * dx, (cy - c) * dx, 1.0])
    cv /= np.linalg.norm(cv)
    return np.arccos(np.clip(np.tensordot(cv, v, axes=1), -1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--nside", type=int, default=1024)
    args = ap.parse_args()
    c = (PATCH_PIX - 1) / 2.0
    report = {"patch_pix": PATCH_PIX, "reso_arcmin": RESO_ARCMIN, "nside_truth": args.nside, "centres": []}

    for lon, lat in [(120.0, 55.0), (10.0, -40.0)]:
        dirs = true_directions(args.nside, lon, lat)
        rec = {"lon": lon, "lat": lat, "B1": [], "B2": []}
        print(f"\n===== patch centre (l, b) = ({lon}, {lat}) =====")
        print("B1: angular distance from a signal centre, code grid vs truth (degrees)")
        print(f"{'centre (cx,cy)':>16s} {'nominal':>8s} {'true on-axis':>13s} {'true diagonal':>14s} {'max |code-true|':>16s} {'max |linear-true|':>18s}")
        for cx, cy in [(c, c), (64.0, 127.5), (200.0, 60.0), (30.0, 220.0)]:
            code = np.degrees(make_angular_distance_grid(PATCH_PIX, RESO_ARCMIN, center_x_pix=cx, center_y_pix=cy))
            lin = np.degrees(linear_plane_grid(cx, cy))
            iy0, ix0 = int(round(cy)), int(round(cx))
            if abs(cx - ix0) > 1e-6 or abs(cy - iy0) > 1e-6:  # half-pixel centre: average the four neighbours
                cv = dirs[:, int(np.floor(cy)):int(np.floor(cy)) + 2, int(np.floor(cx)):int(np.floor(cx)) + 2].reshape(3, -1).mean(axis=1)
                cv /= np.linalg.norm(cv)
            else:
                cv = dirs[:, iy0, ix0]
            true = np.degrees(np.arccos(np.clip(np.tensordot(cv, dirs, axes=1), -1, 1)))
            for nominal in (5.0, 10.0, 15.0, 20.0, 25.0):
                ring = np.abs(code - nominal) < 0.11  # pixels the code calls "nominal" degrees from the centre
                if ring.sum() < 8:
                    continue
                row = iy0 if 0 <= iy0 < PATCH_PIX else int(c)
                on_axis = ring[row, :]
                t_axis = float(np.mean(true[row, :][on_axis])) if on_axis.any() else float("nan")
                iy, ix = np.nonzero(ring)
                diag = np.abs(np.abs(iy - cy) - np.abs(ix - cx)) < 1.5
                t_diag = float(np.mean(true[iy[diag], ix[diag]])) if diag.any() else float("nan")
                entry = {"cx": cx, "cy": cy, "nominal_deg": nominal, "true_on_axis_deg": t_axis, "true_diagonal_deg": t_diag,
                         "true_ring_min_deg": float(true[ring].min()), "true_ring_max_deg": float(true[ring].max())}
                rec["B1"].append(entry)
                print(f"{'(%.1f,%.1f)' % (cx, cy):>16s} {nominal:8.1f} {t_axis:13.2f} {t_diag:14.2f} "
                      f"{float(np.max(np.abs(code - true))):16.2f} {float(np.max(np.abs(lin - true))):18.3f}")
        print("\nB2: where does offset_signal_vector(cx, cy) land in the projected patch?")
        print(f"{'asked (cx,cy)':>16s} {'landed (ix,iy)':>16s} {'offset px':>10s} {'mirrored-x match?':>18s}")
        for cx, cy in [(127.5, 127.5), (60.0, 127.5), (200.0, 127.5), (127.5, 40.0), (127.5, 210.0), (60.0, 200.0), (210.0, 50.0)]:
            v = offset_signal_vector(lon, lat, center_x_pix=cx, center_y_pix=cy)
            dots = np.tensordot(v, dirs, axes=1)
            iy, ix = np.unravel_index(np.argmax(dots), dots.shape)
            off = float(np.hypot(ix - cx, iy - cy))
            mirrored = float(np.hypot(ix - (PATCH_PIX - 1 - cx), iy - cy))
            mirrored_y = float(np.hypot(ix - cx, iy - (PATCH_PIX - 1 - cy)))
            rec["B2"].append({"asked_cx": cx, "asked_cy": cy, "landed_ix": int(ix), "landed_iy": int(iy), "offset_px": off,
                              "offset_if_x_mirrored_px": mirrored, "offset_if_y_mirrored_px": mirrored_y,
                              "residual_deg": float(np.degrees(np.arccos(min(1.0, float(dots[iy, ix])))))})
            print(f"{'(%.1f,%.1f)' % (cx, cy):>16s} {'(%d,%d)' % (ix, iy):>16s} {off:10.1f} {('x-mirror off=%.1f' % mirrored):>18s}  y-mirror off={mirrored_y:.1f}")
        report["centres"].append(rec)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(report, fh, indent=1)
    print("\nwrote", args.out_json)


if __name__ == "__main__":
    main()
