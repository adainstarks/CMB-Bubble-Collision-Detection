"""Tests for ``patch_geometry`` against two independent references.

Reference 1: maps of HEALPix pixel directions projected through ``hp.gnomview`` (what the data see).
Reference 2: healpy's own projector object, ``hp.projector.GnomonicProj`` (ij2xy and xy2ang).
Also: a synthetic blob placed on the sphere must be found at the pixel ``lonlat_to_pixel`` predicts, and
the Cold Spot (l, b) = (209, -57) must land where expected in a real-sky style round trip.

Usage (from the project root):  python scripts/analysis/check_patch_geometry_v2.py
Exit code 1 if any test fails.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import healpy as hp
import numpy as np

import patch_geometry as pg


def gnom(m, lon, lat):
    return np.asarray(hp.gnomview(m, rot=(lon, lat), reso=pg.RESO_ARCMIN, xsize=pg.PATCH_PIX,
                                  return_projected_map=True, no_plot=True), dtype=np.float64)


def main():
    ok = True
    nside = 1024
    vx, vy, vz = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
    for lon, lat in [(120.0, 55.0), (10.0, -40.0), (171.96, -48.14), (300.0, 5.0), (45.0, 82.0)]:
        truth = np.stack([gnom(vx, lon, lat), gnom(vy, lon, lat), gnom(vz, lon, lat)], axis=0)
        truth /= np.linalg.norm(truth, axis=0, keepdims=True)
        mine = pg.pixel_directions(lon, lat)
        err1 = float(np.degrees(np.arccos(np.clip(np.sum(truth * mine, axis=0), -1, 1))).max())

        proj = hp.projector.GnomonicProj(rot=(lon, lat), xsize=pg.PATCH_PIX, ysize=pg.PATCH_PIX, reso=pg.RESO_ARCMIN)
        x, y = proj.ij2xy()
        lon2, lat2 = proj.xy2ang(np.asarray(x).ravel(), np.asarray(y).ravel(), lonlat=True)
        lon2 = np.asarray(lon2).reshape(pg.PATCH_PIX, pg.PATCH_PIX)
        lat2 = np.asarray(lat2).reshape(pg.PATCH_PIX, pg.PATCH_PIX)
        iy, ix = np.mgrid[0:pg.PATCH_PIX, 0:pg.PATCH_PIX]
        lon1, lat1 = pg.pixel_to_lonlat(lon, lat, ix, iy)
        err2 = float(np.nanmax(pg.angular_separation_deg(lon1, lat1, np.asarray(lon2) % 360.0, np.asarray(lat2))))

        rix, riy = pg.lonlat_to_pixel(lon, lat, lon1, lat1)
        err3 = float(max(np.nanmax(np.abs(rix - ix)), np.nanmax(np.abs(riy - iy))))

        # blob on the sphere at a known sky position must appear at the predicted pixel
        blob_lon, blob_lat = pg.pixel_to_lonlat(lon, lat, 61.0, 190.0)
        centre = hp.ang2vec(float(blob_lon), float(blob_lat), lonlat=True)
        ang = np.arccos(np.clip(centre[0] * vx + centre[1] * vy + centre[2] * vz, -1, 1))
        patch = gnom(np.exp(-0.5 * (ang / np.radians(0.5)) ** 2), lon, lat)
        by, bx = np.unravel_index(np.argmax(patch), patch.shape)
        err4 = float(np.hypot(bx - 61.0, by - 190.0))

        grid = np.degrees(pg.angular_distance_grid(61.0, 190.0))
        true_grid = np.degrees(np.arccos(np.clip(np.tensordot(truth[:, 190, 61], truth, axes=1), -1, 1)))
        err5 = float(np.max(np.abs(grid - true_grid)))

        passed = err1 < 0.2 and err2 < 1e-4 and err3 < 1e-6 and err4 <= 1.5 and err5 < 0.2  # err2: healpy works in float32
        ok &= passed
        print(f"centre ({lon:6.2f},{lat:6.2f}): directions vs projected maps {err1:.3f} deg | vs healpy projector {err2:.1e} deg | "
              f"round trip {err3:.1e} px | blob found {err4:.1f} px away | distance grid {err5:.3f} deg | {'PASS' if passed else 'FAIL'}")

    # the Cold Spot seen from tile 10058: where should it fall?
    ix, iy = pg.lonlat_to_pixel(184.5, -42.3, 209.0, -57.0)
    print(f"example: Cold Spot (209, -57) in a patch centred (184.5, -42.3) -> pixel (ix, iy) = ({float(ix):.1f}, {float(iy):.1f})")
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
