"""Gnomonic patch geometry that matches the array returned by ``hp.gnomview``.

Assumptions (measured on 2026-11-21, notebook entry (e); re-checked by ``check_patch_geometry_v2.py``)
-------------------------------------------------------------------------------------------------
* Patches come from ``hp.gnomview(m, rot=(lon, lat), reso=RESO_ARCMIN, xsize=PATCH_PIX,
  return_projected_map=True)``. The returned array is indexed ``[iy, ix]``.
* The projection is linear in the tangent-plane coordinate: a pixel n columns from the centre is at
  angle arctan(n * reso), not n * reso.
* Column index ``ix`` increases to the WEST (decreasing Galactic longitude). Row index ``iy`` increases
  to the NORTH (increasing Galactic latitude).

These replace, for new work, ``phase_dataset_utils.make_angular_distance_grid`` (bug B1),
``phase2_observing_model.offset_signal_vector`` (bug B2) and ``phase3_fullsky_tile.peak_sky_coord``
(bug B3). The old functions are left untouched for provenance. Angles in degrees unless stated.
"""
from __future__ import annotations

import numpy as np

PATCH_PIX = 256
RESO_ARCMIN = 13.0


def _basis(lon_deg: float, lat_deg: float):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    centre = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    return centre, east, north


def _plane(ix, iy, npix=PATCH_PIX, reso_arcmin=RESO_ARCMIN):
    """Tangent-plane coordinates (east-positive, north-positive) of pixel (ix, iy)."""
    c0 = (npix - 1) / 2.0
    dx = np.radians(reso_arcmin / 60.0)
    return -(np.asarray(ix, dtype=np.float64) - c0) * dx, (np.asarray(iy, dtype=np.float64) - c0) * dx


def pixel_directions(lon_deg, lat_deg, npix=PATCH_PIX, reso_arcmin=RESO_ARCMIN):
    """Unit vectors of every patch pixel, shape (3, npix, npix), indexed [component, iy, ix]."""
    centre, east, north = _basis(lon_deg, lat_deg)
    iy, ix = np.mgrid[0:npix, 0:npix]
    xe, yn = _plane(ix, iy, npix, reso_arcmin)
    v = centre[:, None, None] + xe[None] * east[:, None, None] + yn[None] * north[:, None, None]
    return v / np.linalg.norm(v, axis=0, keepdims=True)


def pixel_to_lonlat(lon_deg, lat_deg, ix, iy, npix=PATCH_PIX, reso_arcmin=RESO_ARCMIN):
    """Galactic (lon, lat) of patch pixel (ix = column, iy = row). Accepts scalars or arrays."""
    centre, east, north = _basis(lon_deg, lat_deg)
    xe, yn = _plane(ix, iy, npix, reso_arcmin)
    v = centre.reshape(3, *([1] * np.ndim(xe))) + xe * east.reshape(3, *([1] * np.ndim(xe))) + yn * north.reshape(3, *([1] * np.ndim(xe)))
    v = v / np.linalg.norm(v, axis=0, keepdims=True)
    lon = np.degrees(np.arctan2(v[1], v[0])) % 360.0
    lat = np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    return lon, lat


def lonlat_to_pixel(lon_deg, lat_deg, lon, lat, npix=PATCH_PIX, reso_arcmin=RESO_ARCMIN):
    """Patch pixel (ix, iy), as floats, of sky position (lon, lat); NaN if behind the tangent plane."""
    centre, east, north = _basis(lon_deg, lat_deg)
    lo, la = np.radians(np.asarray(lon, dtype=np.float64)), np.radians(np.asarray(lat, dtype=np.float64))
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=0)
    w = np.tensordot(centre, v, axes=1)
    xe = np.tensordot(east, v, axes=1) / w
    yn = np.tensordot(north, v, axes=1) / w
    c0 = (npix - 1) / 2.0
    dx = np.radians(reso_arcmin / 60.0)
    ix = np.where(w > 0, c0 - xe / dx, np.nan)
    iy = np.where(w > 0, c0 + yn / dx, np.nan)
    return ix, iy


def angular_distance_grid(center_x_pix=None, center_y_pix=None, npix=PATCH_PIX, reso_arcmin=RESO_ARCMIN):
    """True angular distance (radians) from patch pixel (center_x_pix, center_y_pix) to every pixel."""
    c0 = (npix - 1) / 2.0
    cx = c0 if center_x_pix is None else float(center_x_pix)
    cy = c0 if center_y_pix is None else float(center_y_pix)
    iy, ix = np.mgrid[0:npix, 0:npix]
    xe, yn = _plane(ix, iy, npix, reso_arcmin)
    v = np.stack([xe, yn, np.ones_like(xe)], axis=0)
    v /= np.linalg.norm(v, axis=0, keepdims=True)
    cxe, cyn = _plane(cx, cy, npix, reso_arcmin)
    cv = np.array([float(cxe), float(cyn), 1.0])
    cv /= np.linalg.norm(cv)
    return np.arccos(np.clip(np.tensordot(cv, v, axes=1), -1.0, 1.0))


def angular_separation_deg(lon1, lat1, lon2, lat2):
    l1, b1, l2, b2 = (np.radians(np.asarray(x, dtype=np.float64)) for x in (lon1, lat1, lon2, lat2))
    c = np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
