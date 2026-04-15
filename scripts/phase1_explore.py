"""
Phase 1: Data Foundation - Explore the Planck 2018 SMICA CMB map.

Downloads the Planck 2018 SMICA cleaned CMB map, visualizes it,
degrades to Nside=256, applies the galactic mask, and extracts
a sample gnomonic (flat-sky) patch.

Usage (from project root, with cmb conda env activated):
    python scripts/phase1_explore.py
"""

import os
import urllib.request
import numpy as np
import healpy as hp
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")

SMICA_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/"
    "maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits"
)
SMICA_FILE = os.path.join(DATA_DIR, "COM_CMB_IQU-smica_2048_R3.00_full.fits")

MASK_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/"
    "masks/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
)
MASK_FILE = os.path.join(DATA_DIR, "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits")

DISPLAY_UK = 300.0


def download_file(url, dest):
    if os.path.exists(dest):
        try:
            with fits.open(dest, memmap=False):
                pass
            size_mb = os.path.getsize(dest) / 1e6
            print(f"  Already downloaded: {os.path.basename(dest)} ({size_mb:.1f} MB)")
            return
        except Exception:
            print(f"  Existing file is incomplete or invalid, re-downloading: {os.path.basename(dest)}")
            os.remove(dest)

    temp_dest = dest + ".part"
    if os.path.exists(temp_dest):
        os.remove(temp_dest)

    print(f"  Downloading {os.path.basename(dest)}...")
    print(f"  URL: {url}")
    print(f"  This may take a while (large FITS files).")
    try:
        urllib.request.urlretrieve(url, temp_dest)
        with fits.open(temp_dest, memmap=False):
            pass
        os.replace(temp_dest, dest)
    except Exception:
        if os.path.exists(temp_dest):
            os.remove(temp_dest)
        raise

    size_mb = os.path.getsize(dest) / 1e6
    print(f"  Done: {size_mb:.1f} MB")


def to_display_uk(cmb_map, floor_uk=DISPLAY_UK):
    """
    Convert a temperature-anisotropy map from K to µK and choose a symmetric
    display range centered on zero.

    The map itself is already mean-zero anisotropy, but autoscaling in kelvin
    produces a visually misleading pale-blue wash. For presentation we display
    it in µK with symmetric limits, using at least +/-300 µK.
    """
    finite = np.isfinite(cmb_map) & (cmb_map != hp.UNSEEN)
    map_uk = cmb_map * 1e6
    vmax = np.percentile(np.abs(map_uk[finite]), 99.5)
    vmax = max(float(floor_uk), float(vmax))
    return map_uk, -vmax, vmax


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── Step 1: Download the data ──────────────────────────────────────
    print("\n=== Step 1: Download Planck data ===")
    download_file(SMICA_URL, SMICA_FILE)
    download_file(MASK_URL, MASK_FILE)

    # ── Step 2: Load the SMICA map ─────────────────────────────────────
    print("\n=== Step 2: Load SMICA map ===")
    smica_map = hp.read_map(SMICA_FILE, field=0)
    nside_orig = hp.npix2nside(len(smica_map))
    print(f"  Nside: {nside_orig}")
    print(f"  Npix:  {len(smica_map):,}")
    print(f"  Min:   {np.nanmin(smica_map):.6e} K")
    print(f"  Max:   {np.nanmax(smica_map):.6e} K")

    # ── Step 3: Full-sky Mollweide plot at original resolution ─────────
    print("\n=== Step 3: Plot full-sky map (original Nside) ===")
    smica_map_uk, full_vmin, full_vmax = to_display_uk(smica_map)
    hp.mollview(
        smica_map_uk,
        title=f"Planck 2018 SMICA CMB (Nside={nside_orig})",
        unit="µK",
        cmap="RdBu_r",
        min=full_vmin,
        max=full_vmax,
    )
    path_fullsky = os.path.join(PLOT_DIR, "01_smica_fullsky.png")
    plt.savefig(path_fullsky, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path_fullsky}")

    # ── Step 4: Degrade to Nside=256 ───────────────────────────────────
    print("\n=== Step 4: Degrade to Nside=256 ===")
    nside_low = 256
    smica_256 = hp.ud_grade(smica_map, nside_low)
    print(f"  Nside: {nside_low}")
    print(f"  Npix:  {len(smica_256):,}")

    smica_256_uk, low_vmin, low_vmax = to_display_uk(smica_256)
    hp.mollview(
        smica_256_uk,
        title=f"Planck 2018 SMICA CMB (degraded to Nside={nside_low})",
        unit="µK",
        cmap="RdBu_r",
        min=low_vmin,
        max=low_vmax,
    )
    path_degraded = os.path.join(PLOT_DIR, "02_smica_nside256.png")
    plt.savefig(path_degraded, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path_degraded}")

    # ── Step 5: Load and apply galactic mask ───────────────────────────
    print("\n=== Step 5: Apply galactic mask ===")
    try:
        mask = hp.read_map(MASK_FILE, field=0)
        mask_256 = hp.ud_grade(mask, nside_low)
        mask_256 = np.where(mask_256 > 0.5, 1.0, 0.0)

        sky_fraction = np.mean(mask_256)
        print(f"  Sky fraction after masking: {sky_fraction:.1%}")

        masked_map = np.copy(smica_256)
        masked_map[mask_256 < 0.5] = hp.UNSEEN

        masked_map_uk, masked_vmin, masked_vmax = to_display_uk(masked_map)
        hp.mollview(
            masked_map_uk,
            title=f"SMICA Nside={nside_low} with galactic mask ({sky_fraction:.0%} sky)",
            unit="µK",
            cmap="RdBu_r",
            min=masked_vmin,
            max=masked_vmax,
        )
        path_masked = os.path.join(PLOT_DIR, "03_smica_masked.png")
        plt.savefig(path_masked, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path_masked}")
    except Exception as e:
        print(f"  Mask download/load failed ({e}), skipping mask step.")
        masked_map = smica_256

    # ── Step 6: Extract a gnomonic (flat-sky) patch ────────────────────
    print("\n=== Step 6: Extract gnomonic patch ===")
    # Cold Spot approximate location: (l, b) ~ (210, -57) in galactic coords
    # Convert to (theta, phi) in radians for healpy (colatitude, longitude)
    gal_lon_deg, gal_lat_deg = 210.0, -57.0
    theta = np.radians(90.0 - gal_lat_deg)
    phi = np.radians(gal_lon_deg)

    reso_arcmin = 13.0  # resolution per pixel in arcminutes
    patch_size = 256    # 256 x 256 pixels
    fov_deg = reso_arcmin * patch_size / 60.0  # ~55.5 degrees

    print(f"  Center: (l={gal_lon_deg}, b={gal_lat_deg}) deg")
    print(f"  Patch:  {patch_size}x{patch_size} pixels, {reso_arcmin}'/pixel")
    print(f"  FOV:    {fov_deg:.1f} degrees")

    patch = hp.gnomview(
        smica_256_uk,
        rot=(gal_lon_deg, gal_lat_deg),
        reso=reso_arcmin,
        xsize=patch_size,
        title=f"Gnomonic patch near Cold Spot (l={gal_lon_deg}, b={gal_lat_deg})",
        unit="µK",
        cmap="RdBu_r",
        min=low_vmin,
        max=low_vmax,
        return_projected_map=True,
    )
    path_patch = os.path.join(PLOT_DIR, "04_gnomonic_cold_spot.png")
    plt.savefig(path_patch, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path_patch}")
    print(f"  Patch shape: {patch.shape}")
    print(f"  Patch min: {np.nanmin(patch):.2f} µK, max: {np.nanmax(patch):.2f} µK")

    # Extract a second patch at a random high-latitude location for comparison
    gal_lon2, gal_lat2 = 45.0, 60.0
    patch2 = hp.gnomview(
        smica_256_uk,
        rot=(gal_lon2, gal_lat2),
        reso=reso_arcmin,
        xsize=patch_size,
        title=f"Gnomonic patch (l={gal_lon2}, b={gal_lat2})",
        unit="µK",
        cmap="RdBu_r",
        min=low_vmin,
        max=low_vmax,
        return_projected_map=True,
    )
    path_patch2 = os.path.join(PLOT_DIR, "05_gnomonic_highlat.png")
    plt.savefig(path_patch2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path_patch2}")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n=== Done! ===")
    print(f"  Plots saved to: {PLOT_DIR}")
    print(f"  Data saved to:  {DATA_DIR}")
    print("\n  Phase 1 checklist:")
    print("    [x] Downloaded Planck 2018 SMICA map")
    print("    [x] Loaded and inspected HEALPix map")
    print("    [x] Full-sky Mollweide projection")
    print("    [x] Degraded to Nside=256")
    print("    [x] Applied galactic mask")
    print("    [x] Extracted gnomonic (flat-sky) patches")
    print("    [ ] Explore coordinate systems further")
    print("    [ ] Download additional masks (KQ75, etc.)")


if __name__ == "__main__":
    main()
