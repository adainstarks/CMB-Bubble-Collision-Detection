"""Audit local Planck FITS headers, units, and temperature scales.

Assumptions
-----------
* This audit targets the local Planck PR3 component-separated temperature maps
  used by the candidate-screening pipeline, plus the common CMB mask.
* Component-separated CMB temperature maps are expected to store anisotropy
  fields in ``K_CMB`` rather than microkelvin or full thermodynamic temperature,
  consistent with the Planck PR3 products summarized by Planck Collaboration IV
  (2020), arXiv:1807.06208.
* The audit is a provenance and unit gate. It does not validate foreground
  cleanliness, beam transfer functions, or cosmological optimality.
* Physical scale checks use conservative anisotropy bounds: a full-sky cleaned
  CMB anisotropy map with ``|T| > 1 K`` is treated as non-physical for this
  pipeline and likely indicates a unit or product mismatch.
"""

from __future__ import annotations
# Bootstrap organized script folders for direct CLI execution.
import sys as _cmb_sys
from pathlib import Path as _cmb_Path

_cmb_scripts_root = _cmb_Path(__file__).resolve().parents[1]
_cmb_paths = [_cmb_scripts_root]
_cmb_paths.extend(
    sorted(
        _cmb_path
        for _cmb_path in _cmb_scripts_root.iterdir()
        if _cmb_path.is_dir() and _cmb_path.name != "__pycache__"
    )
)
for _cmb_path in reversed(_cmb_paths):
    _cmb_path_text = str(_cmb_path)
    if _cmb_path_text not in _cmb_sys.path:
        _cmb_sys.path.insert(0, _cmb_path_text)
del _cmb_Path, _cmb_sys, _cmb_scripts_root, _cmb_paths, _cmb_path, _cmb_path_text

import argparse
import json
import re
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT / "runs" / "phase3_unet" / "remediated_v1_planck_unit_audit.json"
)
PR3_RELEASE_TAG = "PR3_R3.00"
COMPONENT_MAPS = {
    "smica": {
        "filename": "COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "expected_tunit": "K_CMB",
        "product": "Planck 2018 SMICA component-separated CMB map",
        "map_release": PR3_RELEASE_TAG,
    },
    "nilc": {
        "filename": "COM_CMB_IQU-nilc_2048_R3.00_full.fits",
        "expected_tunit": "K_CMB",
        "product": "Planck 2018 NILC component-separated CMB map",
        "map_release": PR3_RELEASE_TAG,
    },
    "sevem": {
        "filename": "COM_CMB_IQU-sevem_2048_R3.00_full.fits",
        "expected_tunit": "K_CMB",
        "product": "Planck 2018 SEVEM component-separated CMB map",
        "map_release": PR3_RELEASE_TAG,
    },
    "commander": {
        "filename": "COM_CMB_IQU-commander_2048_R3.00_full.fits",
        "expected_tunit": "K_CMB",
        "product": "Planck 2018 Commander component-separated CMB map",
        "map_release": PR3_RELEASE_TAG,
    },
}
MASK_PRODUCT = {
    "filename": "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits",
    "product": "Planck 2018 common CMB mask",
    "map_release": PR3_RELEASE_TAG,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit local Planck FITS headers, units, and temperature scales "
            "for the component-separated PR3 maps used by this repo."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--skip-mask", action="store_true")
    return parser.parse_args()


def infer_map_release(path: Path) -> str:
    """Infer a coarse release tag from a Planck filename."""

    match = re.search(r"_R(\d+\.\d+)_", path.name)
    if match:
        return f"PR3_R{match.group(1)}"
    if "NPIPE" in path.name.upper():
        return "PR4_NPIPE_unclassified"
    return "unknown"


def normalize_header_value(value: Any) -> Any:
    """Normalize FITS header values to plain JSON-friendly Python types."""

    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    if isinstance(value, str):
        return value.strip()
    return value


def select_healpix_hdu(hdul: fits.HDUList) -> tuple[int, fits.BinTableHDU]:
    """Return the first HEALPix-like binary-table HDU with NSIDE metadata."""

    for index, hdu in enumerate(hdul):
        header = hdu.header
        if isinstance(hdu, fits.BinTableHDU) and header.get("NSIDE") is not None:
            return index, hdu
    raise ValueError("Could not find a HEALPix binary-table HDU with NSIDE metadata.")


def finite_stats(values_k: np.ndarray) -> dict[str, float]:
    """Return mean/min/max/RMS in both kelvin and microkelvin."""

    finite = np.asarray(values_k, dtype=np.float64)
    if finite.size == 0 or not np.all(np.isfinite(finite)):
        raise ValueError("Map statistics require a non-empty finite temperature vector.")
    values = finite * u.K
    rms = np.sqrt(np.mean(np.square(finite), dtype=np.float64)) * u.K
    return {
        "min_k": float(np.min(values).to_value(u.K)),
        "max_k": float(np.max(values).to_value(u.K)),
        "mean_k": float(np.mean(values).to_value(u.K)),
        "rms_k": float(rms.to_value(u.K)),
        "min_uk": float(np.min(values).to_value(u.uK)),
        "max_uk": float(np.max(values).to_value(u.uK)),
        "mean_uk": float(np.mean(values).to_value(u.uK)),
        "rms_uk": float(rms.to_value(u.uK)),
    }


def load_temperature_map(path: Path) -> np.ndarray:
    """Load the temperature field as float64 and drop unseen pixels."""

    sky = np.asarray(hp.read_map(str(path), field=0, dtype=np.float64), dtype=np.float64)
    finite = np.isfinite(sky) & (sky != hp.UNSEEN)
    if not np.any(finite):
        raise ValueError("Temperature field has no finite seen pixels.")
    return sky[finite]


def audit_component_map(
    name: str,
    path: Path,
    expected_tunit: str,
    product: str,
    map_release: str,
) -> dict[str, Any]:
    """Audit one component-separated temperature map."""

    failures: list[str] = []
    warnings: list[str] = []
    with fits.open(path, memmap=True) as hdul:
        hdu_index, hdu = select_healpix_hdu(hdul)
        header = hdu.header
        tunit1 = normalize_header_value(header.get("TUNIT1"))
        ordering = normalize_header_value(header.get("ORDERING"))
        nside = normalize_header_value(header.get("NSIDE"))
        tfields = normalize_header_value(header.get("TFIELDS"))
        bunit = normalize_header_value(header.get("BUNIT"))
        extname = normalize_header_value(header.get("EXTNAME"))

    values_k = load_temperature_map(path)
    stats = finite_stats(values_k)
    abs_max_k = max(abs(stats["min_k"]), abs(stats["max_k"]))

    if tunit1 != expected_tunit:
        failures.append(
            f"{name}: expected TUNIT1={expected_tunit!r}, found {tunit1!r}."
        )
    if ordering not in {"NESTED", "RING"}:
        failures.append(f"{name}: unexpected ORDERING={ordering!r}.")
    if int(nside) != 2048:
        failures.append(f"{name}: expected NSIDE=2048, found {nside!r}.")
    if abs_max_k > 1.0:
        failures.append(
            f"{name}: |T| max {abs_max_k:.6g} K exceeds the anisotropy-scale safety bound."
        )
    if stats["rms_k"] <= 0.0:
        failures.append(f"{name}: RMS is non-positive, which is non-physical.")
    if abs(stats["mean_k"]) > 1.0e-3:
        warnings.append(
            f"{name}: mean anisotropy {stats['mean_uk']:.3f} uK is larger than the usual "
            "few-hundred-uK scale; verify that a low-mode subtraction policy is intended."
        )

    return {
        "name": name,
        "kind": "component_map",
        "path": str(path),
        "product": product,
        "map_release": map_release or infer_map_release(path),
        "status": "pass" if not failures else "fail",
        "num_failures": len(failures),
        "num_warnings": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "header": {
            "hdu_index": int(hdu_index),
            "extname": extname,
            "nside": int(nside),
            "ordering": ordering,
            "tfields": int(tfields),
            "tunit1": tunit1,
            "tunit2": normalize_header_value(header.get("TUNIT2")),
            "tunit3": normalize_header_value(header.get("TUNIT3")),
            "bunit": bunit,
        },
        "stats": stats,
    }


def audit_mask(path: Path, product: str, map_release: str) -> dict[str, Any]:
    """Audit the common Planck mask header and value range."""

    failures: list[str] = []
    with fits.open(path, memmap=True) as hdul:
        hdu_index, hdu = select_healpix_hdu(hdul)
        header = hdu.header
        ordering = normalize_header_value(header.get("ORDERING"))
        nside = normalize_header_value(header.get("NSIDE"))
        tfields = normalize_header_value(header.get("TFIELDS"))
        extname = normalize_header_value(header.get("EXTNAME"))
        tunit1 = normalize_header_value(header.get("TUNIT1"))

    values = np.asarray(hp.read_map(str(path), field=0, dtype=np.float64), dtype=np.float64)
    finite = values[np.isfinite(values) & (values != hp.UNSEEN)]
    if finite.size == 0:
        failures.append("mask: no finite seen pixels were found.")
    if not np.all((finite >= 0.0) & (finite <= 1.0)):
        failures.append("mask: expected values in [0, 1] for an intensity/common mask product.")
    if ordering not in {"NESTED", "RING"}:
        failures.append(f"mask: unexpected ORDERING={ordering!r}.")
    if int(nside) != 2048:
        failures.append(f"mask: expected NSIDE=2048, found {nside!r}.")

    return {
        "name": "common_mask",
        "kind": "mask",
        "path": str(path),
        "product": product,
        "map_release": map_release or infer_map_release(path),
        "status": "pass" if not failures else "fail",
        "num_failures": len(failures),
        "failures": failures,
        "header": {
            "hdu_index": int(hdu_index),
            "extname": extname,
            "nside": int(nside),
            "ordering": ordering,
            "tfields": int(tfields),
            "tunit1": tunit1,
        },
        "stats": {
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
            "mean": float(np.mean(finite)) if finite.size else None,
            "sky_fraction_gt_0": float(np.mean(finite > 0.0)) if finite.size else None,
        },
    }


def audit_default_planck_inputs(data_dir: Path, *, include_mask: bool = True) -> dict[str, Any]:
    """Audit the local default Planck PR3 component maps and common mask."""

    file_reports: list[dict[str, Any]] = []
    failures: list[str] = []
    warnings: list[str] = []

    for name, info in COMPONENT_MAPS.items():
        path = data_dir / info["filename"]
        if not path.exists():
            failures.append(f"{name}: missing file {path}.")
            continue
        report = audit_component_map(
            name=name,
            path=path,
            expected_tunit=str(info["expected_tunit"]),
            product=str(info["product"]),
            map_release=str(info["map_release"]),
        )
        file_reports.append(report)
        failures.extend(report.get("failures", []))
        warnings.extend(report.get("warnings", []))

    if include_mask:
        mask_path = data_dir / MASK_PRODUCT["filename"]
        if not mask_path.exists():
            failures.append(f"common_mask: missing file {mask_path}.")
        else:
            report = audit_mask(
                mask_path,
                product=str(MASK_PRODUCT["product"]),
                map_release=str(MASK_PRODUCT["map_release"]),
            )
            file_reports.append(report)
            failures.extend(report.get("failures", []))

    return {
        "status": "pass" if not failures else "fail",
        "num_failures": len(failures),
        "num_warnings": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "data_dir": str(data_dir),
        "assumptions": [
            "Component-separated temperature maps must carry K_CMB units.",
            "The local cleaned-map products are expected to be Planck PR3 R3.00 at NSIDE=2048.",
            "Temperature anisotropy values above |T| = 1 K are treated as non-physical for this pipeline.",
        ],
        "files": file_reports,
    }


def run_default_audit(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    include_mask: bool = True,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Run the default audit and optionally write the JSON artifact."""

    report = audit_default_planck_inputs(Path(data_dir), include_mask=include_mask)
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    report = run_default_audit(
        data_dir=Path(args.data_dir).resolve(),
        include_mask=not args.skip_mask,
        output_json=Path(args.output_json).resolve(),
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
