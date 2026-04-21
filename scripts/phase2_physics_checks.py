"""
Executable physics checks for the Phase 2 bubble-collision signal model.

These checks are intentionally lightweight and deterministic. They validate the
core assumptions that must hold before generated data or model metrics are worth
interpreting:
    - Feeney Eq. 1 evaluates to z0 at the disc center
    - Feeney Eq. 1 evaluates to zcrit at the causal boundary
    - hard-window templates vanish outside the causal disc
    - Feeney full-temperature modulation is implemented exactly
    - McEwen first-order additive approximation differences are quantified
    - patch angular geometry is internally consistent
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phase_config import (
    INJECTION_CONVENTION_FEENEY2011,
    INJECTION_CONVENTION_MCEWEN2012,
)
from phase2_signal_model import (
    RESO_ARCMIN,
    T_CMB_K,
    add_fractional_signal_to_patch,
    bubble_collision_signal,
    inject_signal_into_patch,
)
from phase_dataset_utils import (
    DEFAULT_PATCH_PIX,
    make_angular_distance_grid,
    patch_center_pixel,
    patch_offsets_deg_to_pixel,
    pixel_to_patch_offsets_deg,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run deterministic physics checks for the Phase 2 signal model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def assert_close(name, actual, expected, atol=1e-12, rtol=1e-10):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: actual={actual!r}, expected={expected!r}, atol={atol}, rtol={rtol}")


def check_eq1_special_cases():
    cases = [
        {"z0": 0.0, "zcrit": 0.0, "theta_crit_deg": 5.0},
        {"z0": 5e-5, "zcrit": 0.0, "theta_crit_deg": 10.0},
        {"z0": 0.0, "zcrit": -3e-5, "theta_crit_deg": 25.0},
        {"z0": 4e-5, "zcrit": -2e-5, "theta_crit_deg": 5.0},
        {"z0": -4e-5, "zcrit": 2e-5, "theta_crit_deg": 25.0},
        {"z0": 3e-5, "zcrit": 3e-5, "theta_crit_deg": 12.0},
    ]

    for case in cases:
        theta_crit = np.radians(case["theta_crit_deg"])
        theta = np.asarray([0.0, theta_crit, min(theta_crit + np.radians(0.25), np.pi)])
        signal = bubble_collision_signal(theta, case["z0"], case["zcrit"], theta_crit, edge_sigma_deg=0.0)
        assert_close(f"center=z0 {case}", signal[0], case["z0"])
        assert_close(f"boundary=zcrit {case}", signal[1], case["zcrit"])
        assert_close(f"outside=0 {case}", signal[2], 0.0)

    return {"num_cases": len(cases)}


def check_smooth_window_bounds():
    theta_crit = np.radians(10.0)
    theta = np.linspace(0.0, np.radians(20.0), 4096)
    hard = bubble_collision_signal(theta, 5e-5, -2e-5, theta_crit, edge_sigma_deg=0.0)
    smooth = bubble_collision_signal(theta, 5e-5, -2e-5, theta_crit, edge_sigma_deg=0.5)

    if not np.all(np.isfinite(smooth)):
        raise AssertionError("smoothed signal contains non-finite values")
    if abs(float(hard[0]) - 5e-5) > 1e-12:
        raise AssertionError("hard signal center changed unexpectedly")
    if abs(float(smooth[0]) - 5e-5) > 1e-10:
        raise AssertionError("smoothed signal center changed too much far from boundary")

    return {
        "hard_max_abs": float(np.max(np.abs(hard))),
        "smooth_max_abs": float(np.max(np.abs(smooth))),
    }


def check_injection_conventions():
    y, x = np.mgrid[0:DEFAULT_PATCH_PIX, 0:DEFAULT_PATCH_PIX]
    patch = ((x - x.mean()) * 1.0e-7 + (y - y.mean()) * 0.5e-7).astype(np.float32)
    injected, signal = inject_signal_into_patch(
        patch,
        z0=4e-5,
        zcrit=-2e-5,
        theta_crit_deg=12.0,
        edge_sigma_deg=0.0,
        center_x_pix=130.0,
        center_y_pix=124.0,
    )

    patch64 = patch.astype(np.float64)
    expected = patch64 + signal * (T_CMB_K + patch64)
    assert_close("Feeney full-temperature modulation", injected, expected, atol=1e-12, rtol=1e-10)

    additive = add_fractional_signal_to_patch(
        patch64,
        signal,
        injection_convention=INJECTION_CONVENTION_MCEWEN2012,
    )
    delta = injected - additive
    expected_delta = signal * patch
    assert_close("Feeney-vs-additive cross term", delta, expected_delta, atol=1e-12, rtol=1e-10)

    if float(np.max(np.abs(delta))) <= 0.0:
        raise AssertionError("Feeney-vs-additive difference is unexpectedly zero")

    grid_report = check_injection_convention_grid()

    return {
        "default_injection_convention": INJECTION_CONVENTION_FEENEY2011,
        "matched_filter_approximation_convention": INJECTION_CONVENTION_MCEWEN2012,
        "max_abs_signal": float(np.max(np.abs(signal))),
        "max_abs_feeney_minus_additive_k": float(np.max(np.abs(delta))),
        "grid": grid_report,
    }


def check_multiplicative_injection():
    """Backward-compatible wrapper for older callers."""

    return check_injection_conventions()


def check_injection_convention_grid():
    """Quantify Feeney-vs-first-order additive differences on production scales."""

    y, x = np.mgrid[0:DEFAULT_PATCH_PIX, 0:DEFAULT_PATCH_PIX]
    patch = (
        np.sin(2.0 * np.pi * x / DEFAULT_PATCH_PIX)
        + 0.75 * np.cos(2.0 * np.pi * y / DEFAULT_PATCH_PIX)
        + 0.25 * np.sin(2.0 * np.pi * (x + y) / DEFAULT_PATCH_PIX)
    ).astype(np.float64)
    patch *= 1.0e-4 / max(float(np.std(patch)), 1.0e-30)

    amplitudes = (1.0e-6, 2.0e-6, 5.0e-6, 1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4)
    radii_deg = (5.0, 10.0, 15.0, 20.0, 25.0)
    zcrit_ratios = (0.0, 0.5, 1.0)
    worst = {
        "max_abs_cross_term_k": 0.0,
        "rms_cross_term_k": 0.0,
        "max_abs_cross_over_primary": 0.0,
        "rms_cross_over_primary": 0.0,
        "amplitude": None,
        "theta_crit_deg": None,
        "zcrit_ratio": None,
    }

    for amplitude in amplitudes:
        for theta_crit_deg in radii_deg:
            theta_grid = make_angular_distance_grid(
                DEFAULT_PATCH_PIX,
                RESO_ARCMIN,
                center_x_pix=patch_center_pixel(DEFAULT_PATCH_PIX),
                center_y_pix=patch_center_pixel(DEFAULT_PATCH_PIX),
            )
            for zcrit_ratio in zcrit_ratios:
                signal = bubble_collision_signal(
                    theta_grid,
                    amplitude,
                    amplitude * zcrit_ratio,
                    np.radians(theta_crit_deg),
                    edge_sigma_deg=0.0,
                )
                exact = add_fractional_signal_to_patch(
                    patch,
                    signal,
                    injection_convention=INJECTION_CONVENTION_FEENEY2011,
                )
                additive = add_fractional_signal_to_patch(
                    patch,
                    signal,
                    injection_convention=INJECTION_CONVENTION_MCEWEN2012,
                )
                cross = exact - additive
                primary = signal * T_CMB_K
                support = np.abs(primary) > 0.0
                if not bool(np.any(support)):
                    continue
                max_ratio = float(
                    np.max(np.abs(cross[support]))
                    / max(float(np.max(np.abs(primary[support]))), 1.0e-30)
                )
                rms_ratio = float(
                    np.sqrt(np.mean(cross[support] ** 2))
                    / max(float(np.sqrt(np.mean(primary[support] ** 2))), 1.0e-30)
                )
                max_cross = float(np.max(np.abs(cross[support])))
                rms_cross = float(np.sqrt(np.mean(cross[support] ** 2)))
                if max_ratio > float(worst["max_abs_cross_over_primary"]):
                    worst = {
                        "max_abs_cross_term_k": max_cross,
                        "rms_cross_term_k": rms_cross,
                        "max_abs_cross_over_primary": max_ratio,
                        "rms_cross_over_primary": rms_ratio,
                        "amplitude": float(amplitude),
                        "theta_crit_deg": float(theta_crit_deg),
                        "zcrit_ratio": float(zcrit_ratio),
                    }

    return {
        "patch_rms_k": float(np.std(patch)),
        "amplitudes": [float(value) for value in amplitudes],
        "radii_deg": [float(value) for value in radii_deg],
        "zcrit_ratios": [float(value) for value in zcrit_ratios],
        "worst_case": worst,
    }


def check_patch_geometry():
    center_x = 128.0
    center_y = 128.0
    theta = make_angular_distance_grid(DEFAULT_PATCH_PIX, RESO_ARCMIN, center_x_pix=center_x, center_y_pix=center_y)
    assert_close("angular distance at requested center pixel", theta[int(center_y), int(center_x)], 0.0)

    x_pix, y_pix = patch_offsets_deg_to_pixel(3.25, -7.5)
    dx_deg, dy_deg = pixel_to_patch_offsets_deg(x_pix, y_pix)
    assert_close("dx round trip", dx_deg, 3.25)
    assert_close("dy round trip", dy_deg, -7.5)

    center = (DEFAULT_PATCH_PIX - 1) / 2.0
    nominal_edge_deg = center * RESO_ARCMIN / 60.0
    nominal_corner_deg = np.sqrt(2.0) * nominal_edge_deg
    gnomonic_radial_scale_edge = 1.0 / np.cos(np.radians(nominal_edge_deg)) ** 2
    gnomonic_radial_scale_corner = 1.0 / np.cos(np.radians(nominal_corner_deg)) ** 2

    return {
        "nominal_center_to_edge_deg": float(nominal_edge_deg),
        "nominal_center_to_corner_deg": float(nominal_corner_deg),
        "gnomonic_radial_scale_edge": float(gnomonic_radial_scale_edge),
        "gnomonic_radial_scale_corner": float(gnomonic_radial_scale_corner),
    }


def main():
    args = parse_args()
    report = {
        "eq1_special_cases": check_eq1_special_cases(),
        "smooth_window": check_smooth_window_bounds(),
        "injection_conventions": check_injection_conventions(),
        "patch_geometry": check_patch_geometry(),
        "status": "pass",
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
