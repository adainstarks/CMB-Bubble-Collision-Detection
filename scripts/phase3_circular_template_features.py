"""Circular-template response feature maps for U-Net input channels.

Assumptions
-----------
* These features are circular-template response maps, not an optimal Wiener
  matched filter and not a Bayesian evidence calculation.
* Input patches are CMB anisotropies in Kelvin and are standardized per patch
  before convolution so the response channel emphasizes morphology rather than
  absolute temperature scale.
* The template bank follows the Feeney et al. linear-cap profile family used by
  the project patch-space classical screen.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import fftconvolve

from phase2_signal_model import PATCH_PIX
from phase3_sensitivity_curve import SIGN_QUADRANTS, make_feeney_template_kernel


def circular_template_kernels(
    radii_deg: tuple[float, ...],
    beam_fwhm_arcmin: float,
) -> np.ndarray:
    """Build a circular-template kernel bank for radius/sign quadrants."""

    if not radii_deg:
        raise ValueError("At least one radius is required.")
    if any(float(radius) <= 0.0 for radius in radii_deg):
        raise ValueError("Template radii must be positive.")
    if float(beam_fwhm_arcmin) < 0.0:
        raise ValueError("Beam FWHM must be non-negative.")
    kernels = [
        make_feeney_template_kernel(
            float(radius),
            float(z0_sign),
            float(zcrit_sign),
            float(beam_fwhm_arcmin),
        )
        for radius in radii_deg
        for z0_sign, zcrit_sign in SIGN_QUADRANTS
    ]
    out = np.stack(kernels, axis=0).astype(np.float32)
    if out.shape[1:] != (PATCH_PIX, PATCH_PIX):
        raise ValueError(f"Unexpected kernel shape {out.shape}; expected {PATCH_PIX}x{PATCH_PIX}.")
    if not np.all(np.isfinite(out)):
        raise ValueError("Non-finite circular-template kernel values.")
    return out


def standardize_patch_batch(patches: np.ndarray) -> np.ndarray:
    """Standardize a patch batch by per-patch mean and standard deviation."""

    work = np.asarray(patches, dtype=np.float64)
    if work.ndim != 3:
        raise ValueError(f"Expected patch batch with shape (N, H, W); found {work.shape}.")
    if work.shape[1:] != (PATCH_PIX, PATCH_PIX):
        raise ValueError(f"Expected {PATCH_PIX}x{PATCH_PIX} patches; found {work.shape[1:]}.")
    if not np.all(np.isfinite(work)):
        raise ValueError("Non-finite patch values cannot be standardized.")
    flat = work.reshape(work.shape[0], -1)
    mean = np.mean(flat, axis=1)[:, None, None]
    std = np.std(flat, axis=1)[:, None, None]
    std = np.where(std > 0.0, std, 1.0)
    return ((work - mean) / std).astype(np.float32)


def circular_template_response_maps_scipy(
    patches: np.ndarray,
    kernels: np.ndarray,
) -> np.ndarray:
    """Compute max-response circular-template feature maps with scipy FFTs."""

    patch_batch = standardize_patch_batch(patches)
    kernel_bank = np.asarray(kernels, dtype=np.float32)
    if kernel_bank.ndim != 3 or kernel_bank.shape[1:] != (PATCH_PIX, PATCH_PIX):
        raise ValueError(f"Unexpected kernel bank shape {kernel_bank.shape}.")
    best = np.full(patch_batch.shape, -np.inf, dtype=np.float32)
    for kernel in kernel_bank:
        response = fftconvolve(
            patch_batch,
            kernel[None, ::-1, ::-1],
            mode="same",
            axes=(-2, -1),
        ).astype(np.float32, copy=False)
        np.maximum(best, response, out=best)
    if not np.all(np.isfinite(best)):
        raise ValueError("Non-finite scipy circular-template feature maps.")
    return best


def prepare_circular_template_kernel_fft(
    kernels: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Prepare circular-template kernel FFTs for torch feature generation."""

    kernel_bank = np.asarray(kernels, dtype=np.float32)
    if kernel_bank.ndim != 3 or kernel_bank.shape[1:] != (PATCH_PIX, PATCH_PIX):
        raise ValueError(f"Unexpected kernel bank shape {kernel_bank.shape}.")
    full_shape = (2 * PATCH_PIX - 1, 2 * PATCH_PIX - 1)
    kernel_tensor = torch.zeros(
        (kernel_bank.shape[0], full_shape[0], full_shape[1]),
        dtype=torch.float32,
        device=device,
    )
    kernel_tensor[:, :PATCH_PIX, :PATCH_PIX] = torch.as_tensor(
        np.ascontiguousarray(kernel_bank[:, ::-1, ::-1]),
        dtype=torch.float32,
        device=device,
    )
    return torch.fft.rfft2(kernel_tensor, s=full_shape)


def circular_template_response_maps_torch(
    patches: np.ndarray,
    kernel_fft: torch.Tensor,
    *,
    kernel_chunk: int,
    device: torch.device,
) -> np.ndarray:
    """Compute max-response circular-template feature maps with torch FFTs."""

    if int(kernel_chunk) <= 0:
        raise ValueError("kernel_chunk must be positive.")
    full_shape = (2 * PATCH_PIX - 1, 2 * PATCH_PIX - 1)
    crop = (PATCH_PIX - 1) // 2
    patch_batch = standardize_patch_batch(patches)
    batch_tensor = torch.zeros(
        (patch_batch.shape[0], full_shape[0], full_shape[1]),
        dtype=torch.float32,
        device=device,
    )
    batch_tensor[:, :PATCH_PIX, :PATCH_PIX] = torch.as_tensor(
        np.ascontiguousarray(patch_batch),
        dtype=torch.float32,
        device=device,
    )
    batch_fft = torch.fft.rfft2(batch_tensor, s=full_shape)
    best = torch.full(
        (patch_batch.shape[0], PATCH_PIX, PATCH_PIX),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    for k0 in range(0, int(kernel_fft.shape[0]), int(kernel_chunk)):
        k1 = min(k0 + int(kernel_chunk), int(kernel_fft.shape[0]))
        conv = torch.fft.irfft2(
            batch_fft[:, None, :, :] * kernel_fft[None, k0:k1, :, :],
            s=full_shape,
        )
        same = conv[:, :, crop : crop + PATCH_PIX, crop : crop + PATCH_PIX]
        best = torch.maximum(best, torch.amax(same, dim=1))
        del conv, same
    out = best.detach().cpu().numpy()
    del batch_tensor, batch_fft, best
    if not np.all(np.isfinite(out)):
        raise ValueError("Non-finite torch circular-template feature maps.")
    return out.astype(np.float32, copy=False)
