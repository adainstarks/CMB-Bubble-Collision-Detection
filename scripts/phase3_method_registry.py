"""Method-name registry for Phase 3 screening reports.

Assumptions
-----------
* `matched_filter` is reserved for filters that use a Feeney template with
  beam and CMB/noise inverse-covariance weighting, following McEwen et al.
  (2012) and Osborne/Senatore/Smith (2013).
* The historical disc/ring spatial correlator is a circular template screen,
  not a Wiener or optimal matched filter.
"""

from __future__ import annotations


CIRCULAR_TEMPLATE_SCREEN = "circular_template_screen"
WIENER_FEENEY_MATCHED_FILTER = "wiener_feeney_matched_filter"
SMHW_SCREEN = "smhw_screen"

LEGACY_METHOD_ALIASES = {
    "matched_template": CIRCULAR_TEMPLATE_SCREEN,
}


def canonical_method_name(name: str) -> str:
    """Return the canonical method name for a possibly legacy identifier."""

    return LEGACY_METHOD_ALIASES.get(str(name), str(name))


def method_metadata(name: str) -> dict:
    """Return machine-readable naming metadata for reports."""

    canonical = canonical_method_name(name)
    metadata = {
        "requested_name": str(name),
        "canonical_name": canonical,
        "is_legacy_alias": canonical != str(name),
    }
    if canonical == CIRCULAR_TEMPLATE_SCREEN:
        metadata.update(
            {
                "method_family": "classical_circular_template_correlation",
                "is_wiener_matched_filter": False,
                "naming_warning": (
                    "Historical `matched_template` outputs are circular-template "
                    "correlation screens, not CMB/noise-whitened Feeney matched filters."
                ),
            }
        )
    elif canonical == WIENER_FEENEY_MATCHED_FILTER:
        metadata.update(
            {
                "method_family": "classical_wiener_feeney_matched_filter",
                "is_wiener_matched_filter": True,
                "implemented_in": "scripts/phase3_classical_filters.py",
                "references": [
                    "https://arxiv.org/abs/1012.3667",
                    "https://arxiv.org/abs/1202.2861",
                    "https://arxiv.org/abs/1305.1970",
                ],
            }
        )
    elif canonical == SMHW_SCREEN:
        metadata.update(
            {
                "method_family": "classical_wavelet_screen",
                "is_wiener_matched_filter": False,
                "implemented_in": "scripts/phase3_classical_filters.py",
                "references": ["https://arxiv.org/abs/astro-ph/0212578"],
            }
        )
    return metadata
