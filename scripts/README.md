# Scripts Directory

## Assumptions

- `remediated_v1` is the current science family.
- Root-level files are directory documentation and import bootstrap support.
- Current scripts are grouped by purpose under a small folder taxonomy.
- Provenance-only and pre-remediation command-line entrypoints live in
  `scripts/legacy/`.

## Current And Future-Use Source

- `core/`: shared physical constants, signal/observing-model code, method
  names, threshold helpers, and dataset geometry utilities.
- `data/`: Planck unit audits, synthetic-data generation, null-control
  extraction, dataset audits, and executable physics checks.
- `ml/`: model training and direct model-evaluation harnesses.
- `analysis/`: screening benchmarks, classical filters, sensitivity curves,
  policy searches, deployment burden, same-grid studies, and diagnostic
  analyses.
- `followup/`: candidate emission, score calibration, template fitting,
  projection-robust follow-up, frequency/HM jackknives, and paper handoff
  packets.
- `ops/`: quality gates, reproducibility manifests, artifact-flow audits, and
  operational batch launchers.
- `legacy/`: pre-remediation or provenance-only command-line entrypoints.

Each script folder has a small `sitecustomize.py` that makes the other script
folders importable when a nested CLI is run directly from the repository root.

See `docs/evaluation_inventory.md` and `docs/project_structure.md` for the
maintained source and artifact policy.

## Legacy Source

`scripts/legacy/` contains old CLIs that preserve pre-remediation decisions or
historical diagnostics. They are intentionally not part of the active science
path and should not be used for paper-facing claims unless labelled as
historical.
