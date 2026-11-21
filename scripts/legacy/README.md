# Legacy Scripts

## Assumptions

- `remediated_v1` is the current science family.
- Files in this directory are retained for provenance, not for current
  paper-facing claims.
- Archived scripts may depend on top-level modules in `scripts/`; run them with
  `PYTHONPATH=scripts` from the repository root if you need to reproduce a
  historical artifact.

## Archived Entrypoints

- `phase3_operating_table.py`: legacy mixed operating score for branch triage.
- `phase3_final_policy_eval.py`: pre-remediation composite-policy surface.
- `phase3_eval_tta_ensemble.py`: legacy ensemble/TTA evaluation path.
- `phase3_nside512_probe.py`: historical Nside=512 resolution probe.
- `phase3_threshold_volume_sweep.py`: historical v6 threshold-volume sweep.
- `phase3_two_pass_policy.py`: historical loose-ML plus circular-template
  verifier policy.
- `phase3_visualize_smoothed_examples.py`: historical smoothed-injection
  gallery utility.
- `phase3_real_sky_smoothed_sensitivity.py`: historical v6_aux_only
  real-SMICA smoothed-edge diagnostic.
- `phase3_weak_family_breakdown.py`: historical weak-family recall breakdown.
- `phase3_screen_and_verify.py`: historical two-branch candidate packaging
  layer.

For current work, prefer the active scripts listed in
`docs/evaluation_inventory.md` and `docs/project_structure.md`.
