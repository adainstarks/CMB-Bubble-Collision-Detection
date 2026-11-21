# Pure ML Detector Preflight

## Assumptions

- Deployed pure-ML inputs are raw `patches` only.
- Classical response maps may be external comparators, not model inputs.
- Split overlap checks are row-namespace checks inside each source HDF5.

## Status

- `blocking_split_overlap_ok`: `True`
- `forbidden_deployed_inputs_count`: `2`
- `same_grid_source_rows_comparable_to_synthetic`: `False`

## Forbidden Feature Datasets

- `features/wiener_feeney_response`
- `features/wiener_feeney_response_smoke`

These datasets are allowed to exist in the HDF5 file for other experiments, but this track must not use them as deployed inputs.
