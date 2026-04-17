# Stratified External Evaluation

Dataset: `/data/william/CMB-Collision-Bubbles/data/validation_stratified_mixed_geometry_v1/validation_data.h5`
Matched FPR: `0.08`

| model | AUROC | AUPRC | threshold | precision | recall | FPR | F1 | weak recall | contained recall | truncated recall | Dice+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6_aux_only | 0.727 [0.712,0.741] | 0.891 [0.883,0.899] | 0.979378 | 0.936 | 0.456 | 0.080 | 0.614 | 0.360 [0.342,0.380] | 0.562 | 0.206 | 0.308 [0.296,0.321] |
| v7_mixed_ft | 0.749 [0.735,0.763] | 0.900 [0.893,0.908] | 0.905132 | 0.940 | 0.486 | 0.080 | 0.640 | 0.382 [0.362,0.402] | 0.569 | 0.288 | 0.354 [0.341,0.367] |
