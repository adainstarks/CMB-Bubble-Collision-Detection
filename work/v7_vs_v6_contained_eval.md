# Stratified External Evaluation

Dataset: `/data/william/CMB-Collision-Bubbles/data/validation_stratified_v1/validation_data.h5`
Matched FPR: `0.08`

| model | AUROC | AUPRC | threshold | precision | recall | FPR | F1 | weak recall | contained recall | truncated recall | Dice+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6_aux_only | 0.773 [0.761,0.785] | 0.913 [0.907,0.920] | 0.982702 | 0.945 | 0.539 | 0.080 | 0.687 | 0.422 [0.403,0.443] | 0.539 | nan | 0.401 [0.388,0.417] |
| v7_mixed_ft | 0.779 [0.767,0.791] | 0.916 [0.910,0.922] | 0.905811 | 0.947 | 0.556 | 0.080 | 0.701 | 0.437 [0.417,0.457] | 0.556 | nan | 0.443 [0.428,0.457] |
