# Classical Stratified External Evaluation

Dataset: `/data/william/CMB-Collision-Bubbles/data/validation_stratified_mixed_geometry_v1/validation_data.h5`
Matched FPR: `0.08`

| model | AUROC | AUPRC | threshold | precision | recall | FPR | F1 | weak recall | Dice+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matched_template | 0.691 [0.676,0.705] | 0.866 [0.857,0.875] | 72.225914 | 0.918 | 0.350 | 0.080 | 0.507 | 0.252 [0.234,0.271] | 0.237 [0.224,0.249] |
| centered_disc | 0.609 [0.592,0.625] | 0.815 [0.801,0.827] | 0.599734 | 0.877 | 0.222 | 0.080 | 0.354 | 0.165 [0.151,0.180] | 0.037 [0.033,0.040] |
