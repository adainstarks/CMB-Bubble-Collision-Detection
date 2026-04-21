# Project Literature Audit: CMB Bubble-Collision Screening

Date: 2026-04-21
Repo baseline inspected: `remediated_v1`
Primary local files inspected: `README.md`, `PROJECT_HANDOFF.md`, `RESEARCHREVIEW`, `scripts/phase_config.py`, `scripts/phase3_train_unet.py`, `scripts/phase3_eval_tta_ensemble.py`, `scripts/phase3_cache_matched_filter_channel.py`, `scripts/phase3_classical_filters.py`, `scripts/phase3_matched_filter_snr_curve.py`, `runs/phase3_unet/remediated_v1_matched_filter_snr/`, `runs/phase3_unet/remediated_v1_mf_channel_tile_audit/`, `work/tta_ensemble_eval.md`

## Assumptions First

- This repository is a candidate-screening pipeline, not yet a full Feeney/OSS-style Bayesian evidence pipeline.
- The current production signal family uses Feeney-style full-temperature modulation by default, while the classical matched-filter derivations in the literature use the McEwen first-order additive approximation.
- Bubble-collision detectability on temperature maps is dominated by low-ell CMB covariance and masking/systematics, not by white noise alone.
- A valid ML-versus-classical comparison must be made at matched false-positive burden on the same sky realizations and with the same masking/calibration policy.
- Any recommendation that changes the scientific claim must be backed either by a primary-source result or by a concrete repo-local validation path.
- When I call something `literature-supported`, I mean at least one of the provided papers directly supports it. When I call something `plausible`, I mean it is consistent with the literature and with this repo, but not yet demonstrated for this task.

## What I Verified Locally Before Drawing Conclusions

- The repo already contains a real spherical Wiener/Feeney matched-filter implementation in `scripts/phase3_classical_filters.py`.
- The repo already contains an ideal matched-filter SNR artifact in `runs/phase3_unet/remediated_v1_matched_filter_snr/`, and the true same-grid ML-versus-classical benchmark is now `complete` in `runs/phase3_unet/remediated_v1_same_grid_fullsky_manifest/`.
- The current so-called matched-filter input channel is not a Wiener matched filter. `scripts/phase3_cache_matched_filter_channel.py` explicitly caches a circular-template response map.
- Test-time augmentation is already implemented in `scripts/phase3_eval_tta_ensemble.py`, and the existing mixed-geometry report shows essentially no recall gain from D4 TTA.
- Hard-example emphasis already exists in partial form through `--hard-positive-mining-json` and `--snr-sample-weight-json`.
- The focal-loss ablation and the SNR-guided hard-example ablation are now complete, and neither changes the main scientific conclusion. The remediated true-Wiener two-stream retrain and its corrected full-cache sensitivity comparison are now complete.

## Implementation Status Update

- Same-grid benchmark status: `complete`.
  - Mean recall on the fixed stratified manifest is `0.3841` for `wiener_feeney_matched_filter`, `0.3517` for `imagenet_b64_aux`, `0.2994` for `random_b64_aux`, and `0.2734` for `smhw_screen`.
  - ImageNet beats Wiener in `17 / 35` raw cells, loses in `16 / 35`, and ties in `2`; under non-overlapping exact 95% CIs, ImageNet is better in `7` cells and worse in `8`.
- SNR-guided hard-example ablation status: `complete`.
  - Mean `P_det` delta versus baseline is `-0.00068`; this does not justify promotion.
- Focal-loss ablation status: `complete`.
  - Mean `P_det` delta versus baseline is `+0.00071`; the gain is mild and concentrated away from the lowest-amplitude regime.
- True-Wiener two-stream status: `complete`, with one corrected rerun.
  - The first sensitivity comparison was invalid because only the first `12000 / 33000` sensitivity rows had a populated Wiener auxiliary channel.
  - The full-cache rerun raises mean `P_det` from `0.34875` to `0.36839` overall (`+0.01964`), helps moderate/high-amplitude cells (`+0.03881`) and large-radius cells (`+0.02226`), but remains worse in the hardest low-amplitude subset (`-0.00592`).
  - `scripts/phase3_train_unet.py` now raises on non-finite auxiliary channels so this partial-cache failure mode cannot silently reoccur.
- Projection/clustering audit status: `complete`.
  - The audit artifact is `runs/phase3_unet/remediated_v1_projection_clustering_audit/projection_clustering_audit.json`.
  - The fixed `15 deg` clustering radius materially changes candidate volume, and `17 / 24` frozen candidates are already flagged for projection caution in the downstream Bayesian/template handoff.
- Template/Bayesian handoff status: `complete`.
  - `runs/phase3_unet/remediated_v1_bayesian_template_handoff/bayesian_template_handoff_summary.json` now merges screening, template-fit, and projection-caution metadata for the frozen `24` representatives.

# 1. Paper-by-Paper Deep Review

## 1.1 McEwen et al. 2012, `1202.2861`

Title: *Optimal filters for detecting cosmic bubble collisions*

What the paper actually establishes:

- The candidate-detection stage should use spherical matched filters built from the known axisymmetric bubble template.
- The observable template is an additive disc-like temperature perturbation of the form `ΔT_b(θ,φ) = [c0 + c1 cos θ] s(θ; θ_crit)`.
- For isotropic Gaussian CMB plus homogeneous noise, the matched filter is optimal in the maximum-SNR sense.
- Candidate detection for unknown radii should scan a bank of scales, threshold per scale, then merge duplicates across adjacent scales.
- On their benchmark, matched filters improve sensitivity relative to needlets.

What transfers directly to this repo:

- A matched-filter baseline is not optional if we want to say anything about whether the U-Net is near the information-theoretic ceiling for this template family.
- Radius-by-radius SNR and recall curves are the right baseline diagnostic, not patch RMS.
- The right scientific comparator is spherical and covariance-weighted, not a patch-space correlator.

What does not transfer cleanly:

- Their optimality claim assumes isotropic noise and a simpler noise model than real Planck component-separated maps.
- The paper is a detection-stage paper, not a segmentation paper. It does not by itself justify a two-stream neural architecture.

Immediate consequence for us:

- This is the strongest paper-level support for the professor's matched-filter baseline criticism.

## 1.2 Gobbetti and Kleban 2012, `1201.6380`

Title: *Analyzing Cosmic Bubble Collisions*

What the paper actually establishes:

- The late-time observable perturbation can often be reduced to a small-parameter family derived from the collision lightsheet.
- Thin-wall and free-passage reasoning both generically lead to a piecewise-linear or kink-like perturbation structure.
- Thick-wall effects smooth the edge and can alter the observable morphology.

What transfers directly:

- Our labels and classical templates should remain focused on simple cap-like profiles as first-order observables.
- Edge smoothing is physically meaningful. It is not just an augmentation trick.

What this warns against:

- Overtraining on one overly sharp morphological family.
- Treating filled-disc segmentation alone as the only physically relevant structure when much of the signal lives near the causal boundary.

Immediate consequence for us:

- Boundary-aware loss terms are physically defensible, but only as a mild emphasis, not as a redefinition of the target.
- Any matched-filter or auxiliary feature bank should include smoothed-edge variants if we move beyond a first-order baseline.

## 1.3 Johnson, Peiris, and Lehner 2012, `1112.4487`

Title: *Determining the outcome of cosmic bubble collisions in full General Relativity*

What the paper actually establishes:

- Full-GR collision outcomes depend on the scalar potential structure, especially barrier widths and inflationary regime.
- Free-passage and thin-wall approximations are useful but not universally exact.
- Observable perturbation amplitudes and signs depend more on the potential structure than on kinematics alone.

What transfers directly:

- A single phenomenological signal family is useful for first-pass screening, but we should not oversell it as covering all physically allowed morphologies.
- Strong performance on one injected family does not prove strong performance on the physically broader collision class.

Immediate consequence for us:

- Morphology/domain robustness is a scientific issue, not only a machine-learning issue.
- Curriculum or hard-example schemes should be framed around preserving sensitivity across amplitude and morphology families, not just making the network care more about weak amplitudes.

## 1.4 Czech et al. 2011, `1006.0832`

Title: *Polarizing Bubble Collisions*

What the paper actually establishes:

- Bubble collisions produce a distinctive, purely E-mode polarization pattern with strong symmetry constraints.
- Temperature-only searches are incomplete as a full-physics detection pipeline.

What transfers directly:

- Our current temperature-only pipeline should be framed as an initial screener, not as a complete collision-identification method.
- A future paper claiming detection significance would need polarization follow-up or an explicit discussion of why it is absent.

Immediate consequence for us:

- This is more a claim-limitation paper than a recall-improvement paper.

## 1.5 Lin et al. 2018, `1708.02002`

Title: *Focal Loss for Dense Object Detection*

What the paper actually establishes:

- Focal loss improves dense detection under severe class imbalance by down-weighting easy examples.
- The canonical RetinaNet defaults were `gamma = 2` and `alpha = 0.25`.

What transfers directly:

- Our segmentation masks are also extremely background-dominated at the pixel level.
- The mechanism is relevant: easy background pixels can swamp gradients.

What does not transfer directly:

- This is not a CMB paper and not a segmentation paper on projected HEALPix patches with Dice supervision.
- The exact `alpha = 0.25` choice is not literature-backed for this task because our imbalance regime and existing `pos_weight` usage are different.

Immediate consequence for us:

- Focal loss is worth ablation, but it is not yet a literature-backed conclusion.

## 1.6 Yan et al. 2023, `Yan_2023_ApJ_947_29`

Title: *Recovering Cosmic Microwave Background Polarization Signals with Machine Learning*

What the paper actually establishes:

- Physics-guided losses can improve CMB ML pipelines.
- Cross-split maps with shared sky signal and uncorrelated noise are a powerful way to suppress instrumental noise downstream.
- Patching and realistic sky/noise simulation can work, provided evaluation is aligned with the target observable.

What transfers directly:

- Half-split / half-mission validation is methodologically strong and already aligned with the repo's Phase 5 sign-flip null idea.
- Losses that encode physically relevant structure can help, but they must target the actual observable of interest.

What does not transfer directly:

- Their objective is map reconstruction and power-spectrum recovery, not localized anomaly segmentation.
- Their FFT-based loss is not obviously the right objective for bubble masks.

Immediate consequence for us:

- The paper strengthens the case for rigorous split-map validation, but it does not specifically justify focal loss, TTA, or curriculum.

## 1.7 Obasho et al. 2025, `2509.00139`

Title: *Deep Learning for CMB Foreground Removal and Beam Deconvolution: A U-Net GAN Approach*

What the paper actually establishes:

- Realistic simulation matters: beam asymmetry, scan pattern, anisotropic noise, and foreground complexity materially change performance.
- Patching is computationally practical, but patching artifacts and isotropy violations must be audited explicitly.

What transfers directly:

- Our sim-to-real gap is scientifically important and likely recall-limiting.
- Projection and patching artifacts deserve explicit burden/recovery tests, especially for large radii.

What does not transfer directly:

- GAN losses and reconstruction objectives are not directly relevant to bubble-collision screening.

Immediate consequence for us:

- This is strong support for investing in realism, not for changing the segmentation architecture in a vacuum.

## 1.8 Karkina et al. 2024, `2411.08079`

Title: *Application of Machine Learning Methods for Detecting Atypical Structures in Astronomical Maps*

What the paper actually establishes:

- Unsupervised anomaly-style methods can achieve high precision but low recall on rare astrophysical structures.

What transfers directly:

- If recall matters, generic anomaly detection is a poor primary strategy.

Immediate consequence for us:

- This paper argues against pivoting away from supervised or semi-supervised template-aware screening.

## 1.9 Ocampo and Cañas-Herrera 2026, `2604.05290`

Title: *Explaining Neural Networks on the Sky: Machine Learning Interpretability for CMB Maps*

What the paper actually establishes:

- Interpretability is necessary to rule out shortcut learning from masks or other non-physical artifacts.
- Robustness checks should include split hygiene, leakage checks, and explicit diagnostics for whether the network is using the intended signal.

What transfers directly:

- We should not accept any recall gain from a new model variant without checking that it is not just using projection or mask shortcuts.
- A saliency or attribution audit is not a vanity extra; it is a useful sanity check if a new channel or loss suddenly helps.

What does not transfer directly:

- Their pipeline is a global cosmological classifier, not a localized segmentation detector.
- The paper is partly a proof-of-concept and has placeholder-style results.

Immediate consequence for us:

- This is a strong argument for post-hoc interpretability on any future two-channel or focal-loss winner.

# 2. Cross-Paper Synthesis

## 2.1 Literature-Supported Findings

- A spherical matched-filter baseline is mandatory for this science target. Support: McEwen 2012.
- The first-order screening template is additive. Full-temperature Feeney modulation is physically meaningful, but the matched-filter formalism itself uses the first-order additive approximation. Support: McEwen 2012 plus Feeney-style convention separation already documented in the repo.
- Weak recall is not automatically a physics limit. If ideal matched-filter recall is high in a cell and ML recall is low, the failure is algorithmic or distributional. Support: McEwen 2012 plus the repo's own `remediated_v1_matched_filter_snr` artifact.
- Simulation realism matters and patch/projection artifacts matter. Support: Yan 2023, Obasho 2025.
- Temperature-only screening is incomplete as a physics claim. Support: Czech 2011.

## 2.2 Plausible But Not Yet Verified For This Repo

- A two-channel raw-plus-filtered input may improve recall by separating template extraction from segmentation.
- Focal loss may improve weak-signal recall by suppressing easy-background domination.
- Persistent hard-example emphasis may help more than a pure curriculum.
- Interpretability on a new model variant may expose projection or mask shortcuts before they contaminate paper claims.

These are all plausible. None is proven by the provided bubble-collision papers alone.

## 2.3 Confirmed Contradictions Or Mistakes Relative To Current Project Language

- Calling `features/circular_template_response` a matched filter is wrong. The code already warns about this, but the distinction needs to remain strict everywhere.
- Any remaining argument that patch RMS proves a physical limit is too strong. The current README is cautious, but this must remain enforced.
- The current `Nbar_s^95` number is weaker than Feeney 2013 and should not be framed as a competitive cosmological upper limit.
- TTA is not a missing idea. It is already implemented and its mixed-geometry recall benefit appears negligible in the current report.
- Hard-example emphasis is not fully missing. The repo already has SNR-informed and mined-hard-positive sampling hooks; what is missing is a disciplined ablation program.

# 3. Validation of Professor Suggestions

## 3.1 Matched Filter Baseline

Verdict: `valid and high priority`

Why it might help:

- It gives the physically relevant baseline for a known axisymmetric signal family.
- It tells us which weak-recall cells are truly low-SNR and which ones are failures of feature extraction, optimization, or simulation mismatch.

Evidence:

- McEwen 2012 is the primary support.
- Local artifact: in `runs/phase3_unet/remediated_v1_matched_filter_snr/matched_filter_snr_report.json`, the cell `A=1e-5`, `theta=15 deg` has median ideal fsky-scaled recall `0.834`, while current ImageNet U-Net recall is only `0.106`.

What could go wrong:

- An ideal isotropic-Gaussian matched filter is still not the full masked-sky optimum on real component-separated maps.
- A strong matched-filter result on ideal sims can still overestimate practical performance on real residuals.

Exactly how to test it:

1. Keep `scripts/phase3_matched_filter_snr_curve.py` as the ideal ceiling diagnostic.
2. Close the true same-grid benchmark with `scripts/phase3_same_grid_fullsky_benchmark.py` so ML and spherical Wiener/SMHW methods are evaluated on the same injected full-sky realizations and same null calibration.
3. Report recall/FPR and burden at matched operating points.

What counts as success:

- We can cleanly partition cells into:
  - low ideal recall and low ML recall: genuinely hard under current assumptions;
  - high ideal recall and low ML recall: algorithmic/domain-gap targets.

What counts as failure:

- We still compare ML against patch RMS or against a misnamed patch-space "matched filter."

Specific professor example:

- The exact example `MF SNR ~ 7 at A = 5e-5, r = 15 deg` does not match the current repo artifact. Under the current `remediated_v1` assumptions, that cell has median full-sky SNR about `17.3`, fsky-scaled SNR about `15.2`, and ImageNet recall about `0.928`.
- The general logic is correct: if a cell has high matched-filter SNR and the U-Net misses it, that is a feature-extraction or distribution failure, not a noise-limited failure.
- A better current example is `A = 1e-5, theta = 15 deg`, where ideal recall is high and ML recall is still poor.

## 3.2 Filtered-Map Or Two-Stream Input

Verdict: `plausible, worth testing, but not yet literature-proven for this repo`

Why it might help:

- The network may currently be spending capacity both on learning a template-matching front end and on the segmentation decision.
- A filtered channel could hand the model a physics-informed candidate map and let the learned branch focus on morphology and nuisance rejection.

Evidence:

- McEwen 2012 strongly supports the value of matched-filtered responses for candidate detection.
- Local repo evidence: the legacy two-channel checkpoint audited in `runs/phase3_unet/remediated_v1_mf_channel_tile_audit/` reaches diagnostic real-SMICA recall `0.3526` at FPR `0.044`, better than the current deployment-safe composite recall. But that second channel is a legacy circular-template proxy, not a true Wiener channel, and the training setup is not remediated-v1 clean.

What could go wrong:

- A mismatched filtered channel can inject its own biases and false positives.
- If the channel is computed patch-wise rather than spherically, it may encode projection artifacts rather than true matched-filter evidence.
- A high-performing two-channel model may just overfit residual structure in the auxiliary channel.

Exactly how to test it:

1. Build a remediated-v1 extra channel from a true Wiener/Feeney spherical score map, not from the current circular-template proxy.
2. Train `raw-only` versus `raw + true_wiener_channel` with identical splits, augmentation, and calibration.
3. Compare:
  - matched-FPR recall on synthetic and real-SMICA injection,
  - full-sky tile burden on SMICA/NILC/SEVEM/Commander,
  - calibrated candidate burden after clustering,
  - saliency/interpretability sanity checks.

What counts as success:

- Higher recall at the same calibrated FPR and no unacceptable increase in full-sky candidate burden.

What counts as failure:

- Recall improves only on clean synthetic validation but collapses under real-null calibration or candidate-volume constraints.

Local implementation status:

- The training harness already supports extra channels.
- The missing piece is a scientifically correct remediated feature cache for a true spherical Wiener channel.

## 3.3 Test-Time Augmentation

Verdict: `already implemented, scientifically acceptable, low expected impact`

Why it might help:

- True bubble signatures are approximately rotationally symmetric in local tangent patches, while some nuisance artifacts may not be stable under D4 transforms.

Evidence:

- Local repo evidence is stronger than the literature here:
  - `scripts/phase3_eval_tta_ensemble.py` implements D4 TTA.
  - `work/tta_ensemble_eval.md` shows mixed-geometry recall `0.486` with and without D4 TTA for the current mixed-geometry model, with only a small Dice improvement.

What could go wrong:

- On gnomonic patches, exact symmetry is broken for large radii and for heavily truncated edge cases.
- TTA can slightly wash out sharp boundaries or shift calibration, so thresholds must be reselected after applying it.

Exactly how to test it:

1. Keep the mixed-geometry validation comparison already done.
2. Add one more deployment-facing audit: no-TTA versus D4-TTA full-sky tile burden at recalibrated thresholds.

What counts as success:

- Same or better recall at matched FPR with lower clustered candidate burden or cleaner calibration.

What counts as failure:

- No recall gain and no burden reduction. In that case TTA is only a cosmetic Dice stabilizer.

Current conclusion:

- TTA is appropriate, but it is not one of the main missing recall levers.

## 3.4 Focal Loss

Verdict: `plausible and now implementable as a clean ablation`

Why it might help:

- The segmentation target is dominated by easy background pixels.
- Current BCE-plus-Dice may still devote too much gradient budget to background that is already correctly classified.

Evidence:

- Transfer evidence only: Lin et al. 2018.
- No provided bubble-collision paper directly supports focal loss for this task.

What could go wrong:

- With our existing `pos_weight`, naive focal settings can over-focus on ambiguous noise and hurt calibration or precision.
- `alpha = 0.25` is not obviously correct here because our positive-pixel fraction and class weighting differ from RetinaNet.

Exactly how to test it:

1. Use the new `scripts/phase3_train_unet.py` focal-loss option.
2. Compare:
  - baseline BCE+Dice;
  - focal+Dice with `gamma = 2`, `alpha disabled`;
  - focal+Dice with `gamma = 2`, `alpha = 0.25`.
3. Evaluate on:
  - mixed-geometry validation,
  - real-SMICA injection,
  - tile-recalibrated full-sky burden.

What counts as success:

- Improved recall in the contested weak-to-mid regime, especially `A <= 2e-5`, at matched FPR with acceptable precision/burden.

What counts as failure:

- Any gain in weak-signal recall is offset by calibration collapse or large candidate-volume inflation.

Starting hyperparameter guidance:

- `gamma = 2` is a reasonable first ablation.
- `alpha = 0.25` should be treated as a transfer baseline only, not a default scientific choice.
- Because the repo already uses `pos_weight`, the most defensible first run is `gamma = 2`, `alpha disabled`.

## 3.5 Curriculum Learning / Hard-Example Emphasis

Verdict: `persistent hard-example emphasis is more justified than pure low-SNR-first curriculum`

Why it might help:

- Weak and truncated cases are currently the hard regime.
- The model may need sustained exposure to those cases rather than hoping they are absorbed late in training.

Evidence:

- Direct evidence from the provided literature is thin.
- Local repo evidence matters more here: `phase3_train_unet.py` already has hard-positive mining and matched-filter-SNR-informed sampling, which is exactly the kind of persistent hard-example emphasis the professor is asking about.

What could go wrong:

- A strict low-SNR-first curriculum can make the network memorize noise-like cases before it has learned the basic morphology of easy positives.
- Over-upweighting hard bins can degrade calibration and precision by effectively changing the training prior too aggressively.

Exactly how to test it:

1. Baseline: current mixed-geometry training.
2. Persistent hard-example arm: enable `--snr-sample-weight-json` with `algorithmic_gap` and a moderate strength.
3. Hard-positive mined arm: use `--hard-positive-mining-json`.
4. Optional curriculum arm: if implemented later, do only a late fine-tune schedule, not a pure low-SNR-from-scratch run.

What counts as success:

- Weak-regime recall improves without materially harming contained strong-signal recall or deployment burden.

What counts as failure:

- Gains appear only on synthetic validation but disappear after real-null calibration.

Current conclusion:

- The professor's instinct is directionally right, but the best immediate experiment is not a pure curriculum. It is persistent hard-example sampling using the infrastructure the repo already has.

# 4. Contradictions With the Current Paper/Project

## 4.1 Confirmed

- `features/circular_template_response` is not a matched filter. Any paper language implying otherwise is wrong.
- The same-grid Wiener/SMHW benchmark is now closed and does not support a uniform ML-over-classical claim. Wiener is the strongest average screener, while ML remains locally competitive.
- The present `Nbar_s^95` result is a candidate-screening sensitivity number and is weaker than the Feeney 2013 WMAP benchmark.
- TTA is not missing. It has already been tested and appears low impact on the hard benchmark.

## 4.2 Partly Confirmed

- The matched-filter baseline criticism is valid, but the repo is not empty on this front. The spherical filter code and the ideal SNR artifact already exist. What is missing is same-grid closure and paper-facing discipline.
- The injection criticism is valid as a documentation issue, not as a confirmed gross-bug claim. The code distinguishes Feeney-style full-temperature modulation from McEwen-style first-order additive filtering, but the paper needs to say this clearly.

## 4.3 Likely Misframing Risks In A Paper

- Overstating low recall as a physics limit when some same-grid cells still show large matched-filter headroom.
- Understating projection/systematics risk at large radii.
- Treating cross-map agreement among SMICA/NILC/SEVEM/Commander as stronger independence than it really is.
- Treating a temperature-only screening result as if it were a fully physically discriminating bubble-collision search.

# 5. Concrete Experimental Roadmap

## 5.1 Immediate, Highest-Confidence Work

1. Close the same-grid classical benchmark.
   Why: this is the main scientific credibility gap.
   Risk: expensive, and still not the full OSS optimum.
   Success: a same-grid table with ML, Wiener/Feeney, and SMHW at matched calibration.

2. Run focal-loss ablations on the current mixed-geometry recipe.
   Why: cheap, clean, and now implemented.
   Risk: may worsen calibration.
   Success: improved weak-to-mid amplitude recall at matched burden.

3. Run persistent hard-example emphasis ablations using the existing SNR-informed weighting.
   Why: the code already supports it and it directly targets the observed algorithmic-gap cells.
   Risk: distribution shift in the training prior can inflate false positives.
   Success: improved contested-regime recall after recalibration, not just before.

## 5.2 Next, High-Value But Heavier Work

4. Build a true spherical Wiener-channel cache and retrain a remediated-v1 two-channel model.
   Why: this is the strongest architecture-side hypothesis with actual local precedent.
   Risk: auxiliary channel may encode systematics or projection mismatch.
   Success: recall gain survives tile burden and candidate calibration.

5. Add a deployment-facing TTA burden audit.
   Why: current evidence says TTA is low impact, but the one remaining plausible benefit is slightly cleaner full-sky burden.
   Risk: probably not worth much.
   Success: same recall, fewer clustered null candidates.

6. Quantify projection systematics by radius and distance from patch center.
   Why: physically important for large-radius conclusions.
   Risk: may force us to narrow the operational radius range.
   Success: explicit safe zone or explicit correction policy.

## 5.3 Later, Paper-Strengthening Work

7. Add interpretability on any winning new variant.
   Why: needed to rule out shortcut learning from mask/projection artifacts.
   Risk: extra analysis burden.
   Success: attributions concentrate on physically plausible disc/boundary regions, not mask edges.

8. Build the Bayesian or template-fit follow-up handoff for screened candidates.
   Why: required before any cosmological interpretation.
   Risk: significant new analysis stage.
   Success: candidate-screening result is cleanly separated from posterior inference.

## 5.4 Runnable Commands for Immediate Ablations

Focal-loss ablation, same baseline as current mixed-geometry training except for the pixel loss:

```bash
python scripts/phase3_train_unet.py \
  --data-h5 data/remediated_v1/training_data.h5 \
  --run-name remediated_v1_unet_imagenet_b64_aux_focal_g2 \
  --encoder-name efficientnet-b0 \
  --encoder-weights imagenet \
  --batch-size 64 \
  --pixel-loss focal \
  --focal-gamma 2.0 \
  --focal-alpha -1 \
  --aux-head-weight 0.2
```

Persistent hard-example emphasis using the existing matched-filter headroom artifact:

```bash
python scripts/phase3_train_unet.py \
  --data-h5 data/remediated_v1/training_data.h5 \
  --run-name remediated_v1_unet_imagenet_b64_aux_snr_gap \
  --encoder-name efficientnet-b0 \
  --encoder-weights imagenet \
  --batch-size 64 \
  --aux-head-weight 0.2 \
  --snr-sample-weight-json runs/phase3_unet/remediated_v1_matched_filter_snr/matched_filter_snr_report.json \
  --snr-sample-weight-method algorithmic_gap \
  --snr-sample-weight-ml-method imagenet_b64_aux \
  --snr-sample-weight-strength 2.0 \
  --snr-sample-weight-max 4.0
```

Checkpoint-based focal fine-tune from the current best ImageNet baseline:

```bash
python scripts/phase3_train_unet.py \
  --data-h5 data/remediated_v1/training_data.h5 \
  --run-name remediated_v1_unet_imagenet_b64_aux_focal_g2_ft \
  --epochs 10 \
  --batch-size 64 \
  --learning-rate 5e-5 \
  --scheduler cosine \
  --encoder-name efficientnet-b0 \
  --encoder-weights imagenet \
  --pixel-loss focal \
  --focal-gamma 2.0 \
  --focal-alpha -1 \
  --aux-head-weight 0.2 \
  --normalization-config "${PWD}/runs/phase3_unet/remediated_v1_unet_imagenet_b64_aux/run_config.json" \
  --resume-checkpoint "${PWD}/runs/phase3_unet/remediated_v1_unet_imagenet_b64_aux/best_checkpoint.pt" \
  --model-only-resume \
  --cache-data \
  --skip-data-audit
```

# 6. Ranked Recommendations by Expected Impact on Recall and Scientific Credibility

## 6.1 Rank 1: Close the same-grid Wiener/SMHW benchmark

Why it might help:

- It may not directly improve recall, but it is the single most important check on whether the current ML misses are scientifically meaningful.

Evidence:

- Strongest support from McEwen 2012.
- The repo already has the core code but not the closed benchmark.

What could go wrong:

- It may show that some current ML gains disappear relative to the classical baseline.

Test:

- Full same-grid evaluation with matched calibration and burden accounting.

Success or failure:

- Success means we know exactly where ML has headroom or advantage.
- Failure means the paper still rests on an incomplete comparator.

## 6.2 Rank 2: Use the matched-filter headroom report to drive training ablations

Why it might help:

- It directly targets cells where the model is underperforming relative to a template-aware baseline.

Evidence:

- The current artifact already identifies large algorithmic gaps, for example `A=1e-5`, `theta=15 deg`.

What could go wrong:

- Over-targeting those cells may distort calibration.

Test:

- Persistent SNR-informed sampling versus baseline, evaluated after recalibration.

Success or failure:

- Success is recall gain concentrated in the previously high-headroom cells.
- Failure is synthetic-only gain with no deployment benefit.

## 6.3 Rank 3: Focal-loss ablation

Why it might help:

- It is the cleanest low-engineering-cost way to change the gradient allocation toward hard positives.

Evidence:

- Transfer support from Lin et al. 2018.
- Scientific support is weaker than for the matched-filter items, so it should stay an ablation.

What could go wrong:

- Calibration degradation and noisy over-focus on ambiguous background.

Test:

- BCE+Dice versus focal+Dice at matched operating points.

Success or failure:

- Success is higher weak-regime recall without blowing up burden.
- Failure is no gain or worse calibration.

## 6.4 Rank 4: True two-channel raw-plus-Wiener retrain

Why it might help:

- This is the most plausible architecture-side recall improvement supported by both the literature and the repo's legacy diagnostic.

Evidence:

- McEwen supports the value of the filtered response.
- The repo's legacy two-channel result suggests auxiliary prior information can help.

What could go wrong:

- Channel mismatch, projection leakage, or full-sky burden inflation.

Test:

- Remediated-v1 two-channel retrain with a true spherical Wiener channel and full deployment gating.

Success or failure:

- Success is a recall lift that survives burden and calibration.
- Failure is a synthetic-only or legacy-only improvement.

## 6.5 Rank 5: Keep TTA as a minor stabilizer, not a headline fix

Why it might help:

- It is cheap and physically defensible.

Evidence:

- Existing repo result: essentially no mixed-geometry recall lift, small Dice gain.

What could go wrong:

- Nothing dramatic; the likely failure mode is simply no meaningful benefit.

Test:

- One more full-sky burden audit if desired.

Success or failure:

- Success is cleaner burden at equal recall.
- Failure is neutral behavior, which is acceptable because the cost is low.

## 6.6 Rank 6: Do not replace persistent hard-example sampling with a pure low-SNR curriculum

Why:

- The literature support is weak, and the repo already has a better-targeted mechanism.

Evidence:

- Current code already exposes the more defensible variant: persistent emphasis rather than a hard curriculum reset.

What could go wrong:

- A pure curriculum can make the model memorize noise-like weak cases before it learns the morphological family.

Test:

- If tried at all, compare it only as a controlled late-stage fine-tune, not as a new default.

Success or failure:

- Success would require real-null calibrated gains, not just better training loss.

# Bottom Line

- The professor's strongest suggestion is the matched-filter baseline. That is valid, literature-backed, and partially implemented already.
- The best immediate recall experiments are not TTA or a pure curriculum. They are:
  - same-grid matched-filter closure,
  - SNR-guided hard-example emphasis,
  - focal-loss ablation,
  - then a remediated true-Wiener two-channel retrain.
- The highest-risk scientific misframing is still the same one: treating low recall as a physics limit before the classical baseline is fully closed.
