# VeriFuse reproducible experiment report

This is a **controlled synthetic scene-consistency benchmark**, not evidence of real-world fake-news detection accuracy.

REAL is an unaltered procedural scene paired with its matching caption; FAKE is its paired version with one caption or image attribute altered to contradict the other modality. These labels do not assert real-world news truth.

## Dataset and protocol

Seed: 7031. 864 pairs, 432 source groups. Train/validation/test/OOD groups and decoded image pixels are disjoint across splits. TF-IDF and standardization fit only on train; validation chooses only the late-fusion blend. Test labels are never used in fitting.

| Split | Pairs | Groups | REAL | FAKE |
|---|---:|---:|---:|---:|
| train | 432 | 216 | 216 | 216 |
| val | 144 | 72 | 72 | 72 |
| test | 144 | 72 | 72 | 72 |
| ood_test | 144 | 72 | 72 | 72 |

## Six experiments

FAKE is the positive class. PR-AUC uses average precision with ties grouped. Binary metrics use 0.5; UI uncertainty handling is separate.

| Experiment | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Text-only baseline | 0.4792 | 0.4789 | 0.4722 | 0.4755 | 0.4809 | 0.4752 |
| Image-only baseline | 0.5069 | 0.5059 | 0.5972 | 0.5478 | 0.5017 | 0.5076 |
| Feature-level fusion | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision-level fusion | 0.5069 | 0.5059 | 0.5972 | 0.5478 | 0.5017 | 0.5076 |
| Modality ablation | 0.4861 | 0.4857 | 0.4722 | 0.4789 | 0.4778 | 0.4982 |
| Held-out manipulation | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Text-only baseline

Train-only unigram/bigram TF-IDF followed by L2 logistic regression.

Confusion matrix [[TN, FP], [FN, TP]]: `[[35, 37], [38, 34]]`.

### Image-only baseline

Pixel-derived color, shape, count, foreground fraction, fill and RGB features followed by L2 logistic regression.

Confusion matrix [[TN, FP], [FN, TP]]: `[[30, 42], [29, 43]]`.

### Feature-level fusion

Concatenated TF-IDF, image attributes and explicit cross-modal contradiction features followed by L2 logistic regression.

Confusion matrix [[TN, FP], [FN, TP]]: `[[72, 0], [0, 72]]`.

### Decision-level fusion

Independent text/image probabilities blended with a weight selected by validation log loss.

Confusion matrix [[TN, FP], [FN, TP]]: `[[30, 42], [29, 43]]`.

### Modality ablation

Fixed trained fusion model; missing feature families replaced by training means. This is an inference ablation, not retraining.

Confusion matrix [[TN, FP], [FN, TP]]: `[[36, 36], [38, 34]]`.

- Without Text: accuracy 0.4861, F1 0.4789, matrix [[36, 36], [38, 34]].

- Without Image: accuracy 0.4653, F1 0.4762, matrix [[32, 40], [37, 35]].

- Without Cross Modal: accuracy 0.5000, F1 0.5000, matrix [[36, 36], [36, 36]].

### Held-out manipulation

Count alterations never occur among train/validation/test negatives. OOD has entirely new scene groups and matching REAL controls. This is manipulation OOD within synthetic shapes, not real-photo OOD.

Confusion matrix [[TN, FP], [FN, TP]]: `[[72, 0], [0, 72]]`.

- text_only: accuracy 0.4861, F1 0.4861, matrix [[35, 37], [37, 35]].

- image_only: accuracy 0.5000, F1 0.5385, matrix [[30, 42], [30, 42]].

- feature_fusion: accuracy 1.0000, F1 1.0000, matrix [[72, 0], [0, 72]].

- late_fusion: accuracy 0.5000, F1 0.5385, matrix [[30, 42], [30, 42]].

## Explanation checks

On the first 48 fixed test records, delete each distinct caption token and recompute predictions; compare attribute-token effects with filler-token effects. Correct false captions and introduce a color conflict into matching captions. These measure local perturbation sensitivity, not causal truth or human explanation quality.

Mean absolute attribute-token effect: 0.2859; filler-token effect: 0.0006. Correcting false captions reduced FAKE probability by 0.9239 on average; expected direction rate 100.0%. Introducing color conflicts increased FAKE probability by 0.9268; direction rate 100.0%.

Each detected object rectangle is painted with the light background and rescored. Signed deltas are returned; occlusion can move an example off the training distribution.

Hold encoders fixed and counterfactually flip one attribute comparison; impact equals original minus changed FAKE probability. Baseline cross-modal impacts are zero.

## Low-margin case analysis

The 8 smallest-margin test pairs from each class (16 total), selected after all model fitting. These are relative low-confidence cases, not necessarily errors.

1. **scene-0298-fake** — truth FAKE, predicted FAKE, FAKE probability 0.9434. Caption: A picture of one yellow square. The feature-fusion model finds a caption/scene contradiction: color: caption says yellow, image shows green. This supports FAKE only under the controlled dataset's consistency definition.

2. **scene-0328-fake** — truth FAKE, predicted FAKE, FAKE probability 0.9496. Caption: This image shows one blue triangle. The feature-fusion model finds a caption/scene contradiction: color: caption says blue, image shows green. This supports FAKE only under the controlled dataset's consistency definition.

3. **scene-0358-fake** — truth FAKE, predicted FAKE, FAKE probability 0.9523. Caption: A picture of one blue square. The feature-fusion model finds a caption/scene contradiction: color: caption says blue, image shows green. This supports FAKE only under the controlled dataset's consistency definition.

4. **scene-0334-fake** — truth FAKE, predicted FAKE, FAKE probability 0.9526. Caption: Visible in the image: one blue square. The feature-fusion model finds a caption/scene contradiction: color: caption says blue, image shows green. This supports FAKE only under the controlled dataset's consistency definition.

5. **scene-0330-fake** — truth FAKE, predicted FAKE, FAKE probability 0.9532. Caption: The scene contains one yellow square on a light background. The feature-fusion model finds a caption/scene contradiction: color: caption says yellow, image shows green. This supports FAKE only under the controlled dataset's consistency definition.

6. **scene-0297-fake** — truth FAKE, predicted FAKE, FAKE probability 0.9537. Caption: A picture of one blue square. The feature-fusion model finds a caption/scene contradiction: shape: caption says square, image shows triangle. This supports FAKE only under the controlled dataset's consistency definition.

7. **scene-0288-fake** — truth FAKE, predicted FAKE, FAKE probability 0.9539. Caption: The scene contains one blue circle on a light background. The feature-fusion model finds a caption/scene contradiction: color: caption says blue, image shows yellow. This supports FAKE only under the controlled dataset's consistency definition.

8. **scene-0349-real** — truth REAL, predicted REAL, FAKE probability 0.0458. Caption: The scene contains one green triangle on a light background. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

9. **scene-0352-real** — truth REAL, predicted REAL, FAKE probability 0.0453. Caption: A picture of one red circle. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

10. **scene-0313-real** — truth REAL, predicted REAL, FAKE probability 0.0442. Caption: The scene contains two blue circles on a light background. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

11. **scene-0344-fake** — truth FAKE, predicted FAKE, FAKE probability 0.9558. Caption: Visible in the image: three yellow triangles. The feature-fusion model finds a caption/scene contradiction: color: caption says yellow, image shows red. This supports FAKE only under the controlled dataset's consistency definition.

12. **scene-0359-real** — truth REAL, predicted REAL, FAKE probability 0.0441. Caption: The scene contains two blue circles on a light background. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

13. **scene-0347-real** — truth REAL, predicted REAL, FAKE probability 0.0441. Caption: Visible in the image: one red circle. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

14. **scene-0355-real** — truth REAL, predicted REAL, FAKE probability 0.0439. Caption: This image shows one green square. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

15. **scene-0315-real** — truth REAL, predicted REAL, FAKE probability 0.0427. Caption: The scene contains one yellow triangle on a light background. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

16. **scene-0328-real** — truth REAL, predicted REAL, FAKE probability 0.0420. Caption: This image shows one green triangle. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

## Sanity checks

- supported_matching_pair: expected REAL; observed REAL; PASS.
- no_image: expected UNCERTAIN; observed UNCERTAIN; PASS.
- blank_image: expected UNCERTAIN; observed UNCERTAIN; PASS.
- noise_image: expected UNCERTAIN; observed UNCERTAIN; PASS.
- missing_caption: expected UNCERTAIN; observed UNCERTAIN; PASS.
- negation_not_supported: expected UNCERTAIN; observed UNCERTAIN; PASS.
- general_news_unsupported: expected UNCERTAIN; observed UNCERTAIN; PASS.
- deliberate_caption_mismatch: expected FAKE; observed FAKE; PASS.
- manipulated_image_pair: expected FAKE; observed FAKE; PASS.

## Limits and reproducibility

The fusion model receives explicit comparisons derived from image pixels and caption words. Its success demonstrates learning a narrow consistency task. OOD count manipulation remains within the same drawing domain. No pretrained vision-language network, external news dataset, human explanation study, probability calibration claim, or news benchmark result is implied. Single-modality baselines cannot reliably solve relational inconsistencies.

Run `python scripts/generate_dataset.py` and `python scripts/run_experiments.py`. All training hyperparameters, threshold curves, per-class metrics, group metadata, weights and raw results are saved locally. `docs/DATASET.md` documents importing an external CSV.

Dataset metadata SHA-256: `55d8cb6bbbbb93bf72a90062e0d75d5cdbb38f67a15b3859be3648358ce78baa`.
