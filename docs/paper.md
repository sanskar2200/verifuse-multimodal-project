# VeriFuse Multimodal Content Verification with Transparent Caption Image Consistency Models

CSET 346 course project, Bennett University
Student names and IDs: [complete before submission]

Abstract - Multimodal misinformation can contain a plausible caption and a plausible image whose meanings disagree. This course project studies that relation in a reproducible controlled setting. VeriFuse contains 864 balanced synthetic image-caption pairs derived from 432 source scenes. Grouped splits prevent an original scene and its manipulated variant from crossing evaluation boundaries. We compare train-only TF-IDF and pixel-attribute logistic baselines, feature fusion with explicit cross-modal interactions, and validation-selected late fusion. On the 144-pair in-distribution test set, feature fusion achieves accuracy 1.000 and F1 1.000, compared with F1 0.476 for text and 0.548 for image alone. A separate 144-pair count-manipulation test produces F1 1.000. Modality ablations, 16 difficult cases, grounded feature contributions, and perturbation checks expose what the model uses and where its scope ends. These measurements concern constructed scene-caption consistency. They do not establish natural-image deepfake detection, news truth, or real-world calibration.

Index Terms - multimodal NLP, image caption consistency, feature fusion, explainability, controlled benchmark.

## I Introduction

The CSET 346 project asks whether combining language and visual evidence improves detection and supports useful explanations. A caption may describe the wrong object even when neither its language nor its image looks implausible in isolation. This relational problem motivates a joint model: the label depends on agreement across modalities, so a classifier receiving only one modality may lack the necessary information.

VeriFuse implements the text-plus-image option as a small, inspectable research prototype. The task predicts REAL for an unaltered procedural scene paired with a correct caption and FAKE when one attribute in the image or caption deliberately contradicts the other. The interactive system also returns UNCERTAIN for unsupported inputs. The label vocabulary follows the course brief, but REAL does not certify an actual event and FAKE does not imply deceptive intent.

The contribution is an end-to-end reproducible experiment with two single-modality baselines, two fusion strategies, a withheld manipulation type, missing-modality tests, and explanations tied to model features. The experiment deliberately uses a simple domain to make each source of evidence measurable. Its high in-domain scores, when present, must be read in that context.

## II Background and Task Boundary

Fakeddit provides a multimodal benchmark for fine-grained fake-news research [1]. FaceForensics++ studies manipulated facial imagery [2]. These datasets illustrate distinct evaluation targets: caption-context agreement and image manipulation artifacts need not be the same problem. CLIP learns transferable visual representations using natural-language supervision [3], making pretrained vision-language encoders a plausible future comparison. VeriFuse does not download or train on these datasets and does not use CLIP or another pretrained encoder. The citations provide background rather than benchmark equivalence.

Our chosen scope is controlled cross-modal contradiction, which the course brief permits through carefully constructed data and misleading captions. We exclude faces, identity judgments, lip synchronization, audio, source authentication, and external factual retrieval. A system that agrees with a caption can still be wrong about the world. This distinction guides both the UI wording and interpretation of the metrics.

## III Dataset and Preprocessing

The generator creates 256 by 256 pixel RGB scenes using four colors (red, blue, green, yellow), three shapes (circle, square, triangle), and counts of one to three objects. All 36 combinations recur within each split. Small deterministic changes in position, size, foreground color, and background color create source scenes. Rendering at three times the final resolution followed by downsampling produces anti-aliased boundaries. Four caption templates vary surface wording without changing the intended attributes.

Each source group contains a matching REAL pair and one FAKE pair. The FAKE record changes exactly one attribute in either the caption or the image. The in-distribution splits use color and shape contradictions; only the held-out generalization split uses count contradictions. Thus the label records a controlled intervention, rather than a claim inferred from an annotator or filename. The image model reads image pixels and does not receive manifest labels or scene metadata.

TABLE I Dataset split by source scene

| Split | Groups | Pairs | REAL | FAKE |
| --- | --- | --- | --- | --- |
| Train | 216 | 432 | 216 | 216 |
| Validation | 72 | 144 | 72 | 72 |
| Test | 72 | 144 | 72 | 72 |
| Count OOD | 72 | 144 | 72 | 72 |
| Total | 432 | 864 | 432 | 432 |

The random seed is 7031. All variants of a source scene remain in one split. This is a scene-group split, not a split of random rows. It prevents an original image and its altered partner from directly leaking across boundaries, although all splits still share the same generator and visual grammar. The corpus is procedural original content and contains no collected personal media. Manifest JSON and CSV files preserve identifiers, group, split, source, image path, caption, label, and manipulation metadata.

Text preprocessing lowercases and tokenizes words and numbers. A train-only vocabulary includes unigrams and adjacent bigrams appearing in at least two training documents. Sublinear term frequency and smoothed inverse document frequency produce L2-normalized vectors. Caption parsing separately resolves supported color, shape and count words and a small explicit synonym set. Ambiguous or negated captions receive special handling during interactive inference.

Image preprocessing resizes to 128 by 128 pixels. A saturation mask isolates colored foreground components. Four-neighbor connected components above the minimum area estimate object count and bounding regions. Mean foreground RGB chooses the nearest supported color. Component fill ratio separates squares, circles and triangles. The 15 image features contain one-hot attributes plus foreground fraction, mean RGB and component fill. This hand-designed representation assumes isolated, similarly colored shapes on a pale background.

## IV Models and Explainable Verification

Each learned classifier is a logistic model implemented with NumPy. Training standardizes each feature using training means and scales, optimizes binary cross-entropy with L2 regularization 0.03 through 1400 full-batch Adam steps at learning rate 0.07, and leaves the intercept unpenalized. Predictions use p(FAKE) = sigmoid(w^T z + b). The fitted vocabulary has 93 text features; feature fusion has 114 total features. The saved bundle includes weights, normalizers, vocabulary, configuration and the selected fusion setting. It permits local inference without retraining.

The text baseline learns from TF-IDF alone. The image baseline learns from the 15 pixel-derived features. Feature fusion concatenates both representations with six explicit interaction features: color contradiction, shape contradiction, count contradiction, contradiction fraction, all-attributes match, and known-attribute fraction. These interactions encode the central relation the single-modality models cannot directly observe. Because domain knowledge enters through these features, results do not isolate a benefit of representation concatenation alone.

Late fusion combines separately trained probabilities as alpha times the text score plus (1 - alpha) times the image score. Validation log loss selects alpha = 0.0 in this run, so the reported late-fusion result equals the image baseline. This degenerate optimum is retained rather than replaced after observing test performance. A 0.5 threshold defines binary experiment predictions. No test labels enter vocabulary fitting, feature normalization, model fitting or validation selection.

The explanation uses local probability perturbations. Deleting a caption token and rescoring gives its signed effect. Painting each detected object's rectangle with the light background and rescoring gives a region effect. Holding encoders fixed while flipping one attribute comparison measures its counterfactual effect on p(FAKE). The system shows relevant words and image regions and gives an attribute-based evidence statement. This is evidence grounding within the extractor, not a learned saliency map. Unsupported scenes or captions, or confidence below 0.58, yield UNCERTAIN. The score has no demonstrated real-world probability calibration.

## V Experimental Protocol

The six requested experiment groups are: (1) compare both single-modality models with feature fusion; (2) compare feature and late fusion; (3) test generalization to count manipulation withheld from training; (4) remove text or image evidence at inference; (5) inspect at least 15 difficult predictions; and (6) test explanation relevance through perturbation and faithful contribution accounting. The results JSON stores all measurements, confusion matrices, class-wise scores and threshold curves. The numbered requirements map differs from individual model IDs because several required comparisons reuse the same run.

We report accuracy, precision, recall, F1, ROC-AUC and PR-AUC with FAKE as the positive class. PR-AUC is average precision, calculated by the step integral of precision against recall. Tied scores share a threshold. Confusion matrices use rows [REAL, FAKE] and columns [REAL, FAKE]. Class-wise scores accompany the aggregate comparison. Results come from one deterministic seed and one split, so no statistical significance or sampling confidence interval is claimed.

## VI Results

TABLE II In distribution test metrics

| Model | Acc | Prec | Rec | F1 | ROC | AP |
| --- | --- | --- | --- | --- | --- | --- |
| Text | 0.479 | 0.479 | 0.472 | 0.476 | 0.481 | 0.475 |
| Image | 0.507 | 0.506 | 0.597 | 0.548 | 0.502 | 0.508 |
| Feature | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Late | 0.507 | 0.506 | 0.597 | 0.548 | 0.502 | 0.508 |

Feature fusion achieves test F1 1.000; text-only F1 is 0.476, image-only F1 is 0.548, and late-fusion F1 is 0.548. These measured differences support using explicit agreement information on this task. Weak unimodal performance is expected because the generator keeps individual scenes and captions plausible. It should not be described as evidence that unimodal classifiers are generally ineffective for deepfakes.

TABLE III Feature fusion confusion matrix

| Actual | Pred REAL | Pred FAKE |
| --- | --- | --- |
| REAL | 72 | 0 |
| FAKE | 0 | 72 |

TABLE IV Feature fusion class wise metrics

| Class | Precision | Recall | F1 | n |
| --- | --- | --- | --- | --- |
| REAL | 1.000 | 1.000 | 1.000 | 72 |
| FAKE | 1.000 | 1.000 | 1.000 | 72 |

## A Generalization and ablation

The withheld count-manipulation split gives feature-fusion accuracy 1.000 and F1 1.000. Count contradiction is unseen during fitting, but the architecture contains a general contradiction fraction and all-attributes-match feature. Therefore success here indicates transfer through a designed consistency rule within the same scene domain; it does not establish cross-dataset or natural-image generalization.

TABLE V Withheld count manipulation

| Model | Acc | F1 | ROC |
| --- | --- | --- | --- |
| text only | 0.486 | 0.486 | 0.501 |
| image only | 0.500 | 0.538 | 0.499 |
| feature fusion | 1.000 | 1.000 | 1.000 |
| late fusion | 0.500 | 0.538 | 0.499 |

TABLE VI Modality ablation

| Condition | Acc | F1 | ROC |
| --- | --- | --- | --- |
| Without Text | 0.486 | 0.479 | 0.478 |
| Without Image | 0.465 | 0.476 | 0.471 |
| Without Cross Modal | 0.500 | 0.500 | 0.473 |

The ablation measures a fixed trained model with evidence removed, so it tests dependence at inference rather than the best model retrained for missing data. Missing inputs change the feature distribution. The interactive application consequently treats unavailable or unsupported evidence conservatively. Binary ablation metrics and selective UNCERTAIN decisions answer different questions and should not be combined into one accuracy score.

## B Explanation quality and sanity checks

The explanation evaluation uses the first 48 fixed test records. Removing attribute tokens and filler tokens measures local sensitivity. Repairing a false caption and introducing a color conflict into a matching caption tests whether the score changes in the expected direction. This supports model relevance within the synthetic domain, but does not prove human explanation usefulness or forged-region localization.

TABLE VII Explanation perturbation results

| Measurement | Value |
| --- | --- |
| Mean absolute attribute token effect | 0.286 |
| Mean absolute filler token effect | 0.001 |
| Mean FAKE score drop after correction | 0.924 |
| Correction direction pass rate | 1.000 |
| Mean FAKE score rise after conflict | 0.927 |
| Conflict direction pass rate | 1.000 |

The accompanying machine-readable report records sanity checks for genuine pairs, deliberately conflicting pairs, and unsupported or incomplete inputs. Each check retains its expected behavior and observed outcome, so unsupported inputs cannot silently serve as successful REAL predictions.

## VII Difficult Case Analysis

The saved analysis selects the eight smallest-margin test pairs from each class after fitting, for 16 examples. All are classified correctly and remain confident (the weakest FAKE score is about 0.943), so these are relatively difficult examples, not observed errors. They repeatedly expose color or shape disagreement and matching controls, with several captions describing a single object. Their agreement with the extractor shows the simplicity of the domain rather than robustness to natural language. The report preserves each caption, image path, probability, decision and explanation.

Case 1 (scene-0298-fake). Caption: A picture of one yellow square. Ground truth FAKE; prediction FAKE; p(FAKE) 0.943; correct. Manipulation: color. The feature-fusion model finds a caption/scene contradiction: color: caption says yellow, image shows green. This supports FAKE only under the controlled dataset's consistency definition.

Case 2 (scene-0328-fake). Caption: This image shows one blue triangle. Ground truth FAKE; prediction FAKE; p(FAKE) 0.950; correct. Manipulation: color. The feature-fusion model finds a caption/scene contradiction: color: caption says blue, image shows green. This supports FAKE only under the controlled dataset's consistency definition.

Case 3 (scene-0358-fake). Caption: A picture of one blue square. Ground truth FAKE; prediction FAKE; p(FAKE) 0.952; correct. Manipulation: color. The feature-fusion model finds a caption/scene contradiction: color: caption says blue, image shows green. This supports FAKE only under the controlled dataset's consistency definition.

Case 4 (scene-0334-fake). Caption: Visible in the image: one blue square. Ground truth FAKE; prediction FAKE; p(FAKE) 0.953; correct. Manipulation: color. The feature-fusion model finds a caption/scene contradiction: color: caption says blue, image shows green. This supports FAKE only under the controlled dataset's consistency definition.

Case 5 (scene-0330-fake). Caption: The scene contains one yellow square on a light background. Ground truth FAKE; prediction FAKE; p(FAKE) 0.953; correct. Manipulation: color. The feature-fusion model finds a caption/scene contradiction: color: caption says yellow, image shows green. This supports FAKE only under the controlled dataset's consistency definition.

Case 6 (scene-0297-fake). Caption: A picture of one blue square. Ground truth FAKE; prediction FAKE; p(FAKE) 0.954; correct. Manipulation: shape. The feature-fusion model finds a caption/scene contradiction: shape: caption says square, image shows triangle. This supports FAKE only under the controlled dataset's consistency definition.

Case 7 (scene-0288-fake). Caption: The scene contains one blue circle on a light background. Ground truth FAKE; prediction FAKE; p(FAKE) 0.954; correct. Manipulation: color. The feature-fusion model finds a caption/scene contradiction: color: caption says blue, image shows yellow. This supports FAKE only under the controlled dataset's consistency definition.

Case 8 (scene-0349-real). Caption: The scene contains one green triangle on a light background. Ground truth REAL; prediction REAL; p(FAKE) 0.046; correct. Manipulation: none. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

Case 9 (scene-0352-real). Caption: A picture of one red circle. Ground truth REAL; prediction REAL; p(FAKE) 0.045; correct. Manipulation: none. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

Case 10 (scene-0313-real). Caption: The scene contains two blue circles on a light background. Ground truth REAL; prediction REAL; p(FAKE) 0.044; correct. Manipulation: none. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

Case 11 (scene-0344-fake). Caption: Visible in the image: three yellow triangles. Ground truth FAKE; prediction FAKE; p(FAKE) 0.956; correct. Manipulation: color. The feature-fusion model finds a caption/scene contradiction: color: caption says yellow, image shows red. This supports FAKE only under the controlled dataset's consistency definition.

Case 12 (scene-0359-real). Caption: The scene contains two blue circles on a light background. Ground truth REAL; prediction REAL; p(FAKE) 0.044; correct. Manipulation: none. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

Case 13 (scene-0347-real). Caption: Visible in the image: one red circle. Ground truth REAL; prediction REAL; p(FAKE) 0.044; correct. Manipulation: none. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

Case 14 (scene-0355-real). Caption: This image shows one green square. Ground truth REAL; prediction REAL; p(FAKE) 0.044; correct. Manipulation: none. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

Case 15 (scene-0315-real). Caption: The scene contains one yellow triangle on a light background. Ground truth REAL; prediction REAL; p(FAKE) 0.043; correct. Manipulation: none. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

Case 16 (scene-0328-real). Caption: This image shows one green triangle. Ground truth REAL; prediction REAL; p(FAKE) 0.042; correct. Manipulation: none. The caption and rendered scene agree on color, shape, and count. The feature-fusion model supports REAL under the controlled dataset's consistency definition.

## VIII Limitations and Responsible Use

The narrow generator favors interpretable agreement rules and cannot represent natural-photo complexity. The hand-designed parser and color/shape extractor already contain much of the task structure. The same rendering process appears in every split, the lexical variation is limited, and the single seed gives no measure of variability across different sampled corpora. The reference models are lightweight educational baselines rather than strong pretrained architectures. High synthetic performance cannot justify state-of-the-art or deployment claims.

The uncertainty gate is a conservative heuristic, not a calibrated selective-classification study. Domain heuristics can accept an unusual image or reject a valid stylized one. Negation, multiple object categories, unfamiliar words and natural backgrounds need more capable models. Perturbation effects explain local score changes, yet do not establish causal truth. Token deletion and region masking can also move an input outside the training distribution.

The generated dataset avoids collecting private individuals' media and does not depict real people. An extension to public data must document licenses, consent requirements, provenance, annotator disagreement and demographic or source bias. A deployment must separately assess false accusations and false reassurance. VeriFuse should support coursework and experimentation; it should not make moderation decisions or accuse a person of deception.

## IX Conclusion and Reproducibility

VeriFuse provides a local multimodal verification experiment with measured baselines, two fusion approaches, held-out manipulation testing, ablations and grounded explanations. Feature fusion reaches F1 1.000 on the controlled test and 1.000 on withheld count manipulations. The result is evidence for explicit cross-modal interactions within this generator. Broader claims require licensed natural data, stronger encoders, source-separated evaluation, repeated seeds, calibration and human review of explanations.

To reproduce the delivered run, install the listed Python dependencies, generate the dataset, and run the experiment script from the repository root. The project README gives exact commands. Source data live in data/manifest.json, model parameters in models/bundle.json, and measured evidence in results/experiments.json. Paper and slide builders consume that results file. The author names, student identifiers, affiliation details and repository URL should be added by the submitting team before submission; none are fabricated here.

## References

[1] K. Nakamura, S. Levy, and W. Y. Wang, "Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection," in Proc. LREC, 2020, pp. 6149-6157. https://aclanthology.org/2020.lrec-1.755/

[2] A. Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," in Proc. IEEE/CVF ICCV, 2019. https://arxiv.org/abs/1901.08971

[3] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in Proc. ICML, 2021. https://arxiv.org/abs/2103.00020
