# VeriFuse model card

## What the models do

VeriFuse is an educational prototype for studying agreement between an image and a short caption. Its runnable experiment uses generated scenes containing colored geometric shapes. A matching image-caption pair is called **consistent**; a deliberately mismatched pair is called **inconsistent**.

The word “fake” in experiment labels refers only to this constructed mismatch. The system does not establish whether a news claim, article, photograph, or event is true. A consistent picture and caption can still describe a fabricated event; an inconsistent pair may be a harmless illustration, a crop, or a paraphrase outside the model's vocabulary.

## Intended use

- Demonstrate text-only, image-only and multimodal classification in a reproducible local project.
- Compare models under a controlled train/test split.
- Explore the limitations of matching images to words before using real-world datasets.

It is not suitable for moderation decisions, accusations of misinformation, fact checking, journalism verification, or any decision about a person or organization.

## Training and evaluation data

The project generates its own synthetic examples with a fixed random seed. They are not collected from social media or news publishers, and they carry no verified real-news annotations. Results on this dataset describe performance on the generator's narrow task and should not be generalized to natural photographs or unconstrained language.

Evaluate only on examples excluded from training. Keep all variants derived from an original scene together when assigning dataset splits so that the same underlying scene cannot appear on both sides of an evaluation.

## Model approaches

- **Text baseline:** a NumPy logistic classifier over TF-IDF caption features.
- **Image baseline:** a logistic classifier over engineered scene attributes extracted from the image.
- **Early fusion:** one classifier combining text, image and cross-modal agreement features.
- **Late fusion:** a blend of separately computed model scores.

These lightweight baselines are intended to make the experiment inspectable and runnable locally. They do not use a pretrained vision-language foundation model and should not be described as reproducing a state-of-the-art news verification system.

The included seed-7031 run uses 864 labeled pairs from 432 original scene groups: 432 training, 144 validation, 144 test and 144 held-out-manipulation pairs. Every split is exactly balanced. Models use 1,400 full-batch Adam steps, learning rate 0.07 and L2 regularization 0.03. TF-IDF unigrams/bigrams and feature scaling are fit only on the training split. The image extractor uses connected components, dominant palette color, foreground area and component fill to estimate visible count and shape. Joint features encode color/shape/count contradictions, contradiction fraction, complete agreement and attribute availability. This explicit inductive bias makes the synthetic problem easy.

The late-fusion weight is chosen from 0.00 to 1.00 in increments of 0.05 by validation log loss. In the included run the selected text weight is 0.00, so the blend equals the image baseline; this is reported openly, not treated as an improvement. Binary evaluation uses probability 0.5. The interactive engine additionally abstains if the input is unsupported or the maximum class score is below 0.58. Scores have not been calibrated.

For the generated run's measurements and evaluation details, see [the results report](../results/REPORT.md). Metric values belong to that specific dataset, random seed and split; they are not external benchmark results.

## Interpreting outputs

Any reported false or inconsistency probability is an experimental model score. It is not a probability that an event did not happen, and it is not evidence that a publisher or person is deceptive. Accuracy on generated data does not establish real-world calibration.

Uploaded photographs, unsupported image formats or scene compositions, and captions outside the supported vocabulary should produce **UNCERTAIN** or a clear validation error. Unsupported inputs must not be silently treated as verified real or fake news. Even a confident in-domain prediction is a statement about the controlled caption-consistency task only.

Token attribution is the signed difference between the original FAKE score and the score after removing that token. Image-region attribution paints a detected object rectangle with the scene background and recomputes the score. Cross-modal evidence counterfactually flips an attribute comparison while keeping independent encoders fixed. These are local perturbation explanations, not attention maps, causal proofs or evidence retrieval. Removing content can create out-of-distribution inputs; the report explicitly distinguishes faithfulness checks from factual correctness.

## Known limitations

- Synthetic scenes are much simpler than real photographs.
- The vocabulary and supported scene composition are intentionally small.
- Image and language shortcuts can inflate scores on poorly designed data.
- A model can fail on unfamiliar colors, shapes, backgrounds, paraphrases, or image transformations.
- There is no evidence retrieval, source authentication, reverse image search, provenance analysis, or event verification.
- Real-world robustness, demographic fairness, and probability calibration have not been established.

## Responsible extension

Before a real-world experiment, obtain appropriately licensed data, document annotation procedures and uncertainty, deduplicate images and events across splits, review dataset bias, and report performance with confidence intervals on genuinely held-out sources and events. Separate image-caption agreement from factual verification. Obtain domain review before proposing any consequential use.

The project intentionally provides no pretrained real-news detector or claim of deployment readiness.
