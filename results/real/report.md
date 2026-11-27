# VeriFuse: real-data training report

Fakeddit real Reddit image-text subset (community mirror)

Source six-way label 0 (true) maps to REAL; labels 1-5 (satire, false connection, imposter, manipulated, misleading content) map to FAKE. Labels are inherited distant supervision, not new fact-checks or image-authenticity annotations.

The text model is train-fitted TF-IDF with an L2 logistic classifier. The image model uses frozen CLIP vision features with a separately fitted L2 logistic classifier.

## Dataset partitions

| Partition | Pairs | REAL | FAKE |
|---|---:|---:|---:|
| train | 4799 | 941 | 3858 |
| val | 2371 | 470 | 1901 |
| test | 1415 | 278 | 1137 |

Scalers and heads fit on training only. Validation log loss selects regularization, the late-fusion weight, and the recommended model. Test labels are not used for selection.

## Held-out test measurements

| Model | Accuracy | F1 (FAKE) | Macro F1 | ROC-AUC | Average precision | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Training-majority baseline | 0.8035 | 0.8911 | 0.4455 | 0.5000 | 0.8035 | 0.4955 |
| Text-only baseline | 0.8177 | 0.8963 | 0.5710 | 0.7424 | 0.9130 | 0.4318 |
| Image-only baseline | 0.8085 | 0.8914 | 0.5412 | 0.8218 | 0.9519 | 0.3882 |
| Feature-level fusion | 0.8339 | 0.9040 | 0.6452 | 0.8520 | 0.9587 | 0.3609 |
| Decision-level fusion | 0.8071 | 0.8909 | 0.5280 | 0.8272 | 0.9534 | 0.3867 |

Validation-selected model: **feature_fusion**. Fixed decision threshold: 0.5. Scores are uncalibrated class estimates.

The majority baseline always predicts **FAKE**, chosen from the training distribution. Its test accuracy is **80.4%**; inspect macro F1 and each class's recall when labels are imbalanced.

Confusion matrices use true rows/predicted columns, in REAL then FAKE order.

- Text-only baseline: `[[42, 236], [22, 1115]]`
- Image-only baseline: `[[32, 246], [25, 1112]]`
- Feature-level fusion: `[[74, 204], [31, 1106]]`
- Decision-level fusion: `[[27, 251], [22, 1115]]`

## Selected model by class

| Dataset class | Precision | Recall | F1 | Test examples |
|---|---:|---:|---:|---:|
| REAL | 0.7048 | 0.2662 | 0.3864 | 278 |
| FAKE | 0.8443 | 0.9727 | 0.9040 | 1137 |

Balanced accuracy: **0.6195**. The majority class has greater recall; overall accuracy alone hides the lower recall on REAL-labeled posts.

## Audits and limits

Checked source/hash fields: group_id, image_sha256, pixel_sha256, source_image_id, source_text_id.

- Community mirror lacks a dataset card, original Reddit post IDs and documented sampling/split provenance.
- Fakeddit labels are distant supervision from subreddit categories, not independent fact-checks for every post.
- FAKE includes satire, false connections, imposter, manipulated and misleading content; REAL means source label true, not a new truth verification.
- Image/text exact duplicates are audited, but near-duplicate events and pretraining overlap are not ruled out.
- Only supplied source identifiers and hashes are audited; unidentified related events and near-duplicate images can remain.
- Overlap with the frozen encoder's undisclosed pretraining corpus cannot be ruled out.
- This is a subset/mirror evaluation unless the dataset manifest explicitly establishes otherwise; full published benchmark performance is not claimed.
- Dataset labels may reflect weak supervision or source conventions; the classifier does not establish real-world truth.
- The image branch uses frozen CLIP vision features; the text branch is a separate train-fitted TF-IDF plus logistic regression model.
- Scores are uncalibrated dataset class estimates; arbitrary uploads have not been validated as in-domain.
- No real-world face-swap, synthetic speech, video, human explanation, or external-domain evaluation was performed.

## Sources

- [Fakeddit paper](https://aclanthology.org/2020.lrec-1.755/)
- [Community data mirror](https://huggingface.co/datasets/ams-99/fakeddit_9k)
- [Author-provided label mapping](https://github.com/entitize/Fakeddit/issues/14#issuecomment-815355653)
- [Pretrained CLIP image encoder](https://huggingface.co/openai/clip-vit-base-patch32)

Dataset mirror revision: `2ba956a67d2bab34de97b25d71af520ab3fe1e6e`. Download checksums and exact exclusions are retained locally; see `docs/REAL_DATASET.md`.

## Reproduction

Run `python scripts/train_real.py` after preparing `data/real/manifest.json` and the pinned local encoder in `models/real/clip`. Extraction checkpoints resume automatically by image/text content. Use `--embeddings` only for a documented supplied feature file aligned by record ID.

Dataset manifest SHA-256: `e24b3af4b3d72146d778002121fe618be330d3c3889c886045ed3ec532f8a7d8`.

Training started (UTC): 2026-11-25T07:34:18.599633+00:00. Head fitting/evaluation wall time: 8.998 seconds.

CLIP revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Extraction wall time for the recorded invocation: 471.241 seconds. Wall time for this extraction invocation including cache lookup and model loading; earlier interrupted invocations are excluded.

Runtime versions: `{"Pillow": "12.3.0", "huggingface-hub": "0.36.2", "numpy": "2.4.6", "platform": "Windows-10-10.0.26200-SP0", "python": "3.11.9", "scikit-learn": "1.9.1", "torch": "2.14.0+cpu", "transformers": "4.57.6"}`.

Raw class scores for every partition are saved in `predictions.json`; the complete metrics, validation trials, ablations, and low-margin test cases are saved in `experiments.json`. No external-domain or explanation-faithfulness measurement is claimed.
