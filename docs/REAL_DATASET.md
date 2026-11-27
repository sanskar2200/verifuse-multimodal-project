# Real-media dataset and training scope

VeriFuse uses downloaded images and post text from the public community mirror [ams-99/fakeddit_9k](https://huggingface.co/datasets/ams-99/fakeddit_9k), pinned to revision `2ba956a67d2bab34de97b25d71af520ab3fe1e6e`. The downloader checks source-file sizes and SHA-256 hashes. This project does not generate the images, captions or target labels in this profile.

The original [Fakeddit paper](https://aclanthology.org/2020.lrec-1.755/) describes distant supervision from source communities. The [authors' label mapping](https://github.com/entitize/Fakeddit/issues/14#issuecomment-815355653) is 0 true, 1 satire, 2 false connection, 3 imposter content, 4 manipulated content and 5 misleading content. We map 0 to REAL and the other five categories to FAKE. These labels are not independent verification of each image or claim.

## Partitions and duplicate audit

| Partition | Supplied rows | Retained rows | REAL | FAKE |
|---|---:|---:|---:|---:|
| Training | 4,800 | 4,799 | 941 | 3,858 |
| Validation | 2,405 | 2,371 | 470 | 1,901 |
| Test | 1,441 | 1,415 | 278 | 1,137 |
| Total | 8,646 | 8,585 | 1,689 | 6,896 |

The mirror-supplied partitions are retained. We connect rows sharing a canonical source image URL, exact image-file hash, decoded RGB pixel hash or normalized caption hash. Components keep their highest-priority partition (training, then validation, then test); lower-priority overlapping rows are removed, never moved into training. Exact repeated pairs within a split are also removed. The audit excludes 61 rows and finds zero cross-split collisions on the checked identifiers. Local `data/real/exclusions.json`, `source_audit.json`, `manifest.json` and `download_receipts.json` retain the evidence.

The mirror lacks an original dataset card, Reddit post identifiers and documented sampling history. Its partitions must not be described as the official Fakeddit benchmark split. Five sampled retained rows were checked against the downloaded source CSV and matched; this checks import fidelity, not independent authenticity. Near-duplicate images, shared events and overlap with CLIP pretraining remain possible.

## Training and limitations

The real-data text model is train-fitted TF-IDF word and bigram features with an L2 logistic classifier. Its vocabulary and IDF values are learned on the training partition only. The image model uses frozen [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) vision embeddings and a separate L2 logistic classifier. Feature fusion concatenates the two modality-specific feature vectors; decision fusion blends their validation-selected scores. The real text model is trained separately from the synthetic text model because their label sources and tasks differ. Validation selects regularization and the late-fusion weight; the test set remains held out. Measurements are in `results/real/report.md` and `results/real/experiments.json`.

About 80% of retained rows carry the FAKE label, so results include a training-majority baseline, macro F1, balanced accuracy and per-class metrics. The explicitly manipulated-content category is rare (13 training, 5 validation and 2 test rows); this is not evidence for reliable pixel-edit or face-swap detection. Separate modality scores describe model behavior. No token/region attribution, explanation-faithfulness result or independent-domain accuracy is claimed.

Neither the mirror nor the original repository establishes a clear dataset redistribution license. Downloaded media and raw text stay in the local experiment and are excluded from the project ZIP. The ZIP also excludes the large pretrained encoder; pinned download scripts reproduce these dependencies. Source content may contain people, news claims, memes or satire. Predictions are uncalibrated corpus-label estimates.
