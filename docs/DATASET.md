# Dataset and external-corpus guide

The included corpus is **VeriFuse Controlled Scene Consistency**. It is generated locally from colored shapes. It is not a downloaded news dataset, does not contain journalistic claims, and does not establish real-world misinformation detection performance.

## Labels and provenance

An original scene is a procedural drawing containing one, two or three objects of a single shape (circle, square or triangle) and a single color (red, blue, green or yellow). A **REAL** pair is that original image with its matching literal caption. A **FAKE** pair changes exactly one attribute in either the caption or the rendered image while keeping the other modality unchanged. These are consistency labels. A fabricated real-world event could still have a consistent caption and image.

Seed `7031` controls positions, object sizes, color variation, caption templates, settings order and alteration choices. Images are rendered at 3x resolution with Pillow and downsampled to 256 x 256 RGB. No external image service, model download or third-party media is used.

## Counts and splits

| Split | Original groups | Pairs | REAL | FAKE | Negative alterations |
|---|---:|---:|---:|---:|---|
| Train | 216 | 432 | 216 | 216 | Color or shape |
| Validation | 72 | 144 | 72 | 72 | Color or shape |
| Test | 72 | 144 | 72 | 72 | Color or shape |
| OOD test | 72 | 144 | 72 | 72 | Count only |
| Total | 432 | 864 | 432 | 432 | |

All examples derived from one original are assigned together before encoding. No group crosses splits. The generated run also verifies that no decoded image-pixel hash crosses splits. Sharing the original image between its REAL and text-altered FAKE pair is intentional and remains inside one split. There are 648 distinct generated PNG files.

Each split covers all 4 x 3 x 3 base attribute combinations. Half the negative alterations affect text and half affect images. Color and shape negative alterations occur in training; count alterations are held out completely. OOD here means an unseen **manipulation type within the same drawing domain**, not an unseen photo/news domain. Repeated semantic scene types are intentional; pixels, procedural draws and source groups are separate.

## Files and schema

- `data/manifest.json`: corpus description plus a `records` array.
- `data/manifest.csv`: flat reusable metadata.
- `data/images/*.png`: rendered image assets.
- `models/bundle.json`: fitted TF-IDF vocabulary/IDF, scaling values and logistic weights.
- `results/experiments.json`: actual evaluation results and model settings.

Each record contains `id`, `group_id`, `split`, `label`, `text`, `image_path`, `manipulation`, `altered_modality`, `source`, `scene_seed`, `scene` and `caption`. Image paths are relative to the repository. `scene` and `caption` describe generator ground truth for auditing; **they are never passed into a model**. Models read only the caption text and image pixels. Neither labels, filenames, IDs, source, split, manipulation type nor generator metadata are features.

## Reproduce the included corpus

```powershell
python scripts/generate_dataset.py
python scripts/run_experiments.py
python -m unittest discover -s tests -v
```

Generation overwrites the known generated image paths and manifest; it does not delete unrelated files. Training uses deterministic full-batch Adam and fixed hyperparameters, with validation used only to select the late-fusion weight. Floating point libraries may cause very small numeric differences across machines. The report records the metadata SHA-256 for the evaluated manifest.

## Import an external CSV

Required columns are `image_path,text,label,group_id`. Optional columns are `id,split,manipulation,altered_modality`. Labels must be `REAL` or `FAKE`; their meaning and provenance are the responsibility of the corpus author. Image paths are interpreted relative to the CSV file, and every image is verified before fitting.

```csv
id,group_id,split,label,text,image_path
article-001,source-event-001,train,REAL,"A caption for the first image.",images/first.png
article-002,source-event-001,train,FAKE,"An altered caption for that image.",images/first.png
```

The illustrative two-row snippet is not sufficient for training. Supply both classes in train, validation and test. The optional `data/external_template.csv` is a working small schema example referencing six included pairs; it is a plumbing example, not a meaningful benchmark. If `split` is absent, a stable SHA-256 hash of `group_id` assigns approximately 70%/10%/20% train/validation/test. Group leakage and duplicate record IDs are rejected. Corpus authors must additionally group related events/sources and deduplicate images; row IDs are not a substitute for provenance groups.

```powershell
python scripts/run_experiments.py --csv path/to/corpus.csv --output results/imported
```

An external run writes its separate `bundle.json`, metrics, report and charts under the requested output directory, preserving the demo model. Training also rejects identical decoded image pixels crossing splits. OOD is explicitly `not_available` if no `ood_test` split is provided. The demo engine intentionally retains its synthetic support checks; importing photos does not make it a validated real-news detector. The reusable text model can encode arbitrary text, while the included image encoder and joint comparisons remain designed for colored shapes. Replace the image/text encoders, establish license/provenance, create event-disjoint splits and validate robustness before making a real-world claim. No CLIP or other pretrained encoder is bundled or silently downloaded.

For imported data, `ood_test` means only an author-designated evaluation split. Group-ID and exact-pixel separation are checked, but the software does not infer a held-out manipulation, unseen event/source, balanced class distribution, or distribution shift. External reports retain the supplied label definition and do not inherit the procedural corpus's REAL semantics or count-alteration protocol.

## Data limitations

The controlled vocabulary, plain background and explicit image-caption comparison make the task easy for the fusion model. Perfect synthetic accuracy is therefore expected to overstate generalization to natural media. The parser does not model negation, multiple object categories or arbitrary language; the demo abstains on these inputs. Image support is a heuristic, not a proven out-of-distribution detector. Explanations measure local score changes and visible attribute agreement, not whether an event actually occurred.
