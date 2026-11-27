# VeriFuse

**Multimodal NLP for Deepfake Detection and Explainable Content Verification**  
Bennett University / CSET 346 / PJ 07

VeriFuse is a text-and-image research project with a local Brave demo, trained baselines, two fusion strategies, and reproducible evaluations. It now has a separate **real-media profile** using image-and-text posts from the community `ams-99/fakeddit_9k` mirror. The original synthetic shape benchmark remains available as a separate profile.

**Real-data scope:** separate models handle text and images. Caption text uses train-fitted TF-IDF word and bigram features with an L2 logistic classifier; image pixels use frozen CLIP vision features and a separate L2 logistic classifier. The synthetic profile retains its own TF-IDF text classifier trained on synthetic captions. The models are fitted separately because the two datasets define labels differently. Fakeddit uses distant supervision: its six categories are collapsed to `REAL` (true) and `FAKE` (satire, false connection, imposter content, manipulated content, misleading content). These are inherited dataset labels, not independently verified factual verdicts or a dedicated face-swap detector. Missing evidence returns `UNCERTAIN`; arbitrary uploads have no validated domain guarantee. Read the actual run in [results/real/report.md](results/real/report.md).

## Start the demo

On this computer, double-click **start.bat** to start the server and open Brave, or run:

```powershell
.\start.ps1 --browser brave
```

The demo is at **http://127.0.0.1:8765**. If Brave does not open automatically, paste that address into Brave. Keep the terminal running. Press Ctrl+C to stop.

The default `auto` profile selects the real model when `models/real/bundle.json` exists. Use `--profile real` to require it, or `--profile synthetic` to reproduce the original demo. First real prediction loads CLIP and can take several seconds.

## Reproduce real-data training

On another computer with Python 3.11, install into a project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-real.txt
.\.venv\Scripts\python.exe scripts/download_clip.py
.\.venv\Scripts\python.exe scripts/download_real_data.py
.\.venv\Scripts\python.exe scripts/train_real.py --batch-size 16 --threads 4
.\.venv\Scripts\python.exe app.py --profile real --browser brave
```

The downloads are approximately 605 MB for CLIP and 500 MB for data, plus Python dependencies. Training runs on CPU; cached CLIP image features let interrupted extraction resume. The text model learns a maximum of 2,048 TF-IDF unigram and bigram features from the training captions. The image model uses 512-dimensional frozen CLIP vision features. The pipeline evaluates text-only, image-only, feature-fusion and validation-weighted late-fusion models. Standardization and coefficients fit on training data, validation log loss selects settings, and the supplied test partition is held out from model selection.

The mirror's split provenance is not established as the official Fakeddit benchmark split. The importer audits exact decoded-image, text and source duplicates and removes cross-partition overlap. Related events and near-duplicate pictures can remain. The mirror does not publish a clear media redistribution license, so the ZIP excludes downloaded media, raw metadata, feature caches and large CLIP weights. Reproduce downloads locally from the documented sources. No full-benchmark or independent-domain performance is claimed.

Real artifacts: `models/real/bundle.json`, `data/real/manifest.json`, `results/real/experiments.json`, `results/real/predictions.json`, and [training report](results/real/report.md). The [real dataset note](docs/REAL_DATASET.md) gives exact retained counts and provenance. Download receipts retain pinned source revisions and hashes. Scores are uncalibrated; the real profile reports separate modality scores, with no invented token or region attribution.

## Original synthetic demo

For the lightweight shape benchmark with Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/run_experiments.py
.\.venv\Scripts\python.exe app.py --profile synthetic
```

On macOS or Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. The synthetic profile uses only NumPy and Pillow; the real profile also needs PyTorch, Transformers and local CLIP files. No API key or GPU is needed.

The original synthetic run used Python 3.12.14, NumPy 2.3.5 and Pillow 12.3.0. The real run uses Python 3.11 and records its exact runtime versions in `results/real/experiments.json`. Use `python app.py --port 8766` if port 8765 is already occupied.

## Synthetic demonstration

1. Choose a sample from the gallery and inspect its image and caption.
2. Select feature fusion, then analyze the pair. Read the prediction and highlighted evidence.
3. Change a color, shape, or count in the caption and analyze again.
4. Compare the same input with late fusion.
5. Remove a modality or try an unrelated image and observe the uncertainty behavior.
6. Open Experiments to compare actual held-out results, ablations, generalization, and difficult cases.
7. Download a prediction as JSON for the demonstration record.

## Reproduce the synthetic experiments

```powershell
python scripts/run_experiments.py
python -m unittest discover -s tests -v
```

The pipeline creates the dataset when needed, trains on the training split, selects settings using validation data, and evaluates held-out scenes. Source groups remain separate across splits. Model inputs come from image pixels and caption text, never filenames, labels, or manipulation metadata. All reported numbers come from a real local run.

| Requirement | Implementation and evidence |
| --- | --- |
| At least two modalities | Caption text and image pixels |
| Two unimodal baselines | TF-IDF text logistic classifier and image-feature logistic classifier |
| Two fusion strategies | Feature fusion with cross-modal consistency features, and decision-level late fusion |
| Generalization | Held-out manipulation experiment in `results/experiments.json` |
| Modality ablation | Remove caption or image and re-evaluate |
| At least 15 difficult cases | Selected examples with evidence in results |
| Explanation evaluation | Counterfactual token deletion and image-region interventions |
| Metrics | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC and confusion matrices |
| Sanity checks | Consistent original, manipulated pair, and unsupported or missing evidence |
| Dataset and ethics | `docs/DATASET.md`, `docs/MODEL_CARD.md` |
| Final research paper | Six-page IEEE PDF and editable LaTeX source in `output/IEEE_Verifuse_Paper/` |

Read `results/REPORT.md` for measured findings and limitations. Binary benchmark scores use a 0.5 threshold; the interactive tool can abstain, so binary benchmark accuracy and three-way demo behavior should not be confused. Reported probabilities are experimental model estimates, not calibrated probabilities of real-world truth.

## Final IEEE paper

- [Six-page report PDF](output/IEEE_Verifuse_Paper/VeriFuse_IEEE_Final.pdf)
- [Editable LaTeX source](output/IEEE_Verifuse_Paper/main.tex)
- [Included IEEE conference class](output/IEEE_Verifuse_Paper/IEEEtran.cls)
- [Experiment report](results/real/report.md)
- [Demonstration and viva guide](docs/DEMO_GUIDE.md)

The final paper includes the synthetic and real-media experiments as separate evaluations. Its abstract is single-column, and the remaining paper uses two columns. The package does not include the older synthetic-only paper or presentation exports.

## Project files

```text
app.py                  Local application entry point
verifuse/               Features, models, evaluation, inference and HTTP API
static/                 Responsive browser interface
scripts/                Dataset, experiment and deliverable builders
data/manifest.json      Labels, provenance and group-aware splits
data/images/            Original and manipulated synthetic scene images
models/                 Saved trained model parameters
results/                Measured metrics, difficult cases and report
docs/                   Dataset description, model card and paper sources
output/                 Final IEEE report and project archive
tests/                  Model and API checks
```

## HTTP API

- `GET /api/health` reports application and model readiness.
- `GET /api/samples` returns curated demonstration records and image URLs.
- `GET /api/experiments` returns measured experimental results.
- `POST /api/predict` accepts JSON containing `text`, `model`, and either `sample_id` or `image_base64`. Images may be a data URL or raw base64. Supported formats are PNG, JPEG and WebP, at most 8 MB and 16 million pixels. Caption length is at most 3,000 characters.

Example request body:

```json
{
  "text": "Two blue circles on a light background.",
  "image_base64": "<base64 PNG, JPEG or WebP bytes>",
  "model": "feature_fusion"
}
```

The response includes `prediction`, `fake_probability`, `confidence`, `explanation`, `evidence`, `token_attributions`, `image_regions`, `modality_scores`, `warnings`, and `in_domain`. Missing inputs can trigger abstention. Invalid requests return an `error` message with an HTTP error status.

## Responsible interpretation

The real corpus contains publicly shared social-media images and text, including people. Its inherited labels can be noisy. The local server processes uploaded images in memory and does not persist them. The program binds to localhost and is intended for an individual research demonstration. Avoid treating a prediction as an accusation or proof. Dataset-specific shortcuts restrict generalization; the synthetic profile additionally uses a limited caption grammar.

## Final IEEE report

The six-page IEEE conference report is available at `output/IEEE_Verifuse_Paper/VeriFuse_IEEE_Final.pdf`. Its abstract is single-column, and the remaining paper uses two columns. The editable LaTeX source and the supplied `IEEEtran.cls` are in the same folder.

Create a fresh project archive with `python scripts/package_project.py`. The archive includes the application, source code, documentation, measured results, and final report files. It excludes downloaded third-party media, raw dataset files, feature caches, large pretrained weights, and older report or presentation exports.

## Background

- Nakamura, Levy and Wang, [Fakeddit](https://aclanthology.org/2020.lrec-1.755/), LREC 2020.
- Rössler et al., [FaceForensics++](https://arxiv.org/abs/1901.08971), 2019.
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), 2021.

Fakeddit and CLIP are used by the real-data profile. FaceForensics++ is a background reference only. The distributable ZIP excludes downloaded third-party datasets and large pretrained weights; local downloads remain available on this computer.
