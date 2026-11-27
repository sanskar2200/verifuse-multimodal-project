# PPT alignment with the original PJ_07 brief

The original temporary PDF is no longer present. Its complete four-page text was recovered from the first project turn, and the original rendered pages were reviewed. The recovered text is saved in docs/PJ_07_Recovered_Brief.txt.

The brief requires presentation slides as a deliverable. It does not specify slide count, presentation duration, font, or color scheme. The revised 16-slide structure is an editorial recommendation. It follows the supplied reference's visual design.

## Required experiments

| PJ_07 requirement | Evidence in the project | Revised slide | Status and limit |
|---|---|---|---|
| Experiment 1: single-modality baselines against multimodal | Text-only and image-only logistic heads, compared with feature and decision fusion | 7–9 | Completed on the real-media subset |
| Experiment 2: at least two fusion methods or architectures | Feature concatenation with interactions and weighted decision fusion | 6–8 | Completed |
| Experiment 3: generalization to another manipulation, source, language, or modality combination | Held-out count manipulations in the earlier synthetic study | 12 | Completed only within the controlled synthetic environment; no independent real-data domain evaluation |
| Experiment 4: modality ablation | Fixed real-data fusion model with removed feature families replaced by training means | 10 | Completed as inference ablation, without retraining |
| Experiment 5: qualitative analysis of at least 15 difficult cases | 16 real-data cases logged, eight per source class; earlier synthetic cases also exist | 11 | Real-data human image-caption review and error categorization remain incomplete; a case list alone does not satisfy qualitative analysis |
| Experiment 6: explanation relevance or quality | Token perturbation and caption interventions on 48 synthetic test pairs | 12 | Controlled sensitivity evidence only; real-data token/region attribution and human explanation evaluation remain incomplete |

## Additional requirements

| Requirement | Presentation coverage | Important qualification |
|---|---|---|
| At least two modalities | Slides 2, 3, 6 | Image and text |
| Minimum 300 labeled samples recommended | Slide 4 | 8,585 retained pairs; labels are inherited distant supervision |
| Dataset source, class balance, preprocessing, and splits | Slides 4–5 | Community mirror, not established official benchmark splits |
| Accuracy, precision, recall, F1, ROC-AUC/PR-AUC, and confusion matrix | Slides 8–9 and notes | FAKE F1, macro F1, ROC-AUC, class precision/recall, and balanced accuracy are distinguished |
| Genuine/manipulated/mismatched sanity checks | Slides 11–12 and limitations | Synthetic controlled checks exist; real-data hash audits are not semantic sanity checks |
| REAL / FAKE / UNCERTAIN and explanation | Slides 3, 12–13 | In the real profile, UNCERTAIN handles missing modalities; explanation consists of model diagnostics |
| Conflicting or incomplete inputs | Slides 10–13 | No validated real-data conflict-rejection or out-of-domain mechanism |
| Ethics, privacy, consent, bias, responsible use | Slide 14 | Weak labels, class imbalance, media rights, and human review |
| Demonstration | Slide 13 | Existing local application and actual screenshot |

## Submission deliverables

| Brief deliverable | Current evidence |
|---|---|
| Source code on GitHub | Local source code and project archive exist; this review does not establish a published GitHub repository |
| IEEE-style Word and PDF paper | Existing paper describes the earlier synthetic prototype; revise it before claiming the real-data results |
| Presentation slides | This package supplies revised Canva instructions and content; the older PPTX still describes the synthetic prototype |
| Dataset and preprocessing documentation | docs/REAL_DATASET.md and local audit artifacts |
| Results, confusion matrix, ablation | results/real/report.md and results/real/experiments.json |
| Demonstration | Existing image-text application and saved real-data UI screenshots |

## What changes in the proposed presentation

- Replace the reference's planned ResNet50 and video pipeline with the implemented frozen CLIP image-text system.
- Replace expected outcomes with measured held-out results.
- Replace the unsupported time-series graphic with measured model comparisons.
- Give the confusion matrix and ablation separate, readable slides.
- Keep the synthetic study's results separate from the real-data study.
- Preserve Bennett University and Dr. Sakshi, while leaving student names and IDs blank.
- Describe incomplete experiments honestly. Presentation wording cannot substitute for the missing evaluation work.

## Sources

- docs/PJ_07_Recovered_Brief.txt and tmp/pdfs/brief-1.png through brief-4.png
- C:/Users/Admin/Downloads/nlp demo ppt.pdf
- results/real/report.md
- results/real/experiments.json
- results/experiments.json
- docs/REAL_DATASET.md
- verifuse/real_model.py
