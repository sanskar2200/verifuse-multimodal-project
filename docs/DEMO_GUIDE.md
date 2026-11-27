# Demonstration and viva guide

## Five minute demonstration

Start the app with `start.bat` and open `http://127.0.0.1:8765`. Choose an original sample from the gallery. Read its literal caption aloud, run feature fusion, and explain that REAL means consistency under the benchmark definition.

Change exactly one caption attribute while keeping the image fixed. For example, change blue to red or circles to squares. Run again and show the cross-modal comparison and token deletion effects. A change in the model's output after the edit illustrates the interaction between modalities.

Select late fusion on the same pair. Explain why two independently computed class scores can miss a contradiction that neither modality reveals by itself. Show the actual held-out comparison in Experiments rather than claiming one approach always wins.

Remove one modality. Show the UNCERTAIN response and the ablation results. An image or caption alone cannot establish agreement between the two. Finally, open the selected difficult cases and describe a limitation rather than presenting a single successful example as a complete evaluation.

End by opening the paper's results table and model card. State that real-world deepfake validation remains future work.

## Questions to prepare

**What is the NLP contribution?**  
The text baseline learns from normalized word and bigram TF-IDF features. A restricted caption parser extracts explicit color, shape and count claims. Feature fusion combines those claims with attributes estimated from actual pixels to represent semantic agreement. The parser is a handcrafted component, not a pretrained language model.

**Why can the single-modality baselines perform poorly?**  
Both classes contain plausible captions and plausible images. The label can depend on their relationship. A caption alone does not identify which picture accompanies it, and a picture alone does not reveal its caption. The measured baseline results still need to be reported, including shortcuts and variation.

**What is the difference between feature and late fusion?**  
Feature fusion classifies the combined representations and cross-modal comparisons. Late fusion combines independently estimated probabilities. It does not gain a direct representation of which caption attribute contradicts an image attribute.

**How are labels obtained?**  
The generator starts from a known scene specification and a matching caption. It then deliberately changes one attribute in the image or caption to create a contradictory pair. This gives reproducible labels for the controlled task. It does not label natural events as true or false.

**How do you prevent leakage?**  
All variants of an original scene share a group and split. The pipeline fits text vocabulary, scaling, and classifier parameters using training data. It uses validation data for model choices. Test labels and manipulation metadata are excluded from input features. Repeated-image leakage must also be checked.

**What is the generalization experiment?**  
The run reserves a manipulation attribute for the out-of-distribution split. The result tests transfer within the scene generator. Shared agreement features already encode some knowledge about the held-out attribute, so this is a limited generalization test, not proof of cross-dataset transfer.

**What do highlighted words and regions mean?**  
Their scores quantify changes in the model's inconsistency probability after a word deletion or image-region masking intervention. They measure model sensitivity. A highlight does not establish that a real image was manipulated. Perturbations can themselves fall outside the training distribution.

**What does UNCERTAIN mean?**  
The program abstains if an input lacks required evidence, falls outside its supported scene and caption domain, or produces a score close to the decision boundary. The support check is heuristic, and uncertainty does not guarantee that every novel input will be identified correctly.

**Are the reported probabilities calibrated?**  
No real-world calibration has been established. They are the fitted classifier's scores. Report binary metrics separately from abstaining demo behavior and do not interpret confidence as the probability that a news event is authentic.

**What would you add before testing natural media?**  
Use a documented, legally reusable image-text corpus with group-aware splits, replace geometric image extraction with suitable visual or vision-language representations, and train all baselines again. Add external datasets, confidence calibration, error analysis, and human relevance assessment. A face-manipulation detector is a distinct task requiring appropriate labels and evaluation.

## Submission checklist

- Fill in student names, IDs, affiliation and any instructor-specific title fields.
- Read the generated research paper and verify its conclusions against `results/experiments.json`.
- Rehearse both a successful consistency check and an UNCERTAIN example.
- Include the source, dataset documentation, measured results and editable slides.
- Publish to your own GitHub repository if required by the course. No remote repository is created automatically by this project.
- Describe the work as a controlled multimodal manipulation prototype. Do not claim natural-media detection results that have not been measured.
