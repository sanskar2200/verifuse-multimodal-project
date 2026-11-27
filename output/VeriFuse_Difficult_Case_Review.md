# Real-data difficult-case review ledger

These 16 cases are the eight lowest-margin feature-fusion test examples per source class, selected after fitting. The scores and labels are from the saved run. Qualitative image-caption review has not been completed. This table records candidates for that review, not its conclusions.

| Case ID | Caption excerpt | Source label | Prediction | FAKE score | Correct against source label | Human review |
|---|---|---|---|---:|---|---|
| fakeddit9k_test_00110 | a little spider sat on top of the water | REAL | FAKE | 50.07% | No | Pending |
| fakeddit9k_test_00833 | this cardboard i used for spray painting looks deep in thought | FAKE | FAKE | 50.12% | Yes | Pending |
| fakeddit9k_test_00093 | you can leave fingerprints on dry ice | REAL | REAL | 49.87% | Yes | Pending |
| fakeddit9k_test_00652 | javier bardem on set for no country for old men | FAKE | FAKE | 50.15% | Yes | Pending |
| fakeddit9k_test_00045 | a bottle of smelling salts found in a cupboard at work | REAL | FAKE | 50.27% | No | Pending |
| fakeddit9k_test_00697 | natural disasters impact pets too shiba found guarding its owners shattered body following japane... | FAKE | FAKE | 50.28% | Yes | Pending |
| fakeddit9k_test_01362 | suspect says he was rushing to get around after dwi | FAKE | FAKE | 50.29% | Yes | Pending |
| fakeddit9k_test_00371 | people are so touched by how this dad hilariously handled his yearold daughters accident | FAKE | REAL | 49.71% | No | Pending |
| fakeddit9k_test_00011 | breast cancer drug that can extend lives approved for the nhs | REAL | FAKE | 50.63% | No | Pending |
| fakeddit9k_test_01309 | here is my sisters new hound miles | FAKE | FAKE | 50.85% | Yes | Pending |
| fakeddit9k_test_00682 | theres a gorilla holding the license plate on the bus in front of me | FAKE | REAL | 49.08% | No | Pending |
| fakeddit9k_test_00403 | he rescued this blizzard creature from a sidewalk but he had no idea what it would grow into | FAKE | FAKE | 51.00% | Yes | Pending |
| fakeddit9k_test_00218 | my high school has a glass panel in one of the conference rooms by the office and the school masc... | REAL | FAKE | 51.12% | No | Pending |
| fakeddit9k_test_00199 | monument to apples jobs removed in russia after ceo comes out | REAL | REAL | 48.84% | Yes | Pending |
| fakeddit9k_test_00055 | statue of poseidon | REAL | FAKE | 51.80% | No | Pending |
| fakeddit9k_test_00275 | a panda waving hand | REAL | FAKE | 51.91% | No | Pending |

## Review fields to complete for each case

- Inspect the image and its full caption together.
- Record the visible evidence and any contradiction or ambiguity.
- Distinguish inherited-label uncertainty from an identifiable model error.
- Assign an error category only when the evidence supports it.
- Record whether the displayed modality scores or similarity help explain the decision.
- Record reviewer initials and a short rationale.

The full record IDs, original captions, and local image paths are in results/real/experiments.json.
