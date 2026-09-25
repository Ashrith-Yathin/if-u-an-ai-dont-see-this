# Machine Learning Business Entity Resolution Pipeline

High-performance, scalable entity resolution pipeline built for the Amazon ML Challenge 2026. Resolves multi-source business entity fragments across three heterogeneous vendor sources under precision-heavy macro $F_{0.5}$ evaluation.

---

## 1. System Overview

- **Task:** For each Source 1 reference entity, identify all corresponding records from Source 2 and Source 3.
- **Evaluation Metric:** Macro-averaged $F_{0.5}$ over all Source 1 entities (including singletons).
- **Core Technology:**
  - Dynamic Country-Partitioned Inverted Index Blocking (>96.5% recall ceiling)
  - Phonetic Unidecoding (supporting regional Indian scripts & European accents)
  - 32-dimensional Pairwise Feature Extraction with RapidFuzz C++ kernels
  - LightGBM Gradient Boosted Decision Tree Matcher
  - Macro $F_{0.5}$ Threshold Calibration ($\tau_{S2}=0.75, \tau_{S3}=0.85$)
  - 1-to-1 Target Exclusivity Bipartite Consistency

---

## 2. Directory Structure

```text
code/business_entity_resolution/
├── src/
│   ├── normalization.py      # Unicode cleaning, legal suffix removal, address expansion
│   ├── blocking.py           # Multi-attribute inverted index candidate generator
│   ├── features.py           # 32 lexical, phonological, structural, numerical features
│   ├── model.py              # LightGBM and HistGradientBoosting model wrappers
│   ├── thresholding.py       # Macro F0.5 threshold optimization and deduplication
│   ├── evaluation.py         # Official competition macro F0.5 scoring metric
│   ├── data_loader.py        # Streaming TSV reader and country partitioner
│   ├── output.py             # Submission TSV generation and report writing
│   ├── inference.py          # End-to-end country-streaming test inference engine
│   └── main.py               # CLI entry point with automated submission validation
├── requirements.txt          # Pinned Python dependencies
└── README.md                 # Complete documentation and reproduction guide
```

---

## 3. Installation & Requirements

Ensure Python 3.8+ (tested on Python 3.14) is installed. Install all pinned requirements:

```bash
pip install -r requirements.txt
```

### Dependencies:
- `numpy==2.4.2`
- `scipy==1.17.1`
- `pandas==3.0.6`
- `scikit-learn==1.8.0`
- `lightgbm==4.7.0`
- `rapidfuzz==3.14.6`
- `text-unidecode==1.3`
- `joblib==1.5.3`

---

## 4. End-to-End Reproduction Guide

### Step 1: Model Training & Validation (Optional to retrain from scratch)
To run the full training pipeline on training data and evaluate against the held-out validation set:

```bash
python train.py
```
This fits the model, performs threshold calibration, logs validation metrics, and serializes the model to `models/final_entity_matcher.joblib`.

### Step 2: Test Inference & Submission Generation
To run candidate generation, feature extraction, model scoring, and generate submission files for all test records:

```bash
python code/business_entity_resolution/src/main.py \
    --test-dir dataset/test \
    --output-dir output
```

This generates:
- `output/matching_results.tsv` (Leaderboard submission file)
- `output/candidate_pairs.tsv` (Auditing candidate set file)

### Step 3: Submission Verification
The pipeline automatically runs the competition validator:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Output: **PASS**

---

## 5. Validation Results & Benchmarks

| Metric | Validation Score |
| :--- | :--- |
| **Macro $F_{0.5}$** | **0.976105** |
| **Precision** | **99.24%** |
| **Recall** | **94.54%** |
| **Macro $F_1$** | **96.15%** |
| **Candidate Recall** | **96.48%** |
| **Singleton Accuracy** | **98.02%** |
| **False Positives** | **126 / 17,420 links** |

---

## 6. Compliance & Fair Play

- **Zero External Lookups:** Absolutely no external APIs, Google geocoding, Wikidata, OpenStreetMap, or external business registries are utilized.
- **Permissive Open-Source License:** Model architecture is built exclusively using MIT/Apache 2.0 licensed libraries (LightGBM / Scikit-Learn) with fewer than 100,000 parameters (well within the 8 Billion parameter ceiling).
- **Universal Generalization:** Dynamic string-based country handling natively supports unseen countries (such as France in the test set) without manual hardcoding.
