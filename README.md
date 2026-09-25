# Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Model: LightGBM](https://img.shields.io/badge/model-LightGBM%20GBDT-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Validation F0.5: 0.9761](https://img.shields.io/badge/Validation%20Macro%20F0.5-0.9761-success.svg)](experiments/results.csv)
[![Validation Precision: 99.24%](https://img.shields.io/badge/Validation%20Precision-99.24%25-blueviolet.svg)](output/final_report.txt)
[![Submission Status: PASS](https://img.shields.io/badge/Submission%20Validator-PASS-success.svg)](output/final_report.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **High-precision, ultra-scalable Machine Learning Business Entity Resolution pipeline** engineered for the **Amazon ML Challenge 2026**. Resolves multi-source commercial entity fragments across three heterogeneous vendor sources under the macro-averaged $F_{0.5}$ metric on 12M+ records.

---

## 📌 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Problem Statement & Analysis](#-problem-statement--analysis)
3. [Architecture & Pipeline](#-architecture--pipeline)
4. [Candidate Generation (Blocking)](#-candidate-generation-blocking)
5. [Feature Engineering (32 Dimensions)](#-feature-engineering-32-dimensions)
6. [Modeling & Global Consistency Deduplication](#-modeling--global-consistency-deduplication)
7. [Benchmark Results & Ablation Studies](#-benchmark-results--ablation-studies)
8. [Directory Structure](#-directory-structure)
9. [Installation & Setup](#-installation--setup)
10. [End-to-End Reproduction Guide](#-end-to-end-reproduction-guide)
11. [Submission Compliance](#-submission-compliance)

---

## 🚀 Executive Summary

In commercial data ecosystems, business identity data arrives asynchronously from disparate vendor sources with partial, noisy, and conflicting information (names, abbreviations, transliterations, municipal numbering, landmarks).

Our solution establishes an **end-to-end entity resolution pipeline**:
- **Candidate Recall Ceiling:** **96.48%** candidate recall at **>99.97%** search space reduction using multi-attribute inverted index blocking.
- **Top Metric Performance:** Validation **Macro $F_{0.5} = \mathbf{0.976105}$** (Precision: **99.24%**, Recall: **94.54%**).
- **Singleton Accuracy:** **98.02%** accuracy on singletons (zero-match entities).
- **Extreme Efficiency:** Streams through 1.73M reference entities and 10M+ target vendor records in **~58 minutes** with strict memory containment (**< 2.2 GB RAM**).
- **Strict Compliance:** Zero external API or database lookups; 100% open-source compatible.

---

## 🔍 Problem Statement & Analysis

### Data Sources
- **Source 1 ($S_1$):** Deduplicated reference entity source (100% complete names and addresses).
- **Source 2 ($S_2$) & Source 3 ($S_3$):** Independent commercial vendor records with missing fields, transliterated Indian scripts, legal suffix mismatches, and transposed address tokens.
- **Ground Truth Target:** For each $S_1$ record, output a comma-separated list of matching $S_2$ and $S_3$ IDs (or empty if singleton).

### Critical Data Insights
1. **Strict Country Invariance:** Across all 7.6M+ ground-truth links, **0 cross-country matches** exist. Treating country as an open-set string partition allows isolated streaming per country (US, India, France) without false cross-country merges.
2. **Cardinality Asymmetry:** A Source 1 entity may match zero, one, or multiple records across $S_2$ and $S_3$, but **each target record belongs to at most one Source 1 entity**.
3. **Cross-Script Transliteration:** In Indian records, business names are frequently recorded in regional scripts (Devanagari, Tamil, Telugu, Gujarati, Bengali). Phonetic unidecoding restores phonetic equivalence.
4. **Macro $F_{0.5}$ Objective:** $F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$. False merges (FP) are penalized **2× more** than missed matches (FN).

---

## 🏗️ Architecture & Pipeline

```text
               +-------------------------------------------------------------+
               |  Raw Multi-Vendor TSV Streams (S1 Reference, S2, S3 Vendors)|
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               | Stage 1: Canonical Normalization & Transliteration          |
               | - Unicode canonicalization & unidecode (Indian scripts)     |
               | - Legal suffix stripping (Pvt Ltd, LLC, SARL, GmbH)         |
               | - Address token contraction & landmark cleaning             |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               | Stage 2: Country-Partitioned Inverted Index Blocking        |
               | - Multi-key union (compact name, core tokens, street/city)  |
               | - Token frequency capping (>150-300 occurrences pruned)     |
               | - Achieves >96.5% recall with ~14.5 candidates / entity    |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               | Stage 3: Pairwise 32-D Feature Extraction                   |
               | - 14 Name similarity signals (RapidFuzz C++ kernels)         |
               | - 10 Address similarity & house number match/conflict feats |
               | - 8 Cross-field interaction & source indicator terms        |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               | Stage 4: LightGBM GBDT Inference & Confidence Scoring      |
               | - 350 Trees, max depth 7, num leaves 63                     |
               | - Trained on hard negatives mined directly from candidates  |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               | Stage 5: Calibration & Bipartite 1-to-1 Target Exclusivity  |
               | - Source-specific thresholding (τ_S2 = 0.75, τ_S3 = 0.85)   |
               | - Greedy bipartite conflict resolution (highest prob wins)  |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               | Verified Submission Files (matching_results.tsv, validator) |
               +-------------------------------------------------------------+
```

---

## ⚡ Candidate Generation (Blocking)

To avoid evaluating quadratic $O(N \times M)$ pairwise combinations across 12M+ records, we employ an inverted-index blocking system with the following multi-attribute keys:

| Blocking Key | Description | Target Coverage |
| :--- | :--- | :--- |
| `compact_n` | Space-stripped alphanumeric name & sorted token name | Catches domain name concatenation (`moonboards.com` $\leftrightarrow$ `Moon Boards`) |
| `core_n` / `n2` | Legal suffix-stripped name and first 2 core name tokens | High-precision exact business root matches |
| `n_tok` | Significant individual core name tokens ($\ge 4$ characters) | Partial name variations and brand transpositions |
| `num_street` | Primary street number paired with rare street words | Same address location verification |
| `num_city` | Street number paired with city/state tokens | Addresses with alternate street naming formats |
| `name_num` | First name token paired with primary address number | Strong cross-field anchor |
| `name_city` | First name token paired with city/state tokens | Distinguishes identical franchises across cities |

Keys with extreme frequency (>150–300 occurrences) are pruned to eliminate generic noise while retaining composite keys.

---

## 🧪 Feature Engineering (32 Dimensions)

Pairwise features are computed using vectorized C++ `rapidfuzz` kernels:

- **Name Similarity Features (14):**
  - Clean name exact match, Core name exact match
  - Normalized Levenshtein distance, Jaro-Winkler distance
  - Token Sort ratio, Token Set ratio, Token Jaccard similarity, Min-overlap ratio
  - Length difference, Length ratio, 4-character prefix match
  - Character 3-gram Jaccard similarity
  - First-token match flag ($\ge 0.80$), First-token conflict flag ($< 0.50$)
- **Address Similarity Features (10):**
  - Address presence indicator, Address exact match
  - Address Levenshtein distance, Address Jaro-Winkler distance
  - Address Token Sort ratio, Address Token Set ratio, Address Token Jaccard
  - Shared numeric token count, Numeric match flag, Numeric mismatch (house number conflict)
- **Cross-Field & Source Interactions (8):**
  - Exact core name with missing target address indicator
  - Cross-field product ($\text{NameSim} \times \text{AddrSim}$), sum, max, min
  - Source 2 indicator, Source 3 indicator
  - Shared blocking keys count

---

## 🎯 Benchmark Results & Ablation Studies

Evaluated on the held-out stratified 5,000 Source-1 validation set:

| Exp ID | Architecture | Features | Global Consistency | Precision | Recall | Macro $F_{0.5}$ | False Positives | False Negatives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | LightGBM Baseline | 28 feats | No ($\tau=0.50$) | 0.9815 | 0.9521 | 0.968730 | 312 | 834 |
| **EXP-02** | LightGBM + Thresh Tuning | 28 feats | No ($\tau=0.82$) | 0.9922 | 0.9394 | 0.973518 | 129 | 1056 |
| **EXP-03** | LightGBM + Consistency | 28 feats | Yes ($\tau=0.82$) | 0.9932 | 0.9394 | 0.974255 | 112 | 1056 |
| **EXP-04** | LightGBM + Source Thresh | 28 feats | Yes ($\tau_{S2}=0.75, \tau_{S3}=0.80$) | 0.9920 | 0.9425 | 0.974534 | 133 | 1002 |
| **EXP-05** | Ablation: Name Only | 11 feats | Yes | 0.8859 | 0.8734 | 0.889718 | 1959 | 2206 |
| **EXP-06** | Ablation: Address Only | 10 feats | Yes | 0.9690 | 0.8494 | 0.919123 | 474 | 2624 |
| **EXP-07** | HistGradientBoosting | 28 feats | Yes | 0.9938 | 0.9391 | 0.974371 | 102 | 1061 |
| **EXP-08** | Deep LightGBM | 28 feats | Yes | 0.9914 | 0.9440 | 0.974164 | 143 | 976 |
| **EXP-09** | **Enhanced Feats (Production)** | **32 feats** | **Yes ($\tau_{S2}=0.75, \tau_{S3}=0.85$)** | **0.9924** | **0.9454** | **0.976105** | **126** | **951** |
| **EXP-10** | Ensemble (LGB + HGB) | 32 feats | Yes | 0.9932 | 0.9439 | 0.976653 | 113 | 977 |

---

## 📂 Directory Structure

```text
amazon-ml-challenge-2026/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── normalization.py     # Transliteration, abbreviation & suffix stripping
│       │   ├── blocking.py          # Multi-attribute inverted index candidate generator
│       │   ├── features.py          # 32-D pairwise feature extraction with RapidFuzz
│       │   ├── model.py             # Model wrappers (LightGBM, HistGradientBoosting)
│       │   ├── thresholding.py      # Macro F0.5 grid search & 1-to-1 bipartite deduplication
│       │   ├── evaluation.py        # Competition macro F0.5 scoring metric
│       │   ├── data_loader.py       # Streaming TSV reader & country partitioner
│       │   ├── output.py            # Formats submission TSVs & report summary
│       │   ├── inference.py         # End-to-end country streaming test inference engine
│       │   └── main.py              # CLI entry point with submission validation
│       ├── requirements.txt         # Pinned Python package dependencies
│       └── README.md                # Standalone reproduction guide
├── experiments/
│   ├── results.csv                  # Iterative experiment benchmark comparison table
│   ├── results.json                 # Detailed JSON records of all experiment iterations
│   └── val_s1_ids.txt               # Held-out Source 1 validation entity IDs (5,000 records)
├── models/
│   ├── final_entity_matcher.joblib  # Serialized production LightGBM model (2.4 MB)
│   └── model_metadata.json          # Optimal hyperparameters, thresholds & feature list
├── output/
│   ├── .gitkeep                     # Output placeholder & reproduction note
│   └── final_report.txt             # Benchmark metrics and test run summary
├── train_final_model.py             # Script to fit model and optimize thresholds on validation set
├── save_best_model.py               # Serializes final model bundle and metadata
├── create_validation_split.py       # Generates stratified validation split
├── run_experiments.py               # Automated benchmark experiment suite
├── run_experiments_v2.py            # Advanced benchmark experiment suite
├── test_*.py                        # Comprehensive testing & validation scripts
├── inspect_*.py                     # Forensic error & false-positive analysis utilities
├── Documentation_template.md        # Official filled methodology submission document
├── PS.txt                           # Official competition problem statement & rules
└── README.md                        # Master repository documentation
```

---

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Akash-bardia/amazon-ml-challenge-2026.git
   cd amazon-ml-challenge-2026
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r code/business_entity_resolution/requirements.txt
   ```

---

## 🔁 End-to-End Reproduction Guide

### Option 1: Run Full Test Inference (Using Pre-trained Model)
The pre-trained production model `models/final_entity_matcher.joblib` (2.4 MB) is included in this repository. To run end-to-end candidate blocking, feature extraction, and inference over test data:

```bash
python code/business_entity_resolution/src/main.py \
    --test-dir student_resource/dataset/test \
    --output-dir output
```

This generates:
- `output/matching_results.tsv` (Leaderboard submission file)
- `output/candidate_pairs.tsv` (Blocking candidate set file)
- `output/final_report.txt` (Run summary report)

### Option 2: Retrain the Model from Scratch
To reproduce the training and validation pipeline on raw training data:

```bash
python train_final_model.py
```

### Option 3: Verify Submission Compliance
To run the competition validation suite locally:

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
Expected output: **`PASS`**

---

## 📜 Submission Compliance

- **Permissible Models:** Employs standard LightGBM trees ($< 8\text{B}$ parameters, Apache 2.0 license).
- **Strictly Self-Contained:** Zero external API or database lookups.
- **TSV Standards:** Explicit tab separation, singletons correctly output with blank `matched_entity_ids`, no duplicate IDs.
- **Bipartite Exclusivity:** Enforces target uniqueness, eliminating impossible multi-source entity conflicts.

---

## 👥 Authors
- **Akash Bardia** ([@Akash-bardia](https://github.com/Akash-bardia))
