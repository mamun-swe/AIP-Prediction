# AIP-ML-Stack
### Anti-Inflammatory Peptide Identification — Hybrid Feature Stacking Ensemble

A rigorous, reproducible ML pipeline for AIP identification combining hand-crafted physicochemical features, evolutionary features, and frozen protein language model (PLM) embeddings in a tuned stacking ensemble (XGBoost + LightGBM + SVM).

---

## Project Structure

```
aip-ml-stack/
├── config.yaml                     ← central configuration
├── requirements.txt
├── data/
│   ├── raw/                        ← put your FASTA or CSV files here
│   ├── processed/                  ← cleaned dataset (auto-generated)
│   └── splits/                     ← train.csv, test.csv (auto-generated)
├── src/
│   ├── data/
│   │   ├── preprocessor.py         ← sequence cleaning, redundancy removal
│   │   └── splitter.py             ← stratified train/test split
│   ├── features/
│   │   ├── handcrafted.py          ← AAC, DPC, PAAC, GAAC, CTDC/T/D, QSO
│   │   └── plm_embeddings.py       ← ESM-2 (frozen) mean-pooled embeddings
│   ├── fusion/
│   │   └── selector.py             ← variance + correlation + XGB top-k selection
│   ├── models/
│   │   └── stacking.py             ← XGB+LGBM+SVM stacking with Optuna tuning
│   └── evaluation/
│       └── metrics.py              ← ACC, Sn, Sp, MCC, AUC, bootstrap CI, plots
├── scripts/
│   ├── 01_prepare_data.py          ← data loading, cleaning, splitting
│   ├── 02_extract_features.py      ← feature extraction
│   ├── 03_select_features.py       ← fusion + two-stage selection
│   ├── 04_train_model.py           ← train stacking ensemble
│   ├── 05_evaluate.py              ← independent test evaluation
│   └── 06_validate_all.py          ← 8-test validation suite ✓
└── results/
    ├── features/                   ← saved feature arrays
    ├── models/                     ← saved ensemble + hyperparameters
    └── evaluation/                 ← metrics JSON, CSV, ROC/PR curves
```

---

## Installation

```bash
pip install -r requirements.txt
pip install fair-esm   # for PLM embeddings
```

---

## Getting the Dataset

**Option A — Manavalan 2018 Benchmark (recommended):**
1. Go to http://www.thegleelab.org/AIPpred → Download dataset
2. Save as `data/raw/aip_positive.fasta` and `data/raw/aip_negative.fasta`
3. Run: `python scripts/01_prepare_data.py --mode fasta`

**Option B — Your own CSV:**
```bash
python scripts/01_prepare_data.py --mode csv --csv data/raw/dataset.csv
```

**Option C — Sample dataset (pipeline testing only):**
```bash
python scripts/01_prepare_data.py --mode sample
```

---

## Running the Full Pipeline

```bash
# Step 0 — Validate all components (always run this first)
python scripts/06_validate_all.py

# Step 1 — Prepare data
python scripts/01_prepare_data.py --mode fasta

# Step 2 — Extract features (add --hc_only to skip PLM if fair-esm not installed)
python scripts/02_extract_features.py

# Step 3 — Fuse and select features
python scripts/03_select_features.py --n_features 500

# Step 4 — Train ensemble
python scripts/04_train_model.py --n_optuna_trials 50 --run_cv

# Step 5 — Evaluate on independent test set
python scripts/05_evaluate.py
```

---

## Feature Dimensions

| Feature group | Method       | Dimensions |
|---------------|--------------|-----------|
| AAC           | Hand-crafted | 20        |
| DPC           | Hand-crafted | 400       |
| PAAC          | Hand-crafted | 30        |
| GAAC          | Hand-crafted | 5         |
| CTDC          | Hand-crafted | 21        |
| CTDT          | Hand-crafted | 21        |
| CTDD          | Hand-crafted | 105       |
| QSO           | Hand-crafted | 40        |
| ESM-2 small   | PLM (frozen) | 480       |
| ESM-2 medium  | PLM (frozen) | 640       |
| **Total**     |              | **1,762** |
| **After selection** |        | **500**   |

---

## Evaluation Protocol

This pipeline reports **both** cross-validation and independent test metrics:

| Metric    | CV (10-fold) | Independent test |
|-----------|-------------|-----------------|
| Accuracy  | ✓ reported  | ✓ reported       |
| Sensitivity | ✓         | ✓               |
| Specificity | ✓         | ✓               |
| MCC       | ✓ reported  | ✓ reported       |
| AUC-ROC   | ✓ reported  | ✓ reported       |
| 95% CI    | —           | ✓ bootstrap      |

**Critical:** All feature selection and scaling is fitted on training data only. 
The test set is NEVER seen during fitting of any component.

---

## Novelty Claims

1. **Representation**: First AIP paper to use frozen ESM-2 as a feature extractor feeding a traditional ML stacking ensemble
2. **Fusion**: Three-layer hybrid features (physicochemical + QSO + PLM) with MCC-optimized selection
3. **Evaluation integrity**: Dual reporting (CV + independent test) with explicit data-leakage prevention
