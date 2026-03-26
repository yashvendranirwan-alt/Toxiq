# ⚗️ ToxIQ — Drug Toxicity Intelligence Platform

> **AI-powered multi-task drug toxicity prediction** using Morgan Fingerprints, XGBoost, and SHAP explainability. Built for the **CODECURE AI Hackathon — Track A**.

---

## Table of Contents

1. [What is ToxIQ?](#what-is-toxiq)
2. [System Architecture](#system-architecture)
3. [The Biology — Why These 12 Assays?](#the-biology)
4. [Data Flow](#data-flow)
5. [Cheminformatics Deep Dive](#cheminformatics-deep-dive)
6. [Model Architecture](#model-architecture)
7. [Explainable AI — SHAP](#explainable-ai--shap)
8. [Quick Start](#quick-start)
9. [Project Structure](#project-structure)
10. [Interpreting Results](#interpreting-results)
11. [Technical Decisions & Trade-offs](#technical-decisions--trade-offs)

---

## What is ToxIQ?

ToxIQ is an **industrial-grade drug toxicity prediction system** that answers the question:

> *"Given a new chemical compound, does it carry toxicity risk — and WHY?"*

It processes molecular SMILES strings → converts them to chemically meaningful feature vectors → predicts toxicity across 12 critical biological assays → explains every prediction using SHAP.

**Real-world impact:** Drug development failure due to unexpected toxicity costs the industry $2.5B+ per failed compound. Early computational screening with tools like ToxIQ can identify structural alerts **before wet-lab synthesis**, saving resources and protecting patients.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ToxIQ Architecture                          │
│                                                                     │
│  INPUT                    FEATURE LAYER              OUTPUT         │
│                                                                     │
│  SMILES String            Morgan FP          ┌─── NR-AR prediction │
│  "CC(=O)Oc1cc..."  ──►   (2048 bits)    ──►  ├─── NR-AhR          │
│                     ┐    +                    ├─── NR-ER            │
│  RDKit Mol Object   │    Physicochemical      ├─── SR-p53           │
│  (validated, canon.)┘    Descriptors    ──►  ├─── ... (12 total)   │
│                          (13 features)        └─── SR-MMP           │
│                                                                     │
│                    EXPLANATION LAYER                                │
│                    SHAP TreeExplainer                               │
│                    ↓ Which bits drove it? ↓                        │
│                    Structural Alert Visualization                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | File | Role |
|---|---|---|
| Data Pipeline | `src/data_pipeline.py` | SMILES validation, featurization, quality control |
| Model Trainer | `src/model_trainer.py` | Multi-task XGBoost, CV, serialization |
| Training Entry | `train.py` | CLI for end-to-end training |
| Dashboard | `app.py` | Streamlit UI with interactive analysis |

---

## The Biology — Why These 12 Assays?

The Tox21 dataset covers two families of toxicological endpoints:

### Nuclear Receptor (NR) Pathway Assays

Nuclear receptors are proteins that regulate gene expression when activated by small molecules. **Endocrine-disrupting chemicals** work by hijacking these receptors.

| Assay | Biological Target | Clinical Concern |
|---|---|---|
| **NR-AR** | Androgen Receptor | Male hormone disruption; linked to reproductive toxicity |
| **NR-AR-LBD** | AR Ligand Binding Domain | More specific AR binding measurement |
| **NR-AhR** | Aryl Hydrocarbon Receptor | Activated by dioxins/PCBs; carcinogenicity |
| **NR-Aromatase** | Aromatase Enzyme | Disrupts estrogen synthesis; breast cancer risk |
| **NR-ER** | Estrogen Receptor α | Female hormone disruption; widespread endocrine effects |
| **NR-ER-LBD** | ER Ligand Binding Domain | Refined ER binding specificity |
| **NR-PPAR-gamma** | PPAR-γ | Metabolic disruption; weight gain, insulin resistance |

### Stress Response (SR) Pathway Assays

These measure cellular stress responses — signals that a compound is damaging cells through various mechanisms.

| Assay | Biological Target | Clinical Concern |
|---|---|---|
| **SR-ARE** | Antioxidant Response Element | Oxidative stress; NRF2 pathway activation |
| **SR-ATAD5** | ATAD5 (DNA repair marker) | DNA damage indicator; genotoxicity |
| **SR-HSE** | Heat Shock Element | Proteotoxic stress; protein misfolding |
| **SR-MMP** | Mitochondrial Membrane Potential | Mitochondrial toxicity; energy disruption |
| **SR-p53** | p53 Tumor Suppressor | DNA damage → cancer risk |

---

## Data Flow

```
Tox21 CSV
    │
    ▼
load_tox21_dataset()
    │ • Validate CSV structure
    │ • Log class distributions
    │ • Handle column naming variations
    ▼
build_feature_matrix()
    │
    ├── For each compound:
    │       │
    │       ├── parse_smiles() → RDKit Mol
    │       │   (validates, sanitizes, canonicalizes)
    │       │
    │       ├── compute_morgan_fingerprint()
    │       │   → 2048-bit ECFP4 vector
    │       │   (radius=2, chirality=True)
    │       │
    │       └── compute_physicochemical_descriptors()
    │           → 13 pharmacological descriptors
    │           (MW, LogP, TPSA, HBD, HBA, ...)
    │
    └── Concatenate → X matrix (N × 2061)
    
build_label_matrix()
    │ • Extract 12 assay columns
    │ • Preserve NaN for unlabeled samples
    └── Y matrix (N × 12)
    
                ↓
        
train_all_tasks()
    │
    ├── For each assay (task):
    │       │
    │       ├── Filter to labeled samples only (drop NaN rows)
    │       ├── Compute scale_pos_weight (class imbalance)
    │       ├── 5-fold StratifiedKFold CV → ROC-AUC, PR-AUC
    │       ├── Train final XGBoost on all labeled data
    │       └── Save model + scaler to models/
    │
    └── Save: feature_names.pkl, assay_names.json, training_metrics.json

                ↓
        
streamlit run app.py
    │
    ├── Input: SMILES string
    ├── Featurize → 2061-dim vector
    ├── Load 12 XGBoost models
    ├── predict_proba() × 12 → probability scores
    ├── compute_shap_values() → feature attributions
    └── Render: molecule image, risk table, SHAP chart, descriptors
```

---

## Cheminformatics Deep Dive

### Why Not NLP on SMILES?

SMILES is a **text encoding** of molecular topology — but it's not a language in the NLP sense.

The same molecule (isobutane) can be written as:
```
CC(C)C    or    C(C)(C)C    or    CCC(C)    ← same molecule, different strings!
```

An NLP model would treat these as entirely different entities. **Morgan Fingerprints don't have this problem** — they're computed from the molecular graph, not the string.

### How Morgan Fingerprints Work

```
         Radius 0:      Radius 1:          Radius 2:
         
         C              C-C                C(-C)(-C)=O
         ↓              ↓                  ↓
         Hash           Hash               Hash
         → bit 123      → bit 847          → bit 1204
```

Each atom generates a unique identifier by hashing its atomic environment at increasing radii. At radius 2 (ECFP4), we capture the full context of each atom within 4 bonds.

**Result:** A 2048-bit binary vector where each `1` means "this structural pattern is present."

### Physicochemical Descriptors

| Descriptor | Range | Pharmacological Meaning |
|---|---|---|
| **MolWt** | 0–2000 Da | Membrane permeability threshold ~500 Da |
| **MolLogP** | -5 to +15 | Lipophilicity; > 5 → accumulates in fat tissue |
| **TPSA** | 0–500 Ų | < 90 → oral absorption; > 140 → poor CNS |
| **HBD** | 0–20 | H-bond donors limit membrane crossing |
| **HBA** | 0–30 | H-bond acceptors; too many → poor absorption |
| **Fsp3** | 0–1 | 3D-character; < 0.25 → flat, often more toxic |
| **ArRings** | 0–10 | PAHs (multiple aromatics) are classic toxicophores |

---

## Model Architecture

### Why XGBoost for Fingerprints?

XGBoost (Gradient Boosted Decision Trees) excels on **sparse, high-dimensional binary data** — exactly what fingerprints are:

1. **Built-in feature selection**: Trees naturally ignore uninformative bits (most of 2048)
2. **Non-linearity**: Captures complex structure-activity relationships that linear models miss
3. **Calibrated probabilities**: After platt scaling, probabilities are reliable for risk assessment
4. **Speed**: `tree_method='hist'` processes 2048-dim vectors fast
5. **Benchmark performance**: XGBoost achieved top results in the original Tox21 challenge

### Multi-Task Strategy

```
                  Shared Feature Space (2061 dims)
                           │
         ┌─────────────────┼─────────────────────┐
         ▼                 ▼                     ▼
    XGB Model₁        XGB Model₂    ...    XGB Model₁₂
    (NR-AR)           (NR-ER)               (SR-p53)
         │                 │                     │
    P(toxic|AR)      P(toxic|ER)          P(toxic|p53)
```

Each model is specialized but benefits from:
- Shared hyperparameters tuned for cheminformatics (transfer of knowledge)
- Same feature space (topological + physicochemical)
- Independent optimization per task (correct handling of partial labels)

### Handling Class Imbalance

Tox21 is severely imbalanced (typically 5-20% active compounds):

```python
scale_pos_weight = count(negative) / count(positive)
# Example: 1000 inactive, 50 active → scale_pos_weight = 20
# XGBoost treats each active compound as if it were 20 inactive compounds
```

Why **not** SMOTE? Synthetic minority oversampling creates interpolated fingerprint vectors that don't correspond to real molecules. Class weighting is pharmacologically sound.

---

## Explainable AI — SHAP

### The Problem with Black-Box Models in Drug Safety

A model that says "this compound is 87% likely to be toxic" without explanation is **clinically useless**. A medicinal chemist needs to know:
- Which part of the molecule is dangerous?
- What structural modification would reduce toxicity?
- Is this a real alert or a model artifact?

### SHAP Provides the Answer

```
Base prediction (average toxicity rate)
+ SHAP(FP_bit_847)     = +0.12  "aromatic N adjacent to ring"
+ SHAP(FP_bit_1204)    = +0.08  "halogen on aromatic ring"
+ SHAP(MolLogP)        = +0.06  "high lipophilicity"
+ SHAP(FP_bit_033)     = -0.04  "hydroxyl group — reduces some toxicity"
+ SHAP(TPSA)           = -0.02  "moderate polarity helps"
= Final prediction: 87% toxic
```

The chemist sees: "The nitroaromatic fragment (bits 847+1204) is driving this prediction. Replacing the aromatic nitrogen with a carbon should reduce AhR binding."

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/your-org/toxiq.git
cd toxiq
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Download from Kaggle (requires kaggle CLI):
kaggle datasets download epicskills/tox21-dataset
unzip tox21-dataset.zip
mv tox21.csv ./
```

Or manually download from: https://www.kaggle.com/datasets/epicskills/tox21-dataset

### 3. Train Models

```bash
# Full training with cross-validation (~5-15 min depending on hardware):
python train.py --data tox21.csv --model-dir models/

# Faster (no CV metrics):
python train.py --data tox21.csv --no-cv
```

Training output:
```
2024-01-15 10:23:01 | INFO | ToxIQ.Train | PHASE 1: Data Engineering
2024-01-15 10:23:15 | INFO | ToxIQ.DataPipeline | Feature matrix: (8014, 2061)
2024-01-15 10:23:15 | INFO | ToxIQ.Train | PHASE 2: Multi-Task Training
...
CROSS-VALIDATION PERFORMANCE SUMMARY
====================================================
Assay                 ROC-AUC      ± Std     PR-AUC
----------------------------------------------------
NR-AR                   0.872      0.021      0.312
NR-ER                   0.831      0.018      0.284
NR-AhR                  0.891      0.015      0.445
SR-p53                  0.862      0.019      0.398
...
MEAN                    0.856
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`

### 5. Demo Mode (No Training Required)

The dashboard runs in **demo mode** without trained models, showing heuristic predictions. Useful for UI development and presentations.

```bash
streamlit run app.py  # Will auto-detect missing models and enter demo mode
```

---

## Project Structure

```
toxiq/
├── app.py                      # Streamlit dashboard (entry point)
├── train.py                    # Training CLI entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py        # Data engineering + cheminformatics
│   └── model_trainer.py        # ML training + inference + SHAP
│
└── models/                     # Created by train.py
    ├── NR-AR_model.pkl
    ├── NR-AR_scaler.pkl
    ├── ... (one pair per assay)
    ├── feature_names.pkl
    ├── assay_names.json
    └── training_metrics.json
```

---

## Interpreting Results

### Risk Probability Thresholds

| Probability | Risk Level | Recommended Action |
|---|---|---|
| 0–30% | 🟢 Low | Continue development; monitor |
| 30–60% | 🟡 Moderate | Expert review; consider structural modification |
| 60–100% | 🔴 High | Structural alert — strong toxicity signal; redesign |

**Important**: These are **probabilistic predictions**, not absolute determinations. Always validate with wet-lab assays. The model provides early-stage screening to prioritize compounds.

### Reading SHAP Charts

- **Red bars** (positive SHAP): This feature **increases** toxicity probability
- **Blue bars** (negative SHAP): This feature **decreases** toxicity probability
- **FP_bit_XXX**: A specific molecular substructure; decode with RDKit's `GetMorganFingerprintAsBitVect(mol, 2, bitInfo=info)`
- **Named descriptors** (LogP, TPSA, etc.): Global molecular property contributing to prediction

### Lipinski Violations

A compound with 2+ Lipinski violations is likely not orally bioavailable — even if it's non-toxic, it may not be a viable drug candidate. ToxIQ flags these violations for comprehensive ADMET profiling.

---

## Technical Decisions & Trade-offs

| Decision | Alternative Considered | Rationale |
|---|---|---|
| **Morgan FP (ECFP4)** | Graph Neural Networks (GNNs) | ECFP4 + XGBoost outperforms GNNs on Tox21 in most benchmarks; much simpler to deploy |
| **XGBoost** | Random Forest, SVM, Deep Learning | Best accuracy-speed tradeoff; native SHAP support via TreeExplainer |
| **Independent per-task models** | True multi-task neural network | Handles incomplete labels (NaN) correctly; XGBoost doesn't support joint multi-output natively |
| **class weighting** | SMOTE oversampling | SMOTE on binary fingerprints creates chemically meaningless synthetic molecules |
| **ROC-AUC as primary metric** | Accuracy, F1 | AUC is threshold-independent and handles class imbalance correctly |
| **5-fold stratified CV** | Train/val split | More robust estimate; stratification preserves class ratios |

---

## Expected Performance

Based on published Tox21 benchmarks with ECFP4 + XGBoost:

| Assay | Expected ROC-AUC |
|---|---|
| NR-AhR | 0.87–0.93 |
| SR-p53 | 0.83–0.89 |
| NR-AR | 0.83–0.88 |
| SR-MMP | 0.89–0.93 |
| *Average* | *0.84–0.89* |

These are consistent with the top-performing methods in the 2014 Tox21 Data Challenge.

---

## Dataset Citation

**Tox21 Dataset:**
> Huang R, et al. "Tox21Challenge to Build Predictive Models of Nuclear Receptor and Stress Response Pathways as Mediated by Exposure to Environmental Chemicals and Drugs." *Frontiers in Environmental Science* (2016).

---

## Team life-link

— Track A: Drug Toxicity Prediction (Pharmacology + AI)

*"The best model is not the most complex one — it's the one that provides the most actionable insight to the scientists who need to make decisions."*
