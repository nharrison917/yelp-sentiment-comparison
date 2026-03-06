# Sentiment Modeling & Text Risk Classification

## Overview
This project compares classical machine learning models and transformer-based deep learning models for multi-class sentiment classification on the Yelp Restaurant Reviews dataset.

The primary goal is not only to improve predictive performance, but to analyze *why* models behave differently — particularly under class imbalance and mixed-polarity linguistic structures.

---

## Problem Framing

The dataset contains restaurant reviews labeled into three sentiment categories:

- **0 — Negative**
- **1 — Neutral**
- **2 — Positive**

The data is heavily imbalanced (~68% Positive), making overall accuracy misleading. Therefore, **macro-averaged F1 score** is used as the primary evaluation metric.

---

## Experimental Design

### Dataset Handling

- Source: HuggingFace (`mrcaelumn/yelp_restaurant_review_labelled`)
- Stratified sampling used to preserve class proportions
- Controlled with fixed random seed for reproducibility
- Final full training split:
  - Train: 240k
  - Validation: 30k
  - Test: 60k

---

## Models Compared

### 1. Classical Baselines

**TF‑IDF + Multinomial Naive Bayes**  
**TF‑IDF + Multinomial Logistic Regression (class-balanced)**

TF‑IDF configuration:
- 1–2 n‑grams
- max_features=100,000
- English stopword removal
- min_df=50
- max_df=0.95

These models serve as interpretable lexical baselines.

---

### 2. Transformer Model

**BERT (bert-base-uncased)** fine-tuned using HuggingFace Trainer API.

Key training decisions:
- Weighted cross-entropy loss (class-balanced)
- GPU fine-tuning
- Macro-F1 used for evaluation
- Sequence length experimentation (128 vs 256 tokens)
- Truncation strategy comparison (first tokens vs last tokens)

---

## Final Model Configuration

Full dataset training (DEV_MODE = False)

- Max sequence length: **256**
- Truncation: **Left (preserve final tokens)**
- Class-weighted loss
- 2 epochs

### Final Test Performance (60k samples)

| Class      | Precision | Recall | F1   |
|------------|-----------|--------|------|
| Negative   | 0.8920    | 0.8646 | 0.8780 |
| Neutral    | 0.5747    | 0.5885 | 0.5816 |
| Positive   | 0.9528    | 0.9579 | 0.9553 |

**Macro F1: 0.8050**  
**Accuracy: 0.8972**

---

## Key Findings

### 1. Transformer > Classical Models

Macro F1 progression:

- Naive Bayes → ~0.60
- Logistic Regression → ~0.73
- BERT (weighted, full data) → **0.805**

Transformer models substantially improve minority-class balance and overall macro performance.

---

### 2. Class Weighting Improves Minority Recall

Unweighted BERT over-optimized the majority class (Positive).

Applying balanced class weights:
- Improved Neutral recall
- Slightly reduced overall accuracy
- Increased macro-F1

This demonstrates the tradeoff between global accuracy and balanced class performance.

---

### 3. Context Length & Truncation Matter

Experiments comparing:
- 128 vs 256 tokens
- Truncating first tokens vs last tokens

Findings:

- Increasing sequence length improved polarity separation.
- Preserving final review tokens slightly improved macro-F1.
- Review conclusions often contain summary sentiment signals.

This highlights how document structure interacts with transformer truncation strategy.

---

### 4. Neutral Class Reflects Structural Ambiguity

Despite improvements, Neutral F1 remains significantly lower than Positive or Negative.

Error analysis reveals:
- Mixed-polarity reviews ("good food but slow service")
- Moderate intensity language
- Comparative phrasing

Comparative language appears slightly more often in misclassified reviews (~4.5%) than correctly classified ones (~2.8%), but is not the dominant error driver.

The core challenge is boundary ambiguity rather than pure class imbalance.

---

## Interpretation

The most important insight from this project is:

> Transformer models appear to learn sentiment *intensity* rather than discrete rating categories.

Many Neutral errors arise not from obvious polarity failure, but from reviews containing balanced or moderate sentiment signals. This suggests structural overlap in labeling rather than simple modeling deficiency.

---

## Project Structure

```
src/
 ├── config.py
 ├── data_loader.py
 ├── preprocessing.py
 ├── classical_models.py
 ├── bert_model.py
 ├── evaluation.py
 └── interpretability.py
```

Key features:
- Reproducible environment (conda + pinned requirements)
- GPU training support
- Config-driven experimentation
- Separate training and analysis modes

---

## How to Run

1. Create environment:

```
Install PyTorch with CUDA support following the official PyTorch instructions for your system before running the project.

Developed and tested using:
- Python 3.11
- PyTorch 2.5.x (CUDA enabled)
- Transformers 5.3.x
- HugginFace Datasets 2.x


conda create -n yelp-nlp python=3.11
conda activate yelp-nlp
pip install -r requirements.txt
```

2. Train (optional):

Set in `config.py`:

```
RUN_TRAINING = True
DEV_MODE = False
```

Then:

```
python src/main.py
```

3. Analysis-only mode:

```
RUN_TRAINING = False
```

Loads saved predictions without retraining.

---

## Future Work

- Ordinal sentiment modeling instead of categorical
- Longer context windows (512 tokens)
- Sentence-level aggregation models
- Confidence calibration analysis
- SHAP comparison with LIME

---

## Summary

This project demonstrates:

- Careful metric selection under imbalance
- Model comparison across algorithmic families
- GPU-based transformer fine-tuning
- Structured error analysis
- Linguistic boundary investigation

The final system achieves strong macro-balanced performance while revealing deeper insights about sentiment intensity and label ambiguity in real-world review data.

