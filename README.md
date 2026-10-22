# Yelp Review Sentiment: Classical Baselines vs. Fine-Tuned BERT

## Overview
This project compares classical machine learning models and a fine-tuned transformer for three-class sentiment classification of Yelp restaurant reviews.

The goal is not only to improve predictive performance, but to understand *why* the models behave differently — particularly under class imbalance, and on the Neutral class, where every model struggles.

### Results at a glance (test set, n = 60,000)

| Model | Macro F1 | Accuracy | Negative F1 | Neutral F1 | Positive F1 |
|---|---|---|---|---|---|
| TF‑IDF + Naive Bayes | 0.6259 | 0.8386 | 0.7924 | 0.1725 | 0.9127 |
| TF‑IDF + Logistic Regression (class-balanced) | 0.7094 | 0.7914 | 0.8086 | 0.4403 | 0.8793 |
| **BERT (class-weighted loss)** | **0.8050** | **0.8972** | **0.8780** | **0.5816** | **0.9553** |

BERT macro F1 95% bootstrap CI: **[0.801, 0.809]** (1,000 resamples of the test set).
All metrics are written to `outputs/results.json` by `src/main.py`.

---

## Problem Framing

The dataset labels each review by its star rating:

| Label | Stars | Share of test set |
|---|---|---|
| 0 — Negative | 1–2 | 20.5% |
| 1 — Neutral | 3 | 11.2% |
| 2 — Positive | 4–5 | 68.3% |

The data is heavily imbalanced, so overall accuracy is misleading: a model can score well by favouring Positive while rarely predicting Neutral. **Macro-averaged F1** — which weights each class equally — is the primary metric.

---

## Experimental Design

### Dataset Handling

- Source: HuggingFace [`mrcaelumn/yelp_restaurant_review_labelled`](https://huggingface.co/datasets/mrcaelumn/yelp_restaurant_review_labelled) (4.1M train / 1.0M test reviews)
- Stratified subsample to preserve class proportions, `random_state=42` throughout
- Train and validation drawn from the dataset's train split; test drawn from its separate test split
  - Train: 240,000
  - Validation: 30,000 (used for BERT checkpoint selection only)
  - Test: 60,000

All models are evaluated on the same 60,000 test reviews; `main.py` asserts that saved BERT predictions align row-for-row with the regenerated test split.

---

## Models Compared

### 1. Classical Baselines

- **TF‑IDF + Multinomial Naive Bayes** (no class weighting)
- **TF‑IDF + Multinomial Logistic Regression** (`class_weight="balanced"`)

TF‑IDF configuration: 1–2 n‑grams, `max_features=100,000`, English stopword removal, `min_df=50`, `max_df=0.95`.

These serve as interpretable lexical baselines.

### 2. Transformer Model

**BERT (`bert-base-uncased`)** fine-tuned with the HuggingFace Trainer API.

| Setting | Value |
|---|---|
| Loss | Cross-entropy with balanced class weights |
| Max sequence length | 256 tokens |
| Truncation | Right (first 256 tokens kept) |
| Batch size / learning rate | 8 / 2e‑5 |
| Epochs | 2 scheduled; best checkpoint by validation loss was **epoch 1** |

Validation macro F1 was essentially flat between epochs (0.8078 → 0.8083) while validation loss rose (0.516 → 0.593), a mild overfitting signal that made the epoch‑1 checkpoint the selected model.

---

## Key Findings

### 1. BERT substantially outperforms lexical baselines

BERT improves macro F1 by **+0.096 over Logistic Regression** and **+0.179 over Naive Bayes**, with the largest gain on the hardest class: Neutral F1 rises from 0.17 (NB) to 0.44 (LR) to 0.58 (BERT). It also has the highest accuracy, so the macro-F1 gain does not come at the cost of overall performance.

### 2. Class balancing trades accuracy for minority-class recall

The two classical models illustrate the tradeoff directly:

| | Accuracy | Neutral recall | Neutral precision |
|---|---|---|---|
| Naive Bayes (unweighted) | **0.839** | 0.103 | 0.531 |
| Logistic Regression (balanced) | 0.791 | **0.652** | 0.333 |

Unweighted Naive Bayes achieves higher accuracy largely by rarely predicting Neutral. Balanced Logistic Regression recovers most Neutral reviews at the cost of many false Neutral predictions and lower accuracy — but higher macro F1. (The two models also differ in algorithm, so this illustrates the pattern rather than isolating the effect of weighting.) BERT with class-weighted loss avoids most of this tradeoff, with balanced Neutral precision and recall (0.57 / 0.59).

### 3. Context length (exploratory)

On a development subset (80k train / 10k validation), increasing max sequence length from 128 to 256 tokens raised epoch‑1 validation macro F1 from **0.762 to 0.780**. Batch size was reduced from 16 to 8 at the same time to fit GPU memory, so the gain cannot be attributed to context length alone. For scale: 34.7% of test reviews exceed 128 tokens, versus 10.9% exceeding 256.

### 4. Neutral errors are boundary errors, not polarity errors

Of BERT's 6,169 test errors, **92.7% are between adjacent classes** (Negative↔Neutral or Neutral↔Positive). Only 453 (7.3%) confuse Negative with Positive.

Reading misclassified Neutral reviews qualitatively suggests common patterns:
- Mixed-polarity reviews ("good food but slow service")
- Moderate-intensity language
- Comparative phrasing

Comparative phrases (e.g. "better than", "compared to") appear somewhat more often in misclassified reviews (4.5%) than in correctly classified ones (2.8%), but they are present in only a small fraction of errors and are not the dominant driver.

---

## Interpretation

Neutral is not a distinct kind of opinion: it is the 3-star point on an ordinal scale, and the line between a 3- and a 4-star review is a matter of individual reviewer judgement. The error structure is consistent with this: BERT almost never confuses polarity, but frequently disagrees with the reviewer about *how* positive or negative a review is.

This suggests the remaining Neutral error is driven substantially by genuine label ambiguity — a ceiling set by the labelling scheme — rather than purely by model capacity, and that an ordinal formulation may be a better fit than flat three-class classification.

---

## Limitations

- **Single training run per configuration.** The bootstrap CI reflects test-set sampling only, not variance across random seeds or training runs.
- **Subsample.** Models were trained on ~6% of the available data (240k of 4.1M reviews) to fit the compute budget.
- **Truncation.** The model sees only the first 256 tokens; 10.9% of test reviews are longer. A head-vs-tail truncation comparison was planned, but the tail-truncation setting was passed in a way the tokenizer silently ignores, so that comparison is not reported. It is listed under Future Work.
- **Checkpoint selection.** The best checkpoint was chosen by (class-weighted) validation loss rather than validation macro F1. Selecting by macro F1 would have chosen epoch 2 instead, a validation difference of only 0.0005.
- **Context-length comparison is confounded** with batch size (see Finding 3) and was run on a development subset, at epoch 1 only.
- **Classical baselines were not tuned** beyond the TF‑IDF configuration above, and Naive Bayes was not class-weighted.
- **Error analysis is partly qualitative.** Error patterns come from reading examples; the comparative-language check uses a small keyword list.
- **Label noise.** Star ratings are a reviewer's subjective choice; the same text could plausibly receive 3 or 4 stars from different people.
- **Calibration** of predicted probabilities was not assessed.

---

## Project Structure

```
src/
 ├── config.py            # Dataset sizes, BERT hyperparameters, run flags
 ├── data_loader.py       # HuggingFace dataset loading and stratified splits
 ├── preprocessing.py     # Text/label extraction utilities
 ├── classical_models.py  # TF-IDF + Naive Bayes and Logistic Regression
 ├── bert_model.py        # BERT fine-tuning with weighted cross-entropy loss
 ├── evaluation.py        # Classification report, macro F1, metrics dict
 ├── interpretability.py  # Confusion breakdown, error structure, bootstrap CI
 ├── analysis.py          # Standalone post-hoc analysis (loads saved predictions)
 ├── confusion_matrix.py  # Generates interactive Plotly confusion matrix HTML
 └── main.py              # End-to-end pipeline: classical models + BERT evaluation
outputs/
 ├── results.json                 # Test metrics for all three models
 ├── analysis_output.txt          # Saved output of analysis.py (error structure, CI, example errors)
 └── bert_confusion_matrix.html   # Interactive confusion matrix (Plotly)
```

---

## How to Run

1. Create the environment:

```
conda env create -f environment.yml
conda activate yelp-nlp
```

Developed and tested with Python 3.11, PyTorch 2.5.1 (CUDA 12.1), Transformers 5.3.0, and HuggingFace Datasets 4.x.

2. Run the pipeline from the project root:

```
python src/main.py
```

This trains and evaluates the classical models, evaluates BERT, and writes `outputs/results.json`.

By default (`RUN_TRAINING = False` in `config.py`) BERT results are loaded from `bert_test_predictions.csv`. **The predictions file and model weights are not included in this repository**, because of their size and because the predictions file contains the full text of 60,000 Yelp reviews, which Yelp's dataset terms restrict from redistribution. Set `RUN_TRAINING = True` to fine-tune BERT and regenerate them (several hours on an 8 GB GPU).

3. Post-hoc analysis and visualisation (require `bert_test_predictions.csv`):

```
python src/analysis.py
python src/confusion_matrix.py
```

The output of `analysis.py` from the reported run is saved in `outputs/analysis_output.txt`, so the error analysis can be reviewed without regenerating predictions.

---

## Future Work

- Controlled head-vs-tail truncation comparison (`tokenizer.truncation_side = "left"`)
- Controlled context-length comparison (128 / 256 / 512) with a fixed effective batch size via gradient accumulation; 512 tokens would cover all but 1.6% of reviews
- Ordinal formulation (e.g. ordinal regression or regression on star rating) instead of flat classification
- Checkpoint selection by validation macro F1, and multiple seeds to measure run-to-run variance
- Confidence calibration analysis
- Token-level attribution (e.g. SHAP) to compare what BERT and Logistic Regression attend to

---

## Summary

This project demonstrates:

- Metric selection appropriate to class imbalance
- Model comparison across algorithmic families on an identical, verified test set
- GPU-based transformer fine-tuning with class-weighted loss
- Quantified error analysis linking model errors to the structure of the labels
- Honest reporting of experimental confounds and limitations

The final model reaches 0.805 macro F1, and its errors point to a clear next step: treating review sentiment as an ordinal scale rather than three independent categories.

---

*Developed with assistance from [Claude Code](https://claude.com/claude-code) — code review, bug fixes, results verification, and documentation.*
