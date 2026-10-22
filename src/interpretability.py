# src/interpretability.py

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def analyze_confusion(test_df):
    """
    Print confusion breakdown by class.
    """

    print("\nConfusion Breakdown:\n")

    for true_label in sorted(test_df["label"].unique()):
        subset = test_df[test_df["label"] == true_label]

        counts = subset["bert_pred"].value_counts().sort_index()

        print(f"True: {LABEL_MAP[true_label]}")
        for pred_label, count in counts.items():
            print(
                f"  Predicted {LABEL_MAP[pred_label]}: {count}"
            )
        print()


def get_neutral_errors(test_df, n=5):
    """
    Return misclassified neutral examples.
    """

    neutral_df = test_df[test_df["label"] == 1]

    errors = neutral_df[neutral_df["bert_pred"] != 1]

    print(f"\nTotal Neutral Errors: {len(errors)}\n")

    return errors.sample(min(n, len(errors)), random_state=42)


def analyze_neutral_boundary_cases(test_df, n=5):
    """
    Examine cases where:
    - True Positive predicted Neutral
    - True Negative predicted Neutral
    """

    print("\nPositive -> Neutral:\n")
    pos_to_neutral = test_df[
        (test_df["label"] == 2) & (test_df["bert_pred"] == 1)
    ]

    print(f"Count: {len(pos_to_neutral)}\n")

    for _, row in pos_to_neutral.sample(min(n, len(pos_to_neutral)), random_state=42).iterrows():
        print("=" * 80)
        print("True: Positive | Predicted: Neutral")
        print(row["text"])
        print()

    print("\nNegative -> Neutral:\n")
    neg_to_neutral = test_df[
        (test_df["label"] == 0) & (test_df["bert_pred"] == 1)
    ]

    print(f"Count: {len(neg_to_neutral)}\n")

    for _, row in neg_to_neutral.sample(min(n, len(neg_to_neutral)), random_state=42).iterrows():
        print("=" * 80)
        print("True: Negative | Predicted: Neutral")
        print(row["text"])
        print()

def comparative_language_analysis(test_df):
    keywords = ["better than", "worse than", "compared to", "not as good"]

    test_df = test_df.copy()
    test_df["has_comparative"] = test_df["text"].str.lower().apply(
        lambda x: any(phrase in x for phrase in keywords)
    )

    misclassified = test_df[test_df["label"] != test_df["bert_pred"]]
    correct = test_df[test_df["label"] == test_df["bert_pred"]]

    mis_rate = misclassified["has_comparative"].mean()
    correct_rate = correct["has_comparative"].mean()

    print("\nComparative Language Rates:")
    print(f"Misclassified: {mis_rate:.3f}")
    print(f"Correctly classified: {correct_rate:.3f}")


def adjacent_error_share(test_df):
    """
    Share of errors between neighbouring classes (Neg<->Neu, Neu<->Pos)
    versus polarity flips (Neg<->Pos). A high adjacent share means the model
    mostly errs on the ordinal boundary rather than confusing polarity.
    """

    errors = test_df[test_df["label"] != test_df["bert_pred"]]
    flips = (errors["label"] - errors["bert_pred"]).abs() == 2

    print("\nError Structure:")
    print(f"Total errors: {len(errors)}")
    print(f"Polarity flips (Neg<->Pos): {flips.sum()} ({flips.mean():.1%})")
    print(f"Adjacent-class errors: {(~flips).sum()} ({(~flips).mean():.1%})")

    return 1 - flips.mean()


def bootstrap_macro_f1_ci(test_df, n_resamples=1000, seed=42):
    """
    95% bootstrap confidence interval for BERT macro F1 on the test set.
    """

    y_true = test_df["label"].values
    y_pred = test_df["bert_pred"].values
    rng = np.random.default_rng(seed)

    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro"))

    low, high = np.percentile(scores, [2.5, 97.5])
    print(f"\nMacro F1 95% bootstrap CI ({n_resamples} resamples): [{low:.4f}, {high:.4f}]")

    return low, high
