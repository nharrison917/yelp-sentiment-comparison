# src/evaluation.py

from sklearn.metrics import classification_report, f1_score, accuracy_score


LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Print evaluation metrics and return them as a dict
    (macro F1, accuracy, and per-class precision/recall/F1).
    """

    print(f"\n{'='*40}")
    print(f"{model_name} Evaluation")
    print(f"{'='*40}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nMacro F1 Score: {macro_f1:.4f}")

    report = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, output_dict=True
    )

    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
            }
            for name in LABEL_NAMES
        },
    }
