# src/evaluation.py

from sklearn.metrics import classification_report, f1_score


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Print evaluation metrics and return macro F1 score.
    """

    print(f"\n{'='*40}")
    print(f"{model_name} Evaluation")
    print(f"{'='*40}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nMacro F1 Score: {macro_f1:.4f}")

    return macro_f1