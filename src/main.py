# src/main.py

from config import RUN_TRAINING
import json
import os
import pickle

import pandas as pd

from data_loader import load_and_prepare_data
from preprocessing import get_text_and_labels
from classical_models import train_classical_models, evaluate_classical_models
from bert_model import train_bert, evaluate_bert
from evaluation import evaluate_model


if __name__ == "__main__":

    # =========================
    # DATA
    # =========================
    print("Loading and splitting data...")
    train_df, val_df, test_df = load_and_prepare_data()
    X_train, y_train = get_text_and_labels(train_df)
    X_test, y_test = get_text_and_labels(test_df)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    results = {}

    # =========================
    # CLASSICAL MODELS
    # =========================
    print("\nTraining TF-IDF + Naive Bayes / Logistic Regression...")
    vectorizer, model_nb, model_lr = train_classical_models(X_train, y_train)
    y_pred_nb, y_pred_lr = evaluate_classical_models(
        vectorizer, model_nb, model_lr, X_test
    )

    results["naive_bayes"] = evaluate_model(y_test, y_pred_nb, model_name="Naive Bayes")
    results["logistic_regression"] = evaluate_model(
        y_test, y_pred_lr, model_name="Logistic Regression"
    )

    os.makedirs("models", exist_ok=True)
    with open("models/classical_models.pkl", "wb") as f:
        pickle.dump(
            {"vectorizer": vectorizer, "naive_bayes": model_nb, "logistic_regression": model_lr},
            f,
        )
    print("Saved: models/classical_models.pkl")

    # =========================
    # BERT MODEL
    # =========================
    if RUN_TRAINING:
        print("\nStarting BERT training...\n")

        trainer = train_bert(train_df, val_df)

        y_pred_bert = evaluate_bert(trainer, test_df)

        test_df = test_df.copy()
        test_df["bert_pred"] = y_pred_bert
        test_df.to_csv("bert_test_predictions.csv", index=False)

    else:
        print("\nLoading saved BERT predictions...\n")
        saved_df = pd.read_csv("bert_test_predictions.csv")

        # The saved predictions must cover the same test rows, in the same order,
        # as the regenerated split -- otherwise the classical vs BERT comparison
        # would be scored on different data.
        assert len(saved_df) == len(test_df), "Saved predictions differ in length from test split"
        assert (saved_df["text"].values == test_df["text"].astype(str).values).all(), \
            "Saved predictions are not aligned with the regenerated test split"
        y_pred_bert = saved_df["bert_pred"].values

    results["bert_weighted"] = evaluate_model(y_test, y_pred_bert, model_name="BERT (Weighted)")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: outputs/results.json")
