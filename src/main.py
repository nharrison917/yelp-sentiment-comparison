# src/main.py

from config import RUN_TRAINING
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
    train_df, val_df, test_df = load_and_prepare_data()
    X_train, y_train = get_text_and_labels(train_df)
    X_test, y_test = get_text_and_labels(test_df)


    # =========================
    # CLASSICAL MODELS
    # =========================
    vectorizer, model_nb, model_lr = train_classical_models(X_train, y_train)
    y_pred_nb, y_pred_lr = evaluate_classical_models(
        vectorizer, model_nb, model_lr, X_test
    )

    evaluate_model(y_test, y_pred_nb, model_name="Naive Bayes")
    evaluate_model(y_test, y_pred_lr, model_name="Logistic Regression")

    # =========================
    # BERT MODEL
    # =========================
    if RUN_TRAINING:
        print("\nStarting BERT training...\n")

        trainer = train_bert(train_df, val_df)

        y_pred_bert = evaluate_bert(trainer, test_df)

        test_df = test_df.copy()
        test_df["bert_pred"] = y_pred_bert

        trainer.save_model("./models/bert_weighted")
        test_df.to_csv("bert_test_predictions.csv", index=False)

    else:
        print("\nLoading saved BERT predictions...\n")
        test_df = pd.read_csv("bert_test_predictions.csv")
        y_pred_bert = test_df["bert_pred"].values

    evaluate_model(y_test, y_pred_bert, model_name="BERT (Weighted)")



