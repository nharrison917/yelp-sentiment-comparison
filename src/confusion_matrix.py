# src/confusion_matrix.py
# Run from project root: python src/confusion_matrix.py
# Requires: bert_test_predictions.csv in project root
# Output: outputs/bert_confusion_matrix.html

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import os

LABELS = ["Negative", "Neutral", "Positive"]


def build_confusion_matrix_html(predictions_path, output_path):
    df = pd.read_csv(predictions_path)
    cm = confusion_matrix(df["label"], df["bert_pred"])

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / row_sums * 100

    # Build annotation text: count + row %
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"<b>{cm[i, j]:,}</b><br>({cm_pct[i, j]:.1f}%)",
                    showarrow=False,
                    font=dict(
                        size=13,
                        color="white" if cm_pct[i, j] > 50 else "black",
                    ),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_pct,
            x=LABELS,
            y=LABELS,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Row %"),
            zmin=0,
            zmax=100,
        )
    )

    fig.update_layout(
        title=dict(
            text="BERT (Weighted) - Confusion Matrix<br><sup>Counts and row-normalised percentages | Test set n=60,000</sup>",
            x=0.5,
        ),
        xaxis=dict(title="Predicted Label", side="bottom"),
        yaxis=dict(title="True Label", autorange="reversed"),
        annotations=annotations,
        width=600,
        height=540,
        font=dict(size=13),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    build_confusion_matrix_html(
        predictions_path="bert_test_predictions.csv",
        output_path="outputs/bert_confusion_matrix.html",
    )
