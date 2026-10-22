import os
import sys

import pandas as pd

from interpretability import (
    analyze_confusion,
    analyze_neutral_boundary_cases,
    comparative_language_analysis,
    adjacent_error_share,
    bootstrap_macro_f1_ci,
)


OUTPUT_PATH = "outputs/analysis_output.txt"


class Tee:
    """Write to the terminal and a UTF-8 file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


if __name__ == "__main__":

    os.makedirs("outputs", exist_ok=True)

    # Explicit UTF-8 so review text (emoji, accents) doesn't fail on Windows cp1252
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        sys.stdout = Tee(sys.__stdout__, f)

        print("\nLoading saved predictions...\n")

        test_df = pd.read_csv("bert_test_predictions.csv")

        # Confusion structure
        analyze_confusion(test_df)

        # Neutral boundary analysis
        analyze_neutral_boundary_cases(test_df, n=3)

        # Comparative language rates
        comparative_language_analysis(test_df)

        # Ordinal error structure
        adjacent_error_share(test_df)

        # Uncertainty on headline metric
        bootstrap_macro_f1_ci(test_df)

        sys.stdout = sys.__stdout__

    print(f"\nSaved: {OUTPUT_PATH}")
