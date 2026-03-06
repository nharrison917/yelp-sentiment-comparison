import pandas as pd

from interpretability import (
    analyze_confusion,
    analyze_neutral_boundary_cases,
    comparative_language_analysis,
)


if __name__ == "__main__":

    print("\nLoading saved predictions...\n")

    test_df = pd.read_csv("bert_test_predictions.csv")

    # Confusion structure
    analyze_confusion(test_df)

    # Neutral boundary analysis
    analyze_neutral_boundary_cases(test_df, n=3)

    # Comparative language rates
    comparative_language_analysis(test_df)