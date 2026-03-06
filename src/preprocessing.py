# src/preprocessing.py

def get_text_and_labels(df):
    """
    Extract raw text and labels from a DataFrame.
    """
    X = df["text"].astype(str)
    y = df["label"]
    return X, y