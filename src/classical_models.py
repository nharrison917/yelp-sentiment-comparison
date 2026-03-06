# src/classical_models.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from config import RANDOM_SEED


def train_classical_models(X_train, y_train):
    """
    Train TF-IDF + Naive Bayes and Logistic Regression models.
    Returns trained models and fitted vectorizer.
    """

    # Vectorizer (same for both models)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100_000,
        stop_words="english",
        max_df=0.95,
        min_df=50,
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    # Naive Bayes
    model_nb = MultinomialNB()
    model_nb.fit(X_train_vec, y_train)

    # Logistic Regression
    model_lr = LogisticRegression(
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    model_lr.fit(X_train_vec, y_train)

    return vectorizer, model_nb, model_lr


def evaluate_classical_models(vectorizer, model_nb, model_lr, X_test):
    """
    Generate predictions for both classical models.
    """

    X_test_vec = vectorizer.transform(X_test)

    y_pred_nb = model_nb.predict(X_test_vec)
    y_pred_lr = model_lr.predict(X_test_vec)

    return y_pred_nb, y_pred_lr