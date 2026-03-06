# src/data_loader.py

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, TRAIN_SIZE, VAL_SIZE, TEST_SIZE


def load_yelp_dataset():
    """
    Load Yelp dataset from Hugging Face and return train/test splits as DataFrames.
    """

    dataset = load_dataset("mrcaelumn/yelp_restaurant_review_labelled")

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    return train_df, test_df


def stratified_downsample(train_df, test_df):
    """
    Perform stratified sampling and create train/val/test splits.
    """

    # First sample from train set
    train_sampled, _ = train_test_split(
        train_df,
        train_size=TRAIN_SIZE + VAL_SIZE,
        stratify=train_df["label"],
        random_state=RANDOM_SEED,
    )

    # Sample from test set
    test_sampled, _ = train_test_split(
        test_df,
        train_size=TEST_SIZE,
        stratify=test_df["label"],
        random_state=RANDOM_SEED,
    )

    # Split train into train + validation
    train_final, val_final = train_test_split(
        train_sampled,
        test_size=VAL_SIZE,
        stratify=train_sampled["label"],
        random_state=RANDOM_SEED,
    )

    return train_final, val_final, test_sampled


def load_and_prepare_data():
    """
    Full pipeline: load + downsample + split.
    """

    train_df, test_df = load_yelp_dataset()

    train_df, val_df, test_df = stratified_downsample(train_df, test_df)

    return train_df, val_df, test_df