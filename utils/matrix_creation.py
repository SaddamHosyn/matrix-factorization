import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.data_loader import load_ratings


def train_test_split_ratings(
    data_dir: str = "data",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Load ratings and split into train and test DataFrames.
    """
    ratings = load_ratings(data_dir)

    train_df, test_df = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state
    )

    return train_df, test_df


def build_user_item_matrix(ratings_df: pd.DataFrame):
    """
    Build a user-item matrix from a ratings DataFrame.

    Rows: user_id
    Columns: movie_id
    Values: rating
    """
    user_item = ratings_df.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    )

    return user_item


def normalize_user_item_matrix(user_item: pd.DataFrame):
    """
    Mean-center ratings per user (row-wise).

    For each user, subtract the user's mean rating from their rated movies.
    Missing values (NaN) are kept as NaN.
    """
    # Compute mean rating for each user over non-NaN entries
    user_means = user_item.mean(axis=1)

    # Subtract mean per row (broadcast row-wise)
    normalized = user_item.sub(user_means, axis=0)

    return normalized, user_means


def create_and_save_user_item_matrix(
    data_dir: str = "data",
    processed_dir: str = "processed",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    High-level function:
    - Split ratings into train/test
    - Build user-item matrix from train
    - Normalize
    - Save normalized matrix to processed/user_item_matrix.csv
    - Return train_df, test_df, normalized_matrix, user_means
    """
    train_df, test_df = train_test_split_ratings(
        data_dir=data_dir,
        test_size=test_size,
        random_state=random_state,
    )

    user_item = build_user_item_matrix(train_df)
    normalized, user_means = normalize_user_item_matrix(user_item)

    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "user_item_matrix.csv")
    normalized.to_csv(output_path)

    return train_df, test_df, normalized, user_means
