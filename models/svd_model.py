import os
import json
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from utils.matrix_creation import create_and_save_user_item_matrix


def train_svd(
    data_dir: str = "data",
    processed_dir: str = "processed",
    reports_dir: str = "reports",
    k: int = 50,
):
    """
    Train an SVD-based recommender on the MovieLens data.

    Steps:
    - Build normalized user-item matrix from train ratings
    - Apply truncated SVD (k latent factors)
    - Reconstruct full predicted rating matrix
    - Compute RMSE on test set
    - Save predictions and metrics
    """
    # Step 1: create train/test split and normalized matrix
    train_df, test_df, norm_matrix, user_means = create_and_save_user_item_matrix(
        data_dir=data_dir,
        processed_dir=processed_dir,
        test_size=0.2,
        random_state=42,
    )

    # Keep track of the mapping between user_id/movie_id and matrix indices
    user_ids = norm_matrix.index.values
    movie_ids = norm_matrix.columns.values

    # Fill NaNs with 0 for SVD (since we mean-centered, 0 ≈ user mean)
    R = norm_matrix.fillna(0).values

    # Step 2: truncated SVD
    # R ≈ U * S * Vt, with rank k
    U, sigma, Vt = svds(R, k=k)
    sigma = np.diag(sigma)

    # Step 3: reconstruct normalized ratings
    R_hat_norm = np.dot(np.dot(U, sigma), Vt)

    # Step 4: add user means back to get predictions on original rating scale
    # user_means is a Series indexed by user_id, need to align with matrix row order
    user_mean_array = user_means.loc[user_ids].values.reshape(-1, 1)
    R_hat = R_hat_norm + user_mean_array

    # Convert back to DataFrame with user_id/movie_id labels
    R_hat_df = pd.DataFrame(R_hat, index=user_ids, columns=movie_ids)

    # Step 5: compute RMSE on test set
    rmse = compute_rmse_on_test(test_df, R_hat_df)

    # Step 6: save predictions and metrics
    os.makedirs(reports_dir, exist_ok=True)

    # Save full predicted rating matrix
    np.save(os.path.join(reports_dir, "svd_predictions.npy"), R_hat)

    # Update model_metrics.json
    metrics_path = os.path.join(reports_dir, "model_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                metrics = {}

    metrics["SVD_RMSE"] = rmse

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return R_hat_df, rmse


def compute_rmse_on_test(test_df: pd.DataFrame, pred_matrix: pd.DataFrame) -> float:
    """
    Compute RMSE between actual ratings in test_df and predictions in pred_matrix.

    test_df: columns user_id, movie_id, rating
    pred_matrix: rows user_id, columns movie_id, values predicted rating
    """
    # Filter to only user/movie pairs present in the prediction matrix
    mask = test_df["user_id"].isin(pred_matrix.index) & test_df["movie_id"].isin(pred_matrix.columns)
    test_filtered = test_df[mask].copy()

    # Get predictions for each (user, movie) in test_filtered
    preds = []
    for row in test_filtered.itertuples(index=False):
        u = row.user_id
        m = row.movie_id
        preds.append(pred_matrix.at[u, m])

    preds = np.array(preds)
    actuals = test_filtered["rating"].values

    mse = np.mean((preds - actuals) ** 2)
    rmse = np.sqrt(mse)
    return float(rmse)
