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
    - Estimate item bias from train ratings
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

    # Keep track of mapping between user_id/movie_id and matrix indices
    user_ids = norm_matrix.index.values
    movie_ids = norm_matrix.columns.values

    # Step 2: estimate item bias from train data after user-mean centering
    train_with_user_mean = train_df.copy()
    train_with_user_mean["user_mean"] = train_with_user_mean["user_id"].map(user_means)
    train_with_user_mean["residual"] = (
        train_with_user_mean["rating"] - train_with_user_mean["user_mean"]
    )

    item_bias = train_with_user_mean.groupby("movie_id")["residual"].mean()

    # Align item bias to matrix column order, fill unseen bias with 0
    item_bias_array = (
        pd.Series(movie_ids, index=movie_ids)
        .map(item_bias)
        .fillna(0)
        .values.reshape(1, -1)
    )

    # Step 3: subtract item bias from normalized matrix
    norm_matrix_adjusted = norm_matrix.subtract(item_bias_array.flatten(), axis=1)

    # Fill NaNs with 0 for SVD
    R = norm_matrix_adjusted.fillna(0).values

    # Step 4: truncated SVD
    U, sigma, Vt = svds(R, k=k)
    sigma = np.diag(sigma)

    # Step 5: reconstruct adjusted normalized ratings
    R_hat_adjusted = np.dot(np.dot(U, sigma), Vt)

    # Step 6: add user means and item bias back
    user_mean_array = user_means.loc[user_ids].values.reshape(-1, 1)
    R_hat = R_hat_adjusted + user_mean_array + item_bias_array

    # Convert back to DataFrame
    R_hat_df = pd.DataFrame(R_hat, index=user_ids, columns=movie_ids)

    # Step 7: compute RMSE on test set
    rmse = compute_rmse_on_test(test_df, R_hat_df)

    # Step 8: save predictions and metrics
    os.makedirs(reports_dir, exist_ok=True)

    np.save(os.path.join(reports_dir, "svd_predictions.npy"), R_hat)

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
    mask = test_df["user_id"].isin(pred_matrix.index) & test_df["movie_id"].isin(
        pred_matrix.columns
    )
    test_filtered = test_df[mask].copy()

    preds = []
    for row in test_filtered.itertuples(index=False):
        u = row.user_id
        m = row.movie_id
        pred = pred_matrix.at[u, m]

        # Clip predictions to valid MovieLens range
        pred = np.clip(pred, 1.0, 5.0)
        preds.append(pred)

    preds = np.array(preds)
    actuals = test_filtered["rating"].values

    mse = np.mean((preds - actuals) ** 2)
    rmse = np.sqrt(mse)
    return float(rmse)
