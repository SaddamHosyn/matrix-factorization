import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.matrix_creation import train_test_split_ratings


class PMF:
    def __init__(
        self,
        num_users,
        num_items,
        num_factors=50,
        learning_rate=0.01,
        reg=0.1,
        num_epochs=50,
        seed=42,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.num_epochs = num_epochs
        self.random_state = np.random.RandomState(seed)

        # Initialize user and item latent factors
        self.U = 0.01 * self.random_state.randn(num_users, num_factors)
        self.V = 0.01 * self.random_state.randn(num_items, num_factors)

        self.train_mse_history = []

    def fit(self, train_df: pd.DataFrame):
        """
        Train PMF using stochastic gradient descent over observed ratings.

        train_df must have columns: user_id, movie_id, rating.
        """
        # Map raw IDs to contiguous indices [0, num_users) and [0, num_items)
        user_ids = train_df["user_id"].unique()
        item_ids = train_df["movie_id"].unique()

        self.user_id_to_index = {u: idx for idx, u in enumerate(user_ids)}
        self.item_id_to_index = {i: idx for idx, i in enumerate(item_ids)}

        # Store reverse mappings for later predictions if needed
        self.index_to_user_id = {idx: u for u, idx in self.user_id_to_index.items()}
        self.index_to_item_id = {idx: i for i, idx in self.item_id_to_index.items()}

        # Convert train_df to arrays of indices for fast iteration
        user_idx = train_df["user_id"].map(self.user_id_to_index).values
        item_idx = train_df["movie_id"].map(self.item_id_to_index).values
        ratings = train_df["rating"].values

        n_ratings = len(ratings)

        for epoch in range(self.num_epochs):
            # Shuffle indices
            perm = self.random_state.permutation(n_ratings)

            for idx in perm:
                u = user_idx[idx]
                i = item_idx[idx]
                r_ui = ratings[idx]

                # Prediction: U_u^T V_i
                pred = np.dot(self.U[u], self.V[i])

                # Error
                err = r_ui - pred

                # Gradients with L2 regularization
                grad_Uu = -2 * err * self.V[i] + 2 * self.reg * self.U[u]
                grad_Vi = -2 * err * self.U[u] + 2 * self.reg * self.V[i]

                # Update
                self.U[u] -= self.learning_rate * grad_Uu
                self.V[i] -= self.learning_rate * grad_Vi

            # Compute train MSE at end of epoch
            mse = self._compute_mse(user_idx, item_idx, ratings)
            self.train_mse_history.append(mse)
            print(f"Epoch {epoch+1}/{self.num_epochs} - MSE: {mse:.4f}")

    def _compute_mse(self, user_idx, item_idx, ratings):
        preds = np.sum(self.U[user_idx] * self.V[item_idx], axis=1)
        mse = np.mean((ratings - preds) ** 2)
        return mse

    def predict(self, user_id, item_id):
        """
        Predict rating for a given user_id and item_id.
        """
        if (user_id not in self.user_id_to_index) or (
            item_id not in self.item_id_to_index
        ):
            return None

        u = self.user_id_to_index[user_id]
        i = self.item_id_to_index[item_id]
        return float(np.dot(self.U[u], self.V[i]))


def train_pmf(
    data_dir: str = "data",
    reports_dir: str = "reports",
    num_factors: int = 75,
    learning_rate: float = 0.005,
    reg: float = 0.05,
    num_epochs: int = 30,
):
    # Get train/test ratings
    train_df, test_df = train_test_split_ratings(data_dir=data_dir)

    num_users = train_df["user_id"].nunique()
    num_items = train_df["movie_id"].nunique()

    pmf = PMF(
        num_users=num_users,
        num_items=num_items,
        num_factors=num_factors,
        learning_rate=learning_rate,
        reg=reg,
        num_epochs=num_epochs,
    )

    pmf.fit(train_df)

    # Save latent factors
    factors_dir = os.path.join(reports_dir, "pmf_factors")
    os.makedirs(factors_dir, exist_ok=True)
    np.save(os.path.join(factors_dir, "U.npy"), pmf.U)
    np.save(os.path.join(factors_dir, "V.npy"), pmf.V)

    # Convergence plot (MSE vs iteration)
    os.makedirs(reports_dir, exist_ok=True)
    plt.figure()
    plt.plot(
        range(1, len(pmf.train_mse_history) + 1), pmf.train_mse_history, marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.title("PMF convergence")
    plt.grid(True)
    plt.savefig(os.path.join(reports_dir, "pmf_convergence.png"), bbox_inches="tight")
    plt.close()

    # Compute RMSE on test set
    pmf_rmse = compute_pmf_rmse_on_test(pmf, test_df)

    # Update model_metrics.json
    metrics_path = os.path.join(reports_dir, "model_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                metrics = {}

    # Store PMF RMSE
    metrics["PMF_RMSE"] = pmf_rmse

    # Compute improvement percentage if SVD_RMSE is available
    svd_rmse = metrics.get("SVD_RMSE", None)
    if svd_rmse is not None and svd_rmse > 0:
        improvement = (svd_rmse - pmf_rmse) / svd_rmse * 100.0
        metrics["PMF_vs_SVD_improvement_%"] = improvement

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return pmf, pmf_rmse


def compute_pmf_rmse_on_test(pmf: PMF, test_df: pd.DataFrame) -> float:
    # Filter to users/items known to PMF
    mask = test_df["user_id"].isin(pmf.user_id_to_index.keys()) & test_df[
        "movie_id"
    ].isin(pmf.item_id_to_index.keys())
    test_filtered = test_df[mask].copy()

    preds = []
    for row in test_filtered.itertuples(index=False):
        pred = pmf.predict(row.user_id, row.movie_id)
        preds.append(pred)

    preds = np.array(preds)
    actuals = test_filtered["rating"].values

    mse = np.mean((preds - actuals) ** 2)
    rmse = np.sqrt(mse)
    return float(rmse)
