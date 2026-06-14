from fileinput import filename
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.svd_model import train_svd
from models.pmf_model import train_pmf

from utils.matrix_creation import train_test_split_ratings
from utils.recommendation import generate_recommendations


def plot_rmse_comparison(
    reports_dir: str = "reports",
):
    """
    Read model_metrics.json and create an RMSE comparison bar chart
    between SVD and PMF. Save to reports/rmse_comparison.png.
    """
    metrics_path = os.path.join(reports_dir, "model_metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    svd_rmse = metrics.get("SVD_RMSE", None)
    pmf_rmse = metrics.get("PMF_RMSE", None)

    models = []
    rmses = []

    if svd_rmse is not None:
        models.append("SVD")
        rmses.append(svd_rmse)

    if pmf_rmse is not None:
        models.append("PMF")
        rmses.append(pmf_rmse)

    plt.figure()
    plt.bar(models, rmses, color=["steelblue", "orange"])
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison: SVD vs PMF")
    for i, v in enumerate(rmses):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(os.path.join(reports_dir, "rmse_comparison.png"), bbox_inches="tight")
    plt.close()


def plot_predicted_vs_actual_svd(
    data_dir: str = "data",
    processed_dir: str = "processed",
    reports_dir: str = "reports",
    sample_size: int = 10000,
):
    """
    Plot predicted vs actual ratings for SVD model on a sample
    of test points. Save to reports/predicted_vs_actual.png.
    """
    # Load test set
    _, test_df = train_test_split_ratings(data_dir=data_dir)

    # Load SVD predictions matrix
    svd_pred_path = os.path.join(reports_dir, "svd_predictions.npy")
    R_hat = np.load(svd_pred_path)

    # Load user-item matrix to get user/movie mappings
    user_item_path = os.path.join(processed_dir, "user_item_matrix.csv")
    user_item = pd.read_csv(user_item_path, index_col=0)
    user_ids = user_item.index.values
    movie_ids = user_item.columns.values

    pred_df = pd.DataFrame(R_hat, index=user_ids, columns=movie_ids)

    # Filter test_df to user/movie pairs in prediction matrix
    mask = test_df["user_id"].isin(pred_df.index) & test_df["movie_id"].isin(
        pred_df.columns
    )
    test_filtered = test_df[mask].copy()

    # Optionally sample for plotting
    if len(test_filtered) > sample_size:
        test_filtered = test_filtered.sample(n=sample_size, random_state=42)

    # Build arrays of predicted and actual
    preds = []
    for row in test_filtered.itertuples(index=False):
        preds.append(pred_df.at[row.user_id, row.movie_id])

    preds = np.array(preds)
    actuals = test_filtered["rating"].values

    # Scatter plot
    plt.figure()
    plt.scatter(actuals, preds, alpha=0.3)
    plt.xlabel("Actual rating")
    plt.ylabel("Predicted rating")
    plt.title("SVD: Predicted vs Actual Ratings")
    plt.plot([1, 5], [1, 5], color="red", linestyle="--")  # y = x line

    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(
        os.path.join(reports_dir, "predicted_vs_actual.png"), bbox_inches="tight"
    )
    plt.close()


def plot_top_recommendations(
    user_id: int,
    model: str = "svd",
    top_n: int = 10,
    reports_dir: str = "reports",
    processed_dir: str = "processed",
):
    """
    Plot a bar chart of top-N recommended movies (predicted ratings) for a user.
    Save to reports/top_recommendations.png.
    """
    recs = generate_recommendations(
        user_id=user_id,
        model=model,
        top_n=top_n,
        reports_dir=reports_dir,
        processed_dir=processed_dir,
    )

    plt.figure(figsize=(12, 6))
    x = np.arange(len(recs))
    plt.bar(x, recs["predicted_rating"], color="skyblue")
    plt.xticks(x, recs["title"], rotation=90)
    plt.ylabel("Predicted rating")
    plt.title(f"Top-{top_n} {model.upper()} recommendations for user {user_id}")

    os.makedirs(reports_dir, exist_ok=True)
    plt.tight_layout()
    filename = f"user_{user_id}_top_recommendations_{model}.png"
    plt.savefig(os.path.join(reports_dir, filename), bbox_inches="tight")
    plt.close()


def plot_user_comparison(
    user_id: int,
    data_dir: str = "data",
    processed_dir: str = "processed",
    reports_dir: str = "reports",
    max_points: int = 20,
):
    """
    Compare SVD and PMF predictions against actual ratings for one user.
    Save plot to reports/user_comparison.png.
    """
    svd_pred_df, _ = train_svd(
        data_dir=data_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        k=50,
    )
    pmf_model, _ = train_pmf(
        data_dir=data_dir,
        reports_dir=reports_dir,
        num_factors=50,
        learning_rate=0.01,
        reg=0.1,
        num_epochs=10,
    )

    _, test_df = train_test_split_ratings(data_dir=data_dir)
    user_test = test_df[test_df["user_id"] == user_id].copy()

    if user_test.empty:
        raise ValueError(f"No test ratings found for user {user_id}")

    if len(user_test) > max_points:
        user_test = user_test.sample(n=max_points, random_state=42)

    movie_ids = user_test["movie_id"].values
    actuals = user_test["rating"].values

    svd_preds = []
    pmf_preds = []

    for movie_id in movie_ids:
        if user_id in svd_pred_df.index and movie_id in svd_pred_df.columns:
            svd_preds.append(svd_pred_df.at[user_id, movie_id])
        else:
            svd_preds.append(np.nan)

        pmf_pred = pmf_model.predict(user_id, movie_id)
        pmf_preds.append(pmf_pred if pmf_pred is not None else np.nan)

    comp_df = pd.DataFrame(
        {
            "movie_id": movie_ids,
            "actual": actuals,
            "svd": svd_preds,
            "pmf": pmf_preds,
        }
    )

    x = np.arange(len(comp_df))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, comp_df["actual"], width, label="Actual")
    plt.bar(x, comp_df["svd"], width, label="SVD")
    plt.bar(x + width, comp_df["pmf"], width, label="PMF")

    plt.xlabel("Movie ID")
    plt.ylabel("Rating")
    plt.title(f"SVD vs PMF vs Actual Ratings for User {user_id}")
    plt.xticks(x, comp_df["movie_id"], rotation=90)
    plt.legend()

    os.makedirs(reports_dir, exist_ok=True)
    plt.tight_layout()
    filename = f"user_{user_id}_comparison.png"
    plt.savefig(os.path.join(reports_dir, filename), bbox_inches="tight")
    plt.close()


def compute_baseline_rmse(data_dir: str = "data") -> float:
    """
    Compute RMSE of a simple baseline:
    predict the global mean rating for every user-movie pair.
    """
    train_df, test_df = train_test_split_ratings(data_dir=data_dir)

    # Global mean baseline
    global_mean = train_df["rating"].mean()

    preds = np.full_like(test_df["rating"].values, fill_value=global_mean, dtype=float)
    actuals = test_df["rating"].values

    mse = np.mean((preds - actuals) ** 2)
    rmse = float(np.sqrt(mse))
    return rmse
