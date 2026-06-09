import os
import numpy as np
import pandas as pd

from typing import Optional

from models.pmf_model import PMF  # make sure pmf_model.PMF is imported


def load_svd_predictions(
    reports_dir: str = "reports",
    processed_dir: str = "processed",
):
    svd_pred_path = os.path.join(reports_dir, "svd_predictions.npy")
    R_hat = np.load(svd_pred_path)

    user_item_path = os.path.join(processed_dir, "user_item_matrix.csv")
    user_item = pd.read_csv(user_item_path, index_col=0)
    user_ids = user_item.index.values
    movie_ids = user_item.columns.values

    movie_ids = movie_ids.astype(int)

    pred_df = pd.DataFrame(R_hat, index=user_ids, columns=movie_ids)
    pred_df.index = pred_df.index.astype(int)

    return pred_df


def generate_recommendations(
    user_id: int,
    model: str,
    top_n: int = 10,
    reports_dir: str = "reports",
    processed_dir: str = "processed",
    movies_path: str = "data/movies.dat",
    pmf_model: Optional[PMF] = None,
) -> pd.DataFrame:
    """
    Generate top-N movie recommendations for a given user.

    Parameters
    ----------
    user_id : int
        Target user ID.
    model : str
        'svd' or 'pmf'.
    pmf_model : PMF, optional
        Trained PMF instance (required if model='pmf').

    Returns
    -------
    DataFrame with: movie_id, title, genres, predicted_rating.
    """
    # Load movie metadata
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="ISO-8859-1",
    )
    movies["movie_id"] = movies["movie_id"].astype(int)

    if model.lower() == "svd":
        pred_df = load_svd_predictions(
            reports_dir=reports_dir, processed_dir=processed_dir
        )

        if user_id not in pred_df.index:
            raise ValueError(f"user_id {user_id} not found in SVD predictions")

        user_pred = pred_df.loc[user_id]
        sorted_movies = user_pred.sort_values(ascending=False)
        top_movie_ids = sorted_movies.index[:top_n].astype(int)
        top_ratings = sorted_movies.values[:top_n]

    elif model.lower() == "pmf":
        if pmf_model is None:
            raise ValueError("pmf_model must be provided when model='pmf'")

        # Only consider items known to PMF
        item_indices = list(pmf_model.item_id_to_index.keys())

        preds = []
        for m_id in item_indices:
            pred = pmf_model.predict(user_id, m_id)
            if pred is not None:
                preds.append((m_id, pred))

        if not preds:
            raise ValueError(f"No PMF predictions available for user_id {user_id}")

        preds_df = pd.DataFrame(preds, columns=["movie_id", "predicted_rating"])
        preds_df = preds_df.sort_values("predicted_rating", ascending=False).head(top_n)

        top_movie_ids = preds_df["movie_id"].astype(int).values
        top_ratings = preds_df["predicted_rating"].values

    else:
        raise ValueError("model must be 'svd' or 'pmf'")

    recs = pd.DataFrame({"movie_id": top_movie_ids, "predicted_rating": top_ratings})
    recs["movie_id"] = recs["movie_id"].astype(int)

    recs = recs.merge(movies, on="movie_id", how="left")
    recs = recs[["movie_id", "title", "genres", "predicted_rating"]]

    return recs
