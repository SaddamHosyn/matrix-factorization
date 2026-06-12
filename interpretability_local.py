import pandas as pd

from utils.data_loader import load_ratings, load_movies
from models.pmf_model import train_pmf
from utils.recommendation import generate_recommendations


def analyze_user_local(
    user_id: int,
    data_dir: str = "data",
    reports_dir: str = "reports",
    processed_dir: str = "processed",
):
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)

    # User history: top-rated movies
    user_hist = (
        ratings[ratings["user_id"] == user_id]
        .merge(movies, on="movie_id")
        .sort_values("rating", ascending=False)
        .head(10)
    )

    print(f"\nUser {user_id} top-rated movies:")
    print(user_hist[["movie_id", "title", "genres", "rating"]])

    # Train or load PMF
    pmf, _ = train_pmf(
        data_dir=data_dir,
        reports_dir=reports_dir,
    )

    pmf_recs = generate_recommendations(
        user_id=user_id,
        model="pmf",
        top_n=10,
        pmf_model=pmf,
        reports_dir=reports_dir,
        processed_dir=processed_dir,
    )

    print(f"\nPMF recommendations for user {user_id}:")
    print(pmf_recs[["movie_id", "title", "genres", "predicted_rating"]])


if __name__ == "__main__":
    for uid in [1, 10, 50]:
        analyze_user_local(user_id=uid)
