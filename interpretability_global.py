import numpy as np
import pandas as pd

from utils.data_loader import load_movies
from models.pmf_model import train_pmf


def analyze_global_factor(
    data_dir: str = "data",
    reports_dir: str = "reports",
    factor_idx: int = 0,
    top_n: int = 10,
):
    # Train or load PMF
    pmf, _ = train_pmf(
        data_dir=data_dir,
        reports_dir=reports_dir,
    )

    movies = load_movies(data_dir)

    # V shape: (num_items, num_factors)
    factor_values = pmf.V[:, factor_idx]

    # Top items for this factor
    top_indices = np.argsort(-factor_values)[:top_n]

    # Map factor indices back to movie_ids
    movie_ids = [pmf.index_to_item_id[i] for i in top_indices]

    # Filter movies by those ids and keep same order
    top_movies = movies[movies["movie_id"].isin(movie_ids)].copy()
    top_movies = top_movies.set_index("movie_id").loc[movie_ids].reset_index()

    print(f"\nTop {top_n} movies for latent factor {factor_idx}:")
    print(top_movies[["movie_id", "title", "genres"]])


if __name__ == "__main__":
    # Example: inspect factor 0 and factor 1
    analyze_global_factor(factor_idx=0)
    analyze_global_factor(factor_idx=1)
