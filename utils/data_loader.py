import os
import pandas as pd

def load_ratings(data_dir: str = "data"):
    """
    Load ratings.dat from the MovieLens 1M dataset.

    Parameters
    ----------
    data_dir : str
        Directory where ratings.dat is stored.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: user_id, movie_id, rating, timestamp.
    """
    ratings_path = os.path.join(data_dir, "ratings.dat")

    # MovieLens 1M ratings format: UserID::MovieID::Rating::Timestamp
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",  # needed for multi-character separator
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    # Drop any rows with missing values (defensive)
    df = df.dropna()

    # Enforce types
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(float)

    return df
def load_users(data_dir: str = "data"):
    """
    Load users.dat from the MovieLens 1M dataset.
    Returns columns: user_id, gender, age, occupation, zip_code.
    """
    users_path = os.path.join(data_dir, "users.dat")

    df = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"]
    )

    df = df.dropna()
    df["user_id"] = df["user_id"].astype(int)
    df["age"] = df["age"].astype(int)
    df["occupation"] = df["occupation"].astype(int)

    return df


def load_movies(data_dir: str = "data"):
    """
    Load movies.dat from the MovieLens 1M dataset.
    Returns columns: movie_id, title, genres.
    """
    movies_path = os.path.join(data_dir, "movies.dat")

    df = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="ISO-8859-1"  # handles special characters in titles
    )

    df = df.dropna()
    df["movie_id"] = df["movie_id"].astype(int)

    return df
