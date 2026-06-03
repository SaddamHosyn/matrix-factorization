from utils.data_loader import load_ratings, load_users, load_movies

if __name__ == "__main__":
    ratings = load_ratings("data")
    print(ratings.head())

    users = load_users("data")
    print(users.head())

    movies = load_movies("data")
    print(movies.head())
