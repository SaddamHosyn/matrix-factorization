from utils.evaluation import plot_top_recommendations, plot_user_comparison

if __name__ == "__main__":
    # Choose 2 train users and 1 test user.
    # You may need to tweak these IDs based on your data.
    train_users = [1, 2]  # example train user IDs
    test_users = [500]  # example test user ID

    for uid in train_users + test_users:
        # Top-N recommendations (e.g., for SVD; you can also try model="pmf")
        plot_top_recommendations(
            user_id=uid,
            model="svd",
            top_n=10,
            reports_dir="reports",
            processed_dir="processed",
        )

        # Comparison of Actual vs SVD vs PMF for that user
        plot_user_comparison(
            user_id=uid,
            data_dir="data",
            processed_dir="processed",
            reports_dir="reports",
            max_points=20,
        )

        print(f"Generated recommendations and comparison plots for user {uid}")
