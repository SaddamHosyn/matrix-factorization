from utils.evaluation import plot_user_comparison  # adjust module path if needed

if __name__ == "__main__":
    user_id = 1  # try 1 first; change if no ratings in test set
    plot_user_comparison(
        user_id=user_id,
        data_dir="data",
        processed_dir="processed",
        reports_dir="reports",
        max_points=20,
    )
    print(f"user_comparison.png generated for user {user_id}")
