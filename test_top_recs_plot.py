from utils.evaluation import plot_top_recommendations

if __name__ == "__main__":
    plot_top_recommendations(user_id=1, model="svd", top_n=10)
    print("Top recommendations plot created.")
