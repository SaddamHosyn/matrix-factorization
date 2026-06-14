from utils.evaluation import plot_rmse_comparison

if __name__ == "__main__":
    plot_rmse_comparison(reports_dir="reports")
    print("Saved rmse_comparison.png")
