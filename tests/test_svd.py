import os
import json
import matplotlib.pyplot as plt
from models.svd_model import train_svd

if __name__ == "__main__":
    ks = [50, 100, 150]
    rmses = []

    best_rmse = None
    best_k = None

    for k in ks:
        print(f"\nTraining SVD with k = {k}")
        _, rmse = train_svd(k=k)
        print(f"SVD RMSE (k={k}): {rmse}")
        rmses.append(rmse)

        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_k = k

    # Overwrite metrics with the best k
    reports_dir = "reports"
    metrics_path = os.path.join(reports_dir, "model_metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    metrics["SVD_RMSE"] = best_rmse
    metrics["SVD_best_k"] = best_k
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot k vs RMSE
    os.makedirs(reports_dir, exist_ok=True)
    plt.figure()
    plt.plot(ks, rmses, marker="o")
    plt.xlabel("Number of SVD factors (k)")
    plt.ylabel("RMSE on test set")
    plt.title("SVD: k vs RMSE")
    plt.grid(True)
    plt.savefig(os.path.join(reports_dir, "svd_k_vs_rmse.png"), bbox_inches="tight")
    plt.close()
