import os
import json
from utils.evaluation import compute_baseline_rmse


def update_baseline_metrics(reports_dir: str = "reports", data_dir: str = "data"):
    baseline_rmse = compute_baseline_rmse(data_dir=data_dir)

    metrics_path = os.path.join(reports_dir, "model_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                metrics = {}

    metrics["Baseline_RMSE"] = baseline_rmse

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Baseline RMSE: {baseline_rmse:.6f}")


if __name__ == "__main__":
    update_baseline_metrics()
