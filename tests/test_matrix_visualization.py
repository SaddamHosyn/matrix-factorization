import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_user_item_heatmap(
    processed_dir: str = "processed",
    reports_dir: str = "reports",
    max_users: int = 50,
    max_items: int = 50,
):
    # Load the user–item matrix
    user_item_path = os.path.join(processed_dir, "user_item_matrix.csv")
    user_item = pd.read_csv(user_item_path, index_col=0)

    # Take a small top-left block for visualization
    sub = user_item.iloc[:max_users, :max_items]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        sub.values,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    plt.colorbar(im, label="Rating (or normalized value)")
    plt.xlabel("Item index (subset)")
    plt.ylabel("User index (subset)")
    plt.title("User–Item Rating Matrix (subset)")

    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, "user_item_heatmap.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    plot_user_item_heatmap()
