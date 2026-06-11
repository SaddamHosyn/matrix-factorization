from models.svd_model import train_svd

if __name__ == "__main__":
    for k in [50, 100, 150]:
        print(f"\nTraining SVD with k = {k}")
        _, rmse = train_svd(k=k)
        print(f"SVD RMSE (k={k}): {rmse}")
