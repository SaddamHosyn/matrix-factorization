from models.svd_model import train_svd

if __name__ == "__main__":
    pred_matrix, rmse = train_svd()
    print("SVD RMSE on test set:", rmse)
    print("Predicted matrix shape:", pred_matrix.shape)
