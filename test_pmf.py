from models.pmf_model import train_pmf

if __name__ == "__main__":
    pmf, rmse = train_pmf(num_epochs=10)
    print("PMF RMSE on test set:", rmse)
