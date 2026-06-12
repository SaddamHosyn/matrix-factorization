from models.pmf_model import train_pmf

if __name__ == "__main__":
    pmf, rmse = train_pmf(
        num_factors=75,
        learning_rate=0.005,
        reg=0.05,
        num_epochs=30,
    )
    print("PMF RMSE on test set:", rmse)
