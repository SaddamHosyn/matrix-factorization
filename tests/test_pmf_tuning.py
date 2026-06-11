from models.pmf_model import train_pmf

if __name__ == "__main__":
    settings = [
        {"num_factors": 50, "learning_rate": 0.005, "reg": 0.1, "num_epochs": 30},
        {"num_factors": 50, "learning_rate": 0.005, "reg": 0.05, "num_epochs": 30},
        {"num_factors": 75, "learning_rate": 0.005, "reg": 0.05, "num_epochs": 30},
    ]

    for s in settings:
        print(
            f"\nPMF with factors={s['num_factors']}, "
            f"lr={s['learning_rate']}, reg={s['reg']}, epochs={s['num_epochs']}"
        )
        _, rmse = train_pmf(
            num_factors=s["num_factors"],
            learning_rate=s["learning_rate"],
            reg=s["reg"],
            num_epochs=s["num_epochs"],
        )
        print(f"PMF RMSE: {rmse:.6f}")
