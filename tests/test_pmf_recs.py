from models.pmf_model import train_pmf
from utils.recommendation import generate_recommendations

if __name__ == "__main__":
    pmf, rmse = train_pmf(num_epochs=5)
    print("PMF RMSE:", rmse)

    user_id = 1
    recs = generate_recommendations(
        user_id=user_id,
        model="pmf",
        top_n=10,
        pmf_model=pmf
    )
    print(recs)
