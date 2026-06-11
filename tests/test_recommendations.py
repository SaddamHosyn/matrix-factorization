from utils.recommendation import generate_recommendations

if __name__ == "__main__":
    user_id = 1  # try a user that exists in the data
    recs = generate_recommendations(user_id=user_id, model="svd", top_n=10)
    print(recs)
