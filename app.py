import os
import streamlit as st
import pandas as pd

from utils.data_loader import load_ratings, load_movies
from utils.recommendation import generate_recommendations
from models.svd_model import train_svd
from models.pmf_model import train_pmf
from utils.evaluation import plot_user_comparison

st.set_page_config(page_title="Movie Recommender System", layout="wide")


@st.cache_data
def get_data():
    ratings = load_ratings("data")
    movies = load_movies("data")
    return ratings, movies


@st.cache_resource
def get_models(k, num_factors, learning_rate, reg, num_epochs):
    svd_pred_df, svd_rmse = train_svd(
        data_dir="data",
        processed_dir="processed",
        reports_dir="reports",
        k=k,
    )

    pmf_model, pmf_rmse = train_pmf(
        data_dir="data",
        reports_dir="reports",
        num_factors=num_factors,
        learning_rate=learning_rate,
        reg=reg,
        num_epochs=num_epochs,
    )

    return svd_pred_df, svd_rmse, pmf_model, pmf_rmse


def get_user_history(user_id, ratings, movies, top_n=10):
    user_ratings = ratings[ratings["user_id"] == user_id].copy()
    user_ratings = user_ratings.sort_values(by="rating", ascending=False).head(top_n)
    user_ratings = user_ratings.merge(movies, on="movie_id", how="left")
    return user_ratings[["movie_id", "title", "genres", "rating"]]


def save_combined_recommendations(user_id, svd_recs, pmf_recs, reports_dir="reports"):
    os.makedirs(reports_dir, exist_ok=True)

    svd_tmp = svd_recs.rename(columns={"predicted_rating": "svd_predicted_rating"})
    pmf_tmp = pmf_recs.rename(columns={"predicted_rating": "pmf_predicted_rating"})

    combined = svd_tmp.merge(
        pmf_tmp[["movie_id", "pmf_predicted_rating"]], on="movie_id", how="outer"
    )

    output_path = os.path.join(reports_dir, f"user_{user_id}_recommendations.csv")
    combined.to_csv(output_path, index=False)
    return output_path


def main():
    st.title("Advanced Movie Recommender System")
    st.write("Compare personalized movie recommendations from SVD and PMF models.")

    try:
        ratings, movies = get_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    valid_user_ids = sorted(ratings["user_id"].unique().tolist())

    st.sidebar.header("User Input")
    user_id = st.sidebar.number_input(
        "Enter User ID",
        min_value=int(min(valid_user_ids)),
        max_value=int(max(valid_user_ids)),
        value=1,
        step=1,
    )

    st.sidebar.markdown("### Hyperparameters")
    k = st.sidebar.slider(
        "SVD latent factors (k)", min_value=10, max_value=200, value=50, step=10
    )

    num_factors = st.sidebar.slider(
        "PMF latent factors", min_value=10, max_value=200, value=75, step=5
    )

    learning_rate = st.sidebar.select_slider(
        "PMF learning rate",
        options=[0.001, 0.005, 0.01, 0.02],
        value=0.005,
    )

    reg = st.sidebar.select_slider(
        "PMF regularization",
        options=[0.01, 0.05, 0.1, 0.2],
        value=0.05,
    )

    num_epochs = st.sidebar.slider(
        "PMF epochs", min_value=5, max_value=50, value=30, step=5
    )

    if user_id not in valid_user_ids:
        st.error(f"User ID {user_id} does not exist in the dataset.")
        return

    try:
        svd_pred_df, svd_rmse, pmf_model, pmf_rmse = get_models(
            k, num_factors, learning_rate, reg, num_epochs
        )
    except Exception as e:
        st.error(f"Failed to load/train models: {e}")
        return

    st.sidebar.markdown("### Model Metrics")
    st.sidebar.write(f"SVD RMSE: {svd_rmse:.4f}")
    st.sidebar.write(f"PMF RMSE: {pmf_rmse:.4f}")

    tabs = st.tabs(["History", "SVD Recommendations", "PMF Recommendations", "Plots"])

    with tabs[0]:
        st.subheader(f"User {user_id} - Top Rated Movies")
        try:
            history_df = get_user_history(user_id, ratings, movies, top_n=10)
            st.dataframe(history_df, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load user history: {e}")

    with tabs[1]:
        st.subheader("SVD Recommendations")
        try:
            svd_recs = generate_recommendations(
                user_id=user_id,
                model="svd",
                top_n=10,
                reports_dir="reports",
                processed_dir="processed",
            )
            st.dataframe(svd_recs, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to generate SVD recommendations: {e}")
            svd_recs = pd.DataFrame()

    with tabs[2]:
        st.subheader("PMF Recommendations")
        try:
            pmf_recs = generate_recommendations(
                user_id=user_id,
                model="pmf",
                top_n=10,
                pmf_model=pmf_model,
                reports_dir="reports",
                processed_dir="processed",
            )
            st.dataframe(pmf_recs, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to generate PMF recommendations: {e}")
            pmf_recs = pd.DataFrame()

    with tabs[3]:
        st.subheader("Evaluation & Comparison Plots")

        try:
            plot_user_comparison(
                user_id=user_id,
                data_dir="data",
                processed_dir="processed",
                reports_dir="reports",
                max_points=20,
            )
        except Exception as e:
            st.warning(f"Could not generate dynamic user comparison plot: {e}")

        if os.path.exists("reports/rmse_comparison.png"):
            st.image("reports/rmse_comparison.png", caption="RMSE Comparison")
        else:
            st.warning("RMSE comparison plot not found.")

        if os.path.exists("reports/predicted_vs_actual.png"):
            st.image(
                "reports/predicted_vs_actual.png", caption="Predicted vs Actual Ratings"
            )
        else:
            st.warning("Predicted vs actual plot not found.")

        if os.path.exists("reports/user_comparison.png"):
            st.image(
                "reports/user_comparison.png",
                caption="Dynamic User-level SVD vs PMF Comparison",
            )
        else:
            st.warning("User comparison plot not found.")

        if os.path.exists("reports/top_recommendations.png"):
            st.image(
                "reports/top_recommendations.png", caption="Top Recommendations Plot"
            )
        else:
            st.warning("Top recommendations plot not found.")

    if not svd_recs.empty and not pmf_recs.empty:
        try:
            output_path = save_combined_recommendations(
                user_id=user_id,
                svd_recs=svd_recs,
                pmf_recs=pmf_recs,
                reports_dir="reports",
            )
            st.success(f"Combined recommendations saved to {output_path}")
        except Exception as e:
            st.error(f"Failed to save combined recommendations: {e}")


if __name__ == "__main__":
    main()
