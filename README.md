# Advanced Movie Recommender System

This project implements an advanced movie recommender system using **Singular Value Decomposition (SVD)** and **Probabilistic Matrix Factorization (PMF)** on the MovieLens 1M dataset.

The system generates personalized movie recommendations for users, compares model performance, and provides an interactive dashboard built with **Streamlit**.

## Features

- Load and preprocess the MovieLens 1M dataset
- Build a normalized user-item interaction matrix
- Train an SVD model using `scipy.sparse.linalg.svds`
- Train a PMF model using iterative gradient descent
- Evaluate both models using RMSE
- Generate top-N movie recommendations
- Visualize:
  - PMF convergence
  - RMSE comparison
  - Predicted vs actual ratings
  - User-level SVD vs PMF prediction comparison
  - Top recommended movies
- Interactive Streamlit dashboard with user ID input and model comparison

## Project Structure

```text
matrix-factorization-project/
│
├── data/
│   ├── ratings.dat
│   ├── users.dat
│   └── movies.dat
│
├── models/
│   ├── svd_model.py
│   ├── pmf_model.py
│
├── utils/
│   ├── data_loader.py
│   ├── matrix_creation.py
│   ├── recommendation.py
│   └── evaluation.py
│
├── processed/
│   └── user_item_matrix.csv
│
├── reports/
│   ├── model_metrics.json
│   ├── pmf_convergence.png
│   ├── rmse_comparison.png
│   ├── predicted_vs_actual.png
│   ├── user_comparison.png
│   ├── top_recommendations.png
│   └── user_<id>_recommendations.csv
│
├── app.py
├── requirement.txt
├── Movie_Recommender_System.ipynb
└── README.md
```

## Dataset

Download the MovieLens 1M dataset from the official GroupLens website:

- https://grouplens.org/datasets/movielens/1m/

Place these files in the `data/` folder:

- `ratings.dat`
- `users.dat`
- `movies.dat`

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirement.txt
```

## How to Run

### Run the Streamlit app

```bash
streamlit run app.py
```

### Run model tests manually

```bash
python test_load_ratings.py
python test_matrix_creation.py
python test_svd.py
python test_pmf.py
python test_recommendations.py
python test_top_recs_plot.py
```

## Models

### SVD

SVD factorizes the normalized user-item matrix into lower-dimensional latent factors using truncated singular value decomposition.

### PMF

PMF models user and movie latent vectors and learns them using gradient descent with regularization.

## Evaluation Metrics

The models are evaluated using:

- RMSE
- MSE during PMF convergence

Saved metrics can be found in:

- `reports/model_metrics.json`

## Streamlit Dashboard

The dashboard allows the user to:

- Enter a user ID
- View top-rated movies
- View SVD and PMF recommendations
- Compare model metrics
- See saved visualizations
- Tune model hyperparameters in the sidebar

## Future Improvements

- Better PMF tuning for lower RMSE
- Save/load trained PMF factors without retraining
- Filter out already-rated movies from recommendations
- Improve ranking metrics such as Precision@K and Recall@K
- Add movie search and surprise recommendation mode

## Learning Outcome

This project demonstrates how matrix factorization techniques can be applied in recommender systems and how different factorization models compare in both prediction quality and usability.
