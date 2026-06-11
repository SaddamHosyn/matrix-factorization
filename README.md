# Advanced Movie Recommender System

This project implements an advanced movie recommender system using **Singular Value Decomposition (SVD)** and **Probabilistic Matrix Factorization (PMF)** on the MovieLens 1M dataset.  
The system generates personalized movie recommendations, compares model performance, and exposes everything through an interactive **Streamlit** dashboard.

---

## Features

- Load and preprocess the MovieLens 1M dataset
- Build a normalized user–item interaction matrix
- Train an SVD model using `scipy.sparse.linalg.svds`
- Train a PMF model using gradient descent with regularization
- Evaluate both models using **RMSE**
- Generate top‑N movie recommendations per user
- Visualize:
  - PMF convergence (MSE vs epoch)
  - RMSE comparison between SVD and PMF
  - Predicted vs actual ratings for SVD
  - User‑level SVD vs PMF prediction comparison
  - Top recommended movies for a selected user
- Interactive Streamlit dashboard:
  - User ID input
  - SVD vs PMF recommendation tables
  - Hyperparameter sliders (e.g., number of factors, learning rate, regularization, epochs)

---

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
│   └── pmf_model.py
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
│   ├── user_<id>_svd_recommendations.csv
│   └── user_<id>_pmf_recommendations.csv
│
├── app.py
├── requirement.txt
├── Movie_Recommender_System.ipynb
└── README.md
```

---

## Dataset

The project uses the **MovieLens 1M** dataset from GroupLens.

Download from:

- https://grouplens.org/datasets/movielens/1m/

Place these files into the `data/` folder:

- `ratings.dat`
- `users.dat`
- `movies.dat`

---

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirement.txt
```

---

## How to Run

### Streamlit app

Run the main dashboard:

```bash
streamlit run app.py
```

This will:

- Load the processed user–item matrix
- Train or load SVD and PMF models
- Open a web UI where you can:
  - Select a user ID
  - See the user’s rating history
  - Compare SVD and PMF recommendations
  - Inspect evaluation plots

---

## Testing

A set of simple test scripts is provided to verify each step of the pipeline:

```bash
python test_load_ratings.py          # checks data loading
python test_matrix_creation.py       # checks user–item matrix creation
python test_svd.py                   # trains SVD and computes RMSE
python test_pmf.py                   # trains PMF and computes RMSE
python test_recommendations.py       # checks top‑N recommendation logic
python test_top_recs_plot.py         # generates top_recommendations.png
python test_svd_tuning.py            # (optional) explores SVD k values
python test_pmf_tuning.py            # (optional) explores PMF hyperparameters
```

These scripts help ensure that:

- Metrics in `reports/model_metrics.json` are up to date
- Plots in `reports/` reflect the latest models
- Recommendation logic runs end‑to‑end without errors

---

## Models

### Singular Value Decomposition (SVD)

SVD factorizes the normalized user–item rating matrix into:

- A user latent factor matrix
- A diagonal matrix of singular values
- An item latent factor matrix

Using a truncated SVD (rank \(k\)), the model reconstructs an approximation of the rating matrix and uses it to predict unseen ratings.

### Probabilistic Matrix Factorization (PMF)

PMF models:

- A latent vector for each user
- A latent vector for each item

The rating \(r\_{ui}\) is modeled as an inner product \(U_u \cdot V_i\), learned via gradient descent with L2 regularization. Hyperparameters (number of factors, learning rate, regularization, epochs) are tuned to reduce RMSE on a held‑out test set.

---

## Evaluation Metrics

The main evaluation metric is **Root Mean Squared Error (RMSE)** on a test split.  
Additional signals:

- PMF training MSE per epoch (used to monitor convergence)

Metrics are stored in:

- `reports/model_metrics.json`, e.g.:

```json
{
  "SVD_RMSE": ...,
  "PMF_RMSE": ...,
  "PMF_vs_SVD_improvement_%": ...
}
```

The improvement percentage quantifies how much PMF reduces RMSE relative to SVD.

---

## Streamlit Dashboard

The dashboard includes:

- **Sidebar controls**:
  - User ID selector
  - SVD latent factor slider (k)
  - PMF hyperparameter controls (latent factors, learning rate, regularization, epochs)
- **Tabs**:
  - History: user’s top‑rated movies
  - SVD Recommendations: SVD top‑N recommendations
  - PMF Recommendations: PMF top‑N recommendations
  - Plots: RMSE comparison, predicted vs actual, PMF convergence, user-level comparison, top‑N bar charts

Recommendation tables and plots update according to the selected user and current hyperparameters.

---

## Future Improvements

- Further PMF hyperparameter tuning for even lower RMSE
- Save and reload trained PMF factors to avoid retraining in the app
- Filter out already-rated movies from recommendation lists
- Add ranking metrics (Precision@K, Recall@K, NDCG)
- Integrate movie metadata search and a “surprise me” recommendation mode
- Experiment with additional models (e.g., NMF, neural collaborative filtering)

---

## Learning Outcome

This project shows how matrix factorization methods can be applied in recommender systems, how to evaluate them with RMSE, and how to compare different models (SVD vs PMF) in both offline metrics and an interactive UI.
