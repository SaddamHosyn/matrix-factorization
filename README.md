# Advanced Movie Recommender System (Matrix Factorization)

This project implements an advanced movie recommender system using **Singular Value Decomposition (SVD)** and **Probabilistic Matrix Factorization (PMF)** on the MovieLens 1M dataset.  
It generates personalized movie recommendations, compares model performance, and exposes the results through an interactive **Streamlit** dashboard.

---

## Key Features

- Load and preprocess the **MovieLens 1M** dataset
- Build a normalized **user–item interaction matrix**
- Train:
  - **SVD** using `scipy.sparse.linalg.svds`
  - **PMF** via gradient descent with L2 regularization and bias terms
- Evaluate both models on a held‑out test set using **RMSE**
- Generate top‑N movie recommendations per user for both SVD and PMF
- Save and visualize:
  - `pmf_convergence.png`: PMF **training MSE vs epoch**
  - `rmse_comparison.png`: Baseline vs SVD vs PMF **RMSE bar chart**
  - `predicted_vs_actual.png`: **Predicted vs actual ratings** for SVD
  - `user_<id>_comparison.png`: **User‑level** SVD vs PMF vs actual ratings
  - `user_<id>_top_recommendations_<model>.png`: Top‑N recommendations bar chart
- **Streamlit dashboard**:
  - User ID input
  - SVD vs PMF recommendation tables
  - Hyperparameter controls (e.g., number of factors, learning rate, regularization, epochs)
  - Embedded evaluation plots

---

## Project Structure

```text
.
├── README.md
├── app.py
├── data/
│   ├── movies.dat
│   ├── ratings.dat
│   └── users.dat
│
├── models/
│   ├── pmf_model.py
│   └── svd_model.py
│
├── utils/
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── matrix_creation.py
│   └── recommendation.py
│
├── processed/
│   └── user_item_matrix.csv
│
├── reports/
│   ├── model_metrics.json
│   ├── pmf_convergence.png
│   ├── pmf_factors/
│   │   ├── U.npy
│   │   └── V.npy
│   ├── predicted_vs_actual.png
│   ├── rmse_comparison.png
│   ├── svd_k_vs_rmse.png
│   ├── svd_predictions.npy
│   ├── top_recommendations.png
│   ├── user_item_heatmap.png
│   ├── user_1_recommendations.csv
│   ├── user_1_svd_recommendations.csv
│   ├── user_1_pmf_recommendations.csv
│   ├── user_1_top_recommendations_svd.png
│   ├── user_1_comparison.png
│   ├── user_2_recommendations.csv
│   ├── user_2_top_recommendations_svd.png
│   ├── user_2_comparison.png
│   ├── user_500_top_recommendations_svd.png
│   ├── user_500_comparison.png
│   └── user_comparison.png
│
├── tests/
│   ├── test_baseline.py
│   ├── test_evaluation_plots.py
│   ├── test_load_ratings.py
│   ├── test_matrix_creation.py
│   ├── test_matrix_visualization.py
│   ├── test_pmf.py
│   ├── test_pmf_recs.py
│   ├── test_pmf_tuning.py
│   ├── test_recommendations.py
│   ├── test_rmse_comparison.py
│   ├── test_svd.py
│   ├── test_three_users_plots.py
│   ├── test_top_recs_plot.py
│   └── test_user_comparison.py
│
├── interpretability_global.py
├── interpretability_local.py
└── requirement.txt
```

---

## Dataset

The project uses the **MovieLens 1M** dataset from GroupLens.

- Download: <https://grouplens.org/datasets/movielens/1m/>

Place the files into the `data/` folder:

- `ratings.dat`
- `users.dat`
- `movies.dat`

The preprocessing code converts them into a clean `ratings` DataFrame, then builds a normalized user–item matrix stored at `processed/user_item_matrix.csv`.

---

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

You can sanity‑check core dependencies with:

```bash
python -c "import numpy, pandas, scipy, matplotlib, streamlit"
```

---

## How to Run

### 1. Run the tests / pipelines

From the project root:

```bash
# Train SVD (k = 50) and log RMSE + SVD metrics
python tests/test_svd.py

# Train PMF, log RMSE, plot convergence, and update metrics
python tests/test_pmf.py

# Generate top-N recommendations & user-level comparison plots for 3 users
python tests/test_three_users_plots.py

# Generate evaluation plots from metrics/predictions
python tests/test_rmse_comparison.py
python tests/test_predicted_vs_actual.py
python tests/test_matrix_visualization.py
```

These scripts:

- Update `reports/model_metrics.json` with:
  - `Baseline_RMSE`
  - `SVD_RMSE` (best k)
  - `PMF_RMSE`
  - `PMF_vs_SVD_improvement_%`
- Regenerate all plots in `reports/` using the **latest** models.

### 2. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard lets you:

- Enter a **user ID** (validated against the dataset; invalid IDs show a friendly error).
- See the user’s **rating history**.
- Compare **SVD** and **PMF** top‑N recommendations.
- Inspect key plots:
  - RMSE comparison
  - SVD predicted vs actual ratings
  - PMF training convergence
  - User‑level SVD vs PMF vs actual comparison
  - Top‑N bar charts per user

---

## Models and Evaluation

### SVD

- Uses `scipy.sparse.linalg.svds` on a **normalized** user–item matrix.
- Incorporates **user means** and **item bias**:
  - Center ratings by user mean.
  - Estimate item bias from residuals.
  - Factorize the residual matrix.
- Final prediction:
  \[
  \hat{r}\_{ui} = \mu_u + b_i + q_u^\top p_i
  \]
- Predictions are clipped to the rating range [1, 5].

After tuning, SVD achieves:

- **SVD_RMSE ≈ 0.9002** on the test set (k = 50).

The script `test_svd.py` also produces `svd_k_vs_rmse.png`, which shows how RMSE varies with k (e.g., 50, 100, 150).

### PMF

- Implements a **Probabilistic Matrix Factorization** model with:
  - User latent factors `U`
  - Item latent factors `V`
  - User and item bias terms
  - Global mean rating
- Trained with stochastic gradient descent and L2 regularization on all parameters.
- Tracks and plots training MSE per epoch in `pmf_convergence.png`.

Final tuned PMF model:

- **PMF_RMSE ≈ 0.8491** on the test set.
- Outperforms SVD by about **6.6%** relative improvement in RMSE.

### Metrics

All key metrics are stored in `reports/model_metrics.json`, e.g.:

```json
{
  "Baseline_RMSE": 1.1197,
  "SVD_RMSE": 0.9002,
  "PMF_RMSE": 0.8491,
  "PMF_vs_SVD_improvement_%": 6.64
}
```

The `rmse_comparison.png` bar chart summarizes these values visually.

---

## Interpretability: What the Latent Factors Capture

Matrix factorization learns latent “themes” that explain user behavior.

- **Latent factor 0** appears to capture **comedy‑oriented content**: the movies with the highest scores on this factor are almost all comedies (e.g., _Dumb & Dumber_, _What About Bob?_).
- **Latent factor 1** emphasizes **darker, more serious content** with action, horror, or drama elements (e.g., _The 13th Warrior_, _The Haunting_, _Phantoms_).

These interpretations come from inspecting the top‑scoring movies along each factor and seeing consistent genres and tones.

---

## User‑Level Interpretation (Examples)

Using the plots and recommendation CSVs for specific users:

- **User 1 (train user)**
  - Top rated movies are dominated by **dramas** (_One Flew Over the Cuckoo’s Nest_, _Awakenings_, _Rain Man_) and **animated children’s/musical films** (_Toy Story_, _Beauty and the Beast_, _A Bug’s Life_).
  - PMF recommends other high‑quality dramas (_The Shawshank Redemption_, _Schindler’s List_, _Gone with the Wind_, _Inherit the Wind_) and more classic animated films (_Bambi_, _The Lion King_).
  - This shows that the latent factors successfully capture user 1’s taste for emotional dramas and family‑oriented animation. The `user_1_comparison.png` plot confirms that PMF’s predicted ratings are close to the actual ratings for these styles.

- **User 2 (train user)**
  - Exhibits a different mix of genres (e.g., more action or sci‑fi).
  - The `user_2_comparison.png` plot highlights where SVD and PMF diverge; in some cases SVD may over‑ or under‑estimate certain genres, explaining less accurate recommendations for this user.

- **User 500 (test user)**
  - Not seen during training, so this tests generalization.
  - The `user_500_comparison.png` plot shows that PMF still tracks the user’s preferences reasonably well, though errors are larger for less common items; this is expected when the model must generalize to unseen patterns.

These qualitative analyses complement the RMSE numbers and help explain **why** particular recommendations are made.

---

## Future Work

- Deeper PMF hyperparameter search (e.g., Bayesian optimization)
- Persist and reload trained factor matrices to avoid retraining in the dashboard
- Filter out already‑rated movies from recommendation lists
- Add ranking metrics (Precision@K, Recall@K, NDCG)
- Incorporate movie metadata (genres, year) for hybrid recommendation
- Experiment with alternative models (NMF, neural collaborative filtering)

---

## Learning Outcomes

This project demonstrates end‑to‑end how to:

- Build a user–item matrix from raw rating data
- Implement and tune **SVD** and **PMF** for movie recommendation
- Evaluate models with **RMSE** and learning curves
- Interpret latent factors and user‑level predictions
- Expose recommender logic and visualizations in a **Streamlit** app
