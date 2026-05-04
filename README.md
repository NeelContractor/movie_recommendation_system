# 🎬 Movie Recommendation System

A hybrid movie recommendation system combining content-based filtering and collaborative filtering, built on the TMDB 5000 dataset and deployed as an interactive Streamlit web app.

---

## 📊 System Performance

| Approach      | Method                     | Key Metric              |
| ------------- | -------------------------- | ----------------------- |
| Content-Based | TF-IDF + Cosine Similarity | Genre overlap: 9/10 avg |
| Collaborative | TruncatedSVD (50 factors)  | RMSE: 1.22 (1-10 scale) |
| Hybrid        | Weighted combination       | Best of both approaches |

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy, scikit-learn, scipy
- **Scikit-learn** — TF-IDF, Cosine Similarity, TruncatedSVD
- **Scikit-surprise** — SVD cross-validation (notebook only)
- **Streamlit** — Interactive web app with 3 recommendation modes

---

## 📁 Project Structure

```
movie_recommendation_system/
├── data/tmdb_5000_movies.csv      # Raw movie metadata
├── data/tmdb_5000_credits.csv     # Cast and crew data
├── main.ipynb            # EDA + modeling walkthrough
├── app.py                    # Streamlit web app
├── cosine_sim.pkl            # Precomputed similarity matrix (176MB)
├── movies_clean.csv          # Cleaned movie dataset
├── movie_indices.json        # Title → index mapping
├── top_movies.csv            # Top 200 movies for collaborative filtering
├── ratings_df.csv            # Simulated user ratings
├── predicted_ratings.csv     # Precomputed SVD predictions
├── requirements.txt          # Dependencies
└── README.md
```

---

## 🚀 Getting Started

**1. Clone the repo**

```bash
git clone https://github.com/NeelContractor/movie_recommendation_system.git
cd movie_recommendation_system
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Get both files from [Kaggle — TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata):

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

**4. Run the notebook**

Open `main.ipynb` in Jupyter and run all cells to generate the required `.pkl`, `.csv`, and `.json` files.

**5. Launch the web app**

```bash
streamlit run app.py
```

---

## 🔍 Approach

### Content-Based Filtering

Each movie is converted to a **tag soup** combining:

- Genres + top 5 keywords
- Top 3 cast members
- Director (weighted 3x — strongest signal)
- Plot overview

Tags are vectorized with **TF-IDF** (5,000 features) and similarity is measured using **cosine similarity** across all 4,809 × 4,809 movie pairs.

### Collaborative Filtering

Simulates 500 users rating 200 movies. Uses **TruncatedSVD** (50 latent factors) to decompose the sparse user-movie matrix and predict ratings for unseen movies. Latent factors capture hidden preferences like "action fan" or "art film lover."

### Hybrid System

```
hybrid_score = content_weight × content_similarity
             + collab_weight  × collaborative_score
```

Default: 60% content + 40% collaborative. Adjustable via slider in the app.

---

## 💡 Key Findings

- 🎬 **Director is the strongest content signal** — weighting director 3x causes Nolan films to cluster perfectly (Dark Knight → 6 Nolan recommendations in top 6)
- 🔗 **Genre overlap: 9/10** average across test movies — production-quality content filtering
- 👥 **Different users get different results** — collaborative filtering personalizes recommendations based on taste profile
- ❄️ **Cold start problem**: new users or new movies have no ratings history → content-based handles these cases gracefully
- 🔀 **Hybrid weight slider** — shifting from content to collaborative visibly changes recommendations in real time

---

## ⚠️ Notes

- `cosine_sim.pkl` is ~176MB — add to `.gitignore` before pushing to GitHub
- Collaborative filtering uses simulated ratings (TMDB has no individual user ratings)
- `scikit-surprise` is used in the notebook for cross-validation only — the app uses sklearn SVD

---

## 📌 Dataset

[Kaggle — TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
4,809 movies with metadata including genres, keywords, cast, crew, and ratings (1916–2017).
