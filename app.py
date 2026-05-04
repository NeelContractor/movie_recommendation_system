import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import ast

# --- Load assets ---
@st.cache_resource
def load_assets():
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
    return cosine_sim

@st.cache_data
def load_data():
    movies = pd.read_csv('movies_clean.csv')
    movies['genres_list'] = movies['genres_list'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    top_movies = pd.read_csv('top_movies.csv')
    ratings_df = pd.read_csv('ratings_df.csv')
    predicted_df = pd.read_csv('predicted_ratings.csv', index_col=0)
    predicted_df.index = predicted_df.index.astype(int)
    predicted_df.columns = predicted_df.columns.astype(int)
    with open('movie_indices.json') as f:
        indices = json.load(f)
    return movies, top_movies, ratings_df, predicted_df, indices

cosine_sim = load_assets()
movies, top_movies, ratings_df, predicted_df, indices = load_data()
n_movies = len(top_movies)

# --- Recommendation functions ---
def recommend_content(title, n=10):
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    similarity    = [round(i[1], 3) for i in sim_scores]
    result = movies.iloc[movie_indices][
        ['title', 'genres_str', 'vote_average', 'director']].copy()
    result['similarity'] = similarity
    return result.reset_index(drop=True)

def recommend_collaborative(user_id, n=10):
    if user_id not in predicted_df.index:
        return pd.DataFrame()

    # Movies this user already rated
    rated = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    unrated = [mid for mid in predicted_df.columns if mid not in rated]

    preds = predicted_df.loc[user_id, unrated].sort_values(ascending=False)
    top_ids = preds.head(n).index.tolist()
    top_scores = preds.head(n).values.tolist()

    results = []
    for mid, score in zip(top_ids, top_scores):
        if int(mid) < len(top_movies):
            title = top_movies.iloc[int(mid)]['title']
            results.append({'title': title, 'predicted_rating': round(score, 2)})

    result = pd.DataFrame(results)
    if not result.empty:
        result = result.merge(
            movies[['title', 'genres_str', 'vote_average', 'director']],
            on='title', how='left')
    return result

def recommend_hybrid(title, user_id, n=10,
                     content_weight=0.6, collab_weight=0.4):
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_dict = {i: score for i, score in sim_scores}

    # Collaborative scores from precomputed matrix
    collab_preds = {}
    if user_id in predicted_df.index:
        for mid in predicted_df.columns:
            if int(mid) < len(top_movies):
                t = top_movies.iloc[int(mid)]['title']
                collab_preds[t] = predicted_df.loc[user_id, mid]

    hybrid_scores = []
    for i, row in movies.iterrows():
        t = row['title']
        content_score = sim_dict.get(i, 0)
        raw_collab = collab_preds.get(t, 5)
        collab_score = (raw_collab - predicted_df.values.min()) / \
                       (predicted_df.values.max() - predicted_df.values.min() + 1e-9)
        hybrid = content_weight * content_score + collab_weight * collab_score
        hybrid_scores.append((i, t, hybrid))

    hybrid_scores.sort(key=lambda x: x[2], reverse=True)
    hybrid_scores = [(i, t, s) for i, t, s in hybrid_scores if t != title][:n]

    result = pd.DataFrame(hybrid_scores, columns=['idx', 'title', 'hybrid_score'])
    result = result.merge(
        movies[['title', 'genres_str', 'vote_average', 'director']],
        on='title', how='left')
    result['hybrid_score'] = result['hybrid_score'].round(3)
    return result[['title', 'hybrid_score', 'genres_str', 'vote_average', 'director']]

# --- Page config ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")
st.markdown("Content-based, collaborative filtering, and hybrid recommendations.")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Content-Based",
    "👥 Collaborative Filtering",
    "🔀 Hybrid",
    "📚 How It Works"
])

all_titles = sorted(movies['title'].tolist())

# ---- TAB 1: Content-Based ----
with tab1:
    st.subheader("Find Movies Similar to One You Love")
    st.markdown("Recommends based on **genres, keywords, cast, and director**.")

    selected_movie = st.selectbox("Choose a movie:", all_titles,
                                   index=all_titles.index('Inception')
                                   if 'Inception' in all_titles else 0)
    n_recs = st.slider("Number of recommendations:", 5, 20, 10)

    if st.button("🎯 Get Recommendations",
                  use_container_width=True, type="primary", key="cb"):
        recs = recommend_content(selected_movie, n=n_recs)
        if recs is not None:
            movie_info = movies[movies['title'] == selected_movie].iloc[0]
            st.markdown(f"**Selected:** {selected_movie} | "
                        f"⭐ {movie_info['vote_average']} | "
                        f"🎬 {movie_info['director']} | "
                        f"🏷️ {movie_info['genres_str']}")
            st.divider()
            st.subheader(f"Top {n_recs} Similar Movies")
            for _, row in recs.iterrows():
                c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
                c1.markdown(f"**{row['title']}**")
                c2.markdown(f"⭐ {row['vote_average']}")
                c3.markdown(f"🔗 {row['similarity']}")
                c4.markdown(f"_{row['genres_str']}_")
        else:
            st.error("Movie not found in dataset.")

# ---- TAB 2: Collaborative ----
with tab2:
    st.subheader("Personalized Recommendations by User Taste")
    st.markdown("Recommends based on **what similar users liked**.")

    user_id = st.number_input("Enter User ID (1-500):",
                               min_value=1, max_value=500, value=42)
    n_collab = st.slider("Number of recommendations:", 5, 20, 10, key="collab_n")

    if st.button("👥 Get Recommendations",
                  use_container_width=True, type="primary", key="cf"):
        # Show what this user has rated
        user_rated = ratings_df[ratings_df['userId'] == user_id]\
            .sort_values('rating', ascending=False).head(5)
        st.markdown(f"**User {user_id}'s top rated movies:**")
        for _, r in user_rated.iterrows():
            mid = int(r['movieId'])
            title = top_movies.iloc[mid]['title'] if mid < len(top_movies) else 'Unknown'
            st.markdown(f"- {title}: ⭐ {r['rating']}")

        st.divider()
        recs = recommend_collaborative(user_id, n=n_collab)
        if not recs.empty:
            st.subheader(f"Top {n_collab} Predicted Movies for User {user_id}")
            for _, row in recs.iterrows():
                c1, c2, c3 = st.columns([3, 1, 3])
                c1.markdown(f"**{row['title']}**")
                c2.markdown(f"🎯 {row['predicted_rating']}")
                c3.markdown(f"_{row.get('genres_str', '')}_")
        else:
            st.warning("No recommendations found for this user.")

# ---- TAB 3: Hybrid ----
with tab3:
    st.subheader("Best of Both — Content + Collaborative")
    st.markdown("Combines **movie similarity** with **your personal taste**.")

    col1, col2 = st.columns(2)
    with col1:
        hybrid_movie = st.selectbox("Choose a seed movie:", all_titles,
                                     index=all_titles.index('Inception')
                                     if 'Inception' in all_titles else 0,
                                     key="hybrid_movie")
    with col2:
        hybrid_user = st.number_input("User ID (1-500):",
                                       min_value=1, max_value=500,
                                       value=42, key="hybrid_user")

    c_weight = st.slider("Content weight:", 0.0, 1.0, 0.6, 0.1)
    st.caption(f"Content: {c_weight:.0%} | Collaborative: {1-c_weight:.0%}")
    n_hybrid = st.slider("Recommendations:", 5, 20, 10, key="hybrid_n")

    if st.button("🔀 Get Hybrid Recommendations",
                  use_container_width=True, type="primary", key="hyb"):
        recs = recommend_hybrid(hybrid_movie, hybrid_user,
                                 n=n_hybrid,
                                 content_weight=c_weight,
                                 collab_weight=1 - c_weight)
        if recs is not None:
            movie_info = movies[movies['title'] == hybrid_movie].iloc[0]
            st.markdown(f"**Seed:** {hybrid_movie} | "
                        f"⭐ {movie_info['vote_average']} | "
                        f"🎬 {movie_info['director']}")
            st.divider()
            st.subheader("Hybrid Recommendations")
            for _, row in recs.iterrows():
                c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
                c1.markdown(f"**{row['title']}**")
                c2.markdown(f"⭐ {row['vote_average']}")
                c3.markdown(f"🔗 {row['hybrid_score']}")
                c4.markdown(f"_{row.get('genres_str', '')}_")
            st.info("💡 Try moving the content/collaborative slider "
                    "and see how recommendations change!")
        else:
            st.error("Movie not found.")

# ---- TAB 4: How It Works ----
with tab4:
    st.subheader("Three Approaches to Recommendation")
    st.markdown("""
    ### 1. Content-Based Filtering
    Each movie is converted to a **tag soup** combining genres, keywords,
    top 3 cast members, director (weighted 3x), and plot overview.
    Tags are vectorized with **TF-IDF** (5,000 features) and similarity
    is computed using **cosine similarity**.
    - Similarity of 1.0 = identical movies
    - Similarity of 0.0 = completely different

    ### 2. Collaborative Filtering (SVD)
    Builds a **500 user × 200 movie** ratings matrix.
    **TruncatedSVD** (50 latent factors) decomposes the sparse matrix to find
    hidden user preference patterns — like "action fan" or "art film lover" —
    and predicts ratings for movies the user hasn't seen.

    ### 3. Hybrid System
    ```
    hybrid_score = content_weight × content_similarity
                 + collab_weight  × collaborative_score
    ```
    Use the slider to shift weight between approaches and watch recommendations change.

    ### Cold Start Problem
    - **New user** (no ratings history) → collaborative fails → use content-based
    - **New movie** (no ratings yet) → collaborative fails → use content-based
    - Hybrid systems handle this gracefully

    ### Key Numbers
    - 4,809 movies (1916–2017)
    - TF-IDF matrix: 4,809 × 5,000 features
    - Cosine similarity matrix: 4,809 × 4,809 (176 MB, precomputed)
    - SVD: 50 latent factors, 500 users × 200 movies
    - Genre overlap: 9/10 average across test movies
    """)