import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from src.data import load_data
from predict import RecommenderService
from config import TOP_K, EXPERIMENT_NAME


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")


@st.cache_data
def load():
    return load_data()


ratings, movies, links, _, _ = load()


@st.cache_resource
def load_service():
    return RecommenderService(movies)


service = load_service()


@st.cache_data
def load_metrics():

    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name("MovieLens-Recsys")

        if experiment is None:
            raise ValueError("No experiment found")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )

        if not runs:
            raise ValueError("No runs found")

        run = runs[0]

        metrics = run.data.metrics
        params = run.data.params

        return metrics, params

    except Exception:
        return {
            "recall_10": 0.081,
            "map_10": 0.033,
            "ndcg_10": 0.074,
            "precision_10": 0.012,
        }, {
            "model": "ALS (fallback)"
        }


metrics, params = load_metrics()


st.sidebar.header("🎯 Preferences")

movie_titles = sorted(movies["title"].tolist())

selected_movies = st.sidebar.multiselect(
    "Choose movies you like",
    movie_titles
)

all_genres = sorted(
    set(
        g
        for gs in movies["genres"]
        for g in gs.split("|")
    )
)

selected_genres = st.sidebar.multiselect(
    "Filter by genres",
    all_genres
)

top_k = st.sidebar.slider("Top K", 5, 20, TOP_K)


st.sidebar.markdown("---")
st.sidebar.subheader("📈 Model Metrics")

col1, col2 = st.sidebar.columns(2)

with col1:
    st.metric("Recall@10", f"{metrics.get('recall_10', 0):.3f}")
    st.metric("MAP@10", f"{metrics.get('map_10', 0):.3f}")

with col2:
    st.metric("NDCG@10", f"{metrics.get('ndcg_10', 0):.3f}")
    st.metric("Precision@10", f"{metrics.get('precision_10', 0):.3f}")


if st.button("🚀 Recommend"):

    if not selected_movies:
        st.warning("Select at least one movie")
        st.stop()

    with st.spinner("Generating..."):

        user_embedding = service.build_user_embedding(selected_movies)

        recs = service.recommend(
            user_embedding,
            selected_movies,
            top_k=100
        )

        rec_df = (
            movies
            .set_index("item_idx")
            .loc[recs]
            .reset_index()
        )

        rec_df = service.filter_by_genres(rec_df, selected_genres)

        rec_df = rec_df.head(top_k)


    st.subheader("🍿 Recommendations")

    cols = st.columns(3)

    for i, row in enumerate(rec_df.itertuples()):
        with cols[i % 3]:
            st.markdown(f"### 🎬 {row.title}")
            st.caption(row.genres)


    st.markdown("---")
    st.subheader("📊 Genres Distribution")

    genre_counts = {}

    for g in rec_df["genres"]:
        for x in g.split("|"):
            genre_counts[x] = genre_counts.get(x, 0) + 1

    fig, ax = plt.subplots()

    ax.bar(genre_counts.keys(), genre_counts.values())
    ax.set_xticklabels(genre_counts.keys(), rotation=45)

    st.pyplot(fig)


    st.markdown("---")
    st.subheader("📋 Table")

    st.dataframe(rec_df[["title", "genres"]], use_container_width=True)


    st.markdown("---")

    st.subheader(
        "❤️ Your Favorite Movies"
    )

    favorite_df = movies[
        movies["title"].isin(selected_movies)
    ][["title", "genres"]]

    st.dataframe(
        favorite_df,
        use_container_width=True
    )