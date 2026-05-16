import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "data"

ratings_path = BASE_DIR / "ratings.csv"
movies_path = BASE_DIR / "movies.csv"
links_path = BASE_DIR / "links.csv"

def load_data():

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    links = pd.read_csv(links_path)

    ratings["datetime"] = pd.to_datetime(
        ratings["timestamp"],
        unit="s"
    )

    ratings = ratings.drop(columns=["timestamp"])

    user_le = LabelEncoder()
    item_le = LabelEncoder()

    ratings["user_idx"] = user_le.fit_transform(
        ratings["userId"]
    )

    ratings["item_idx"] = item_le.fit_transform(
        ratings["movieId"]
    )

    movies = movies[
        movies["movieId"].isin(item_le.classes_)
    ].copy()

    movies["item_idx"] = item_le.transform(
        movies["movieId"]
    )

    return ratings, movies, links, user_le, item_le