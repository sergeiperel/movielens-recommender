import numpy as np
import joblib
from config import MODEL_PATH, TOP_K


class RecommenderService:

    def __init__(self, movies_df):
        self.movies = movies_df
        self.model = joblib.load(MODEL_PATH)


    def build_user_embedding(self, selected_titles):

        item_embeddings = []

        for title in selected_titles:

            item_idx = self.movies.loc[
                self.movies["title"] == title,
                "item_idx"
            ].values[0]

            item_embeddings.append(
                self.model.model.item_factors[item_idx]
            )

        return np.mean(item_embeddings, axis=0)


    def recommend(self, user_embedding, selected_titles, top_k=TOP_K):

        item_factors = self.model.model.item_factors
        scores = item_factors @ user_embedding
        selected_idx = self.movies[
            self.movies["title"].isin(selected_titles)
        ]["item_idx"].values

        scores[selected_idx] = -np.inf

        top_items = np.argsort(scores)[::-1][:top_k]

        return top_items


    def filter_by_genres(self, rec_df, selected_genres):

        if not selected_genres:
            return rec_df

        return rec_df[
            rec_df["genres"].apply(
                lambda x: any(
                    g in x for g in selected_genres
                )
            )
        ]