import mlflow
import datetime
import numpy as np
import os
import joblib

from src.data import load_data
from src.utils import build_interaction_matrix
from src.metrics import recall_at_k, precision_at_k, map_at_k, ndcg_at_k

from src.models.svd_model import SVDRecommender
from src.models.als_model import ALSRecommender



K = 10
CUTOFF = datetime.datetime(2016, 1, 1)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

mlflow.set_experiment("MovieLens-Recsys")


ratings, movies, links, _, _ = load_data()

n_users = ratings["user_idx"].nunique()
n_items = ratings["item_idx"].nunique()


train_df = ratings[ratings["datetime"] < CUTOFF]
test_df = ratings[ratings["datetime"] >= CUTOFF]


train_matrix = build_interaction_matrix(train_df, n_users, n_items)

test_user_items = {
    u: set(g["item_idx"])
    for u, g in test_df.groupby("user_idx")
}


def build_test_dict(df):
    user2items = {}
    for row in df.itertuples():
        user2items.setdefault(row.user_idx, set()).add(row.item_idx)
    return user2items


test_user_items = build_test_dict(test_df)


def evaluate_model(model, train_matrix, test_user_items, k=10):

    recalls, precisions, maps, ndcgs = [], [], [], []

    for user in test_user_items.keys():

        if user >= train_matrix.shape[0]:
            continue

        recs = model.recommend(user, k=k)
        true_items = test_user_items[user]

        recalls.append(recall_at_k(recs, true_items, k))
        precisions.append(precision_at_k(recs, true_items, k))
        maps.append(map_at_k(recs, true_items, k))
        ndcgs.append(ndcg_at_k(recs, true_items, k))

    return {
        "recall_10": float(np.mean(recalls)),
        "precision_10": float(np.mean(precisions)),
        "map_10": float(np.mean(maps)),
        "ndcg_10": float(np.mean(ndcgs)),
    }


def save_model(model, name):
    path = os.path.join(ARTIFACT_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    return path



with mlflow.start_run(run_name="SVD"):

    svd = SVDRecommender(n_factors=20)
    svd.fit(train_matrix)

    mlflow.log_param("model", "SVD")
    mlflow.log_param("n_factors", 20)

    metrics = evaluate_model(svd, train_matrix, test_user_items, k=K)

    mlflow.log_metrics(metrics)

    svd_path = save_model(svd, "svd_model")

    mlflow.log_artifact(svd_path)



with mlflow.start_run(run_name="ALS"):

    als = ALSRecommender(factors=64)
    als.fit(train_matrix)

    mlflow.log_param("model", "ALS")
    mlflow.log_param("factors", 64)

    metrics = evaluate_model(als, train_matrix, test_user_items, k=K)

    mlflow.log_metrics(metrics)

    als_path = save_model(als, "als_model")

    mlflow.log_artifact(als_path)



