import numpy as np


# =========================
# RECALL@K
# =========================

def recall_at_k(recommended_items, relevant_items, k=10):
    recommended_items = recommended_items[:k]

    if len(relevant_items) == 0:
        return 0.0

    hits = len(set(recommended_items) & set(relevant_items))

    return hits / len(relevant_items)


# =========================
# PRECISION@K
# =========================

def precision_at_k(recommended_items, relevant_items, k=10):
    recommended_items = recommended_items[:k]

    if k == 0:
        return 0.0

    hits = len(set(recommended_items) & set(relevant_items))

    return hits / k


# =========================
# AP@K (Average Precision for one user)
# =========================

def average_precision_at_k(recommended_items, relevant_items, k=10):
    recommended_items = recommended_items[:k]
    relevant_items = set(relevant_items)

    if len(relevant_items) == 0:
        return 0.0

    hits = 0
    score = 0.0

    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(relevant_items), k)


# =========================
# MAP@K
# =========================

def map_at_k(recommended_items, relevant_items, k=10):
    return average_precision_at_k(recommended_items, relevant_items, k)


# =========================
# DCG / NDCG
# =========================

def dcg_at_k(recommended_items, relevant_items, k=10):
    recommended_items = recommended_items[:k]
    relevant_items = set(relevant_items)

    dcg = 0.0

    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0 issue

    return dcg


def ndcg_at_k(recommended_items, relevant_items, k=10):
    dcg = dcg_at_k(recommended_items, relevant_items, k)

    # ideal DCG
    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0

    return dcg / idcg