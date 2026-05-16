import numpy as np
from numpy.linalg import svd


class SVDRecommender:

    def __init__(self, n_factors=20):

        self.n_factors = n_factors

    def fit(self, matrix):

        self.matrix = matrix

        U, S, Vt = svd(matrix, full_matrices=False)

        U = U[:, :self.n_factors]
        S = S[:self.n_factors]
        Vt = Vt[:self.n_factors, :]

        self.user_factors = U @ np.diag(np.sqrt(S))
        self.item_factors = np.diag(np.sqrt(S)) @ Vt

        return self

    def recommend(self, user_id, k=10):

        user_vector = self.matrix[user_id]

        user_embedding = (
            user_vector @ self.item_factors.T
        )

        scores = (
            user_embedding @ self.item_factors
        )

        interacted_items = np.where(user_vector > 0)[0]

        scores[interacted_items] = -np.inf

        top_items = np.argsort(scores)[::-1]

        return top_items[:k]