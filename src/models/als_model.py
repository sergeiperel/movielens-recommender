from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import numpy as np


class ALSRecommender:

    def __init__(
        self,
        factors=64,
        regularization=0.01,
        iterations=20
    ):

        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations

        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations
        )

    def fit(self, matrix):

        self.matrix = csr_matrix(matrix)

        self.model.fit(self.matrix)

        return self

    def recommend(
        self,
        user_id,
        k=10
    ):

        ids, scores = self.model.recommend(
            userid=user_id,
            user_items=self.matrix[user_id],
            N=k,
            filter_already_liked_items=True
        )

        return ids