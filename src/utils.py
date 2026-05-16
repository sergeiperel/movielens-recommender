import numpy as np


def build_interaction_matrix(df, n_users, n_items):

    matrix = np.zeros((n_users, n_items))

    for row in df.itertuples():
        matrix[row.user_idx, row.item_idx] = row.rating

    return matrix
