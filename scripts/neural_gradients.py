"""
Resources for calculating neural gradients during learning
"""

import numpy as np


def calc_g(C):
    return np.sum(C - C.T, axis=1)


def calc_g_norm(J, shuffle=False, rows=True):
    # calculates the neural gradients with p=2
    C = J**2
    if shuffle:
        C = np.random.permutation(C) if rows else np.random.permutation(C.T).T
    g = calc_g(C)
    return np.linalg.norm(g)


def calculate_neural_gradients(
        train_results,
        n_shuff=1e4
):
    gf_shuff_rows = [
        [
            calc_g_norm(res['weights'][-1]['W_rec'], shuffle=True, rows=True)
            for _ in range(n_shuff)
        ]
        for res in train_results
    ]
    gf_true = [
        calc_g_norm(res['weights'][-1]['W_rec'], shuffle=False)
        for res in train_results
    ]

    return gf_shuff_rows, gf_true


