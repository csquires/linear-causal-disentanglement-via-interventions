# === IMPORTS: THIRD-PARTY ===
import numpy as np


def get_permutation_matrix(perm: list):
    nodes = list(range(len(perm)))
    mat = np.zeros((len(perm), len(perm)), dtype=int)
    # mat[nodes, perm] = 1
    mat[perm, nodes] = 1
    return mat


def get_inverse_permutation(perm: list):
    return [perm.index(i) for i in range(len(perm))]
