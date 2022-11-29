# === IMPORTS: BUILT-IN ===
import itertools as itr

# === IMPORTS: THIRD-PARTY ===
import numpy as np


def mat2perm(mat):
    perm = []
    for i in range(mat.shape[0]):
        sigma_i = np.argmax(mat[i])
        perm.append(sigma_i)
    return perm


def get_permutation_matrix(perm: list):
    nodes = list(range(len(perm)))
    mat = np.zeros((len(perm), len(perm)), dtype=int)
    mat[nodes, perm] = 1
    return mat


def get_inverse_permutation(perm: list):
    return [perm.index(i) for i in range(len(perm))]


def all_permutations(p):
    perms = list(itr.permutations(list(range(p))))
    mats = [get_permutation_matrix(perm) for perm in perms]
    return list(zip(perms, mats))