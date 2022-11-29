# === IMPORTS: BUILT-IN ===
import itertools as itr

# === IMPORTS: THIRD-PARTY ===
from numpy.linalg import matrix_rank, svd
import numpy as np


class RankOneTester:
    def __init__(self):
        pass

    def is_rank_one(self, mat):
        pass


class ExactRankOneTester(RankOneTester):
    def __init__(self, tol=1e-8):
        self.tol = tol

    def is_rank_one(self, mat):
        return matrix_rank(mat, tol=1e-8) <= 1


class CutoffRankOneTester(RankOneTester):
    def __init__(self, cutoff=1e-2):
        self.cutoff = cutoff

    def is_rank_one(self, mat):
        u, s, v = svd(mat)
        s_sorted = sorted(s, reverse=True)
        return s_sorted[1] <= self.cutoff



class RankOneScorer:
    def __init__(self):
        pass

    def score(self, mat):
        pass


class ExactRankOneScorer(RankOneScorer):
    def __init__(self, tol=1e-8):
        self.tol = tol

    def score(self, mat):
        return matrix_rank(mat, tol=1e-8) <= 1


class SVDRankOneScorer(RankOneScorer):
    def __init__(self):
        pass

    def score(self, mat):
        u, s, v = svd(mat)
        s2 = np.sort(s) ** 2
        # print(s2)
        return s2[-1] / s2.sum()


class DeterminantRankOneScorer(RankOneScorer):
    def __init__(self):
        super().__init__()
    
    def score(self, mat):
        p = mat.shape[0]
        pairs = itr.combinations(range(p), 2)
        pairs_of_pairs = list(itr.combinations_with_replacement(pairs, 2))
        max_minor = float('-inf')
        for (i, j), (u, v) in pairs_of_pairs:
            submat = mat[np.ix_([i, j], [u, v])]
            minor = np.abs(np.linalg.det(submat))
            max_minor = max(minor, max_minor)
        return 1-max_minor