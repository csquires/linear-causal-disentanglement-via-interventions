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
