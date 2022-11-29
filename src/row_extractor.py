import numpy as np


class SingleRowExtractor:
    def __init__(self):
        pass

    def extract_row(self, mat):
        pass


class MaxNormSingleRowExtractor(SingleRowExtractor):
    def __init__(self, tol=1e-8):
        self.tol = tol

    def extract_row(self, mat, forbidden_rows=None):
        mat[np.abs(mat) < 1e-8] = 0
        norms = np.linalg.norm(mat, axis=1)
        if forbidden_rows is not None:
            norms[list(forbidden_rows)] = -float("inf")
        target = np.argmax(norms)
        row = mat[target]
        return target, row
