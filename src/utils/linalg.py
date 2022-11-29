# === IMPORTS: BUILT-IN ===
from typing import List

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from numpy.linalg import svd, lstsq
from scipy.linalg import null_space, orth
from sympy import Matrix, sqrt
from sympy.matrices import GramSchmidt
import causaldag as cd
from scipy.linalg import rq


def RQ_partial_order(
    H: np.ndarray, 
    partial_order: cd.DAG
):
    Q = np.zeros(H.shape)
    R = np.zeros(H.shape)
    p = H.shape[0]
    order = partial_order.topological_sort()

    for i in order:
        ancestors = list(partial_order.ancestors_of(i))

        if len(ancestors) > 0:
            Hcurrent = np.vstack((H[i], Q[ancestors]))
            Rsubset, Qsubset = rq(Hcurrent, mode="economic")
                
            Q[i] = Qsubset[0]
            QQ = np.vstack((Q[i], Q[ancestors]))
            r = lstsq(QQ.T, H[i])[0]
            R[i, i] = r[0]
            R[i, ancestors] = r[1:]
        else:
            Q[i] = normalize(H[i])
            R[i, i] = np.linalg.norm(H[i])
    
    diff = R @ Q - H
    mdiff = np.max(np.abs(diff))
    print(mdiff)

    return R, Q



def normalize(vec):
    return vec / np.linalg.norm(vec)


def normalize_H(H):
    p = H.shape[0]
    argmax_ixs = np.argmax(np.abs(H), axis=1)
    argmax_vals = H[list(range(p)), argmax_ixs]
    Hnew = H / argmax_vals[:, None]
    return Hnew


class SubspaceFloat:
    def __init__(self, vecs: List[np.ndarray], vlength=None):
        self.vecs = vecs
        if len(vecs) > 0:
            self.W = orth(np.array(vecs).T)  # p * d
            self.Wperp = null_space(self.W.T)  # p * (p - d)
        else:
            self.W = np.zeros((vlength, 0))
            self.Wperp = np.eye(vlength)

    @property
    def ambient_dim(self):
        return self.W.shape[0]

    @property
    def dim(self):
        return self.W.shape[1]

    def quotient(self, mat):
        assert mat.shape[0] == self.ambient_dim
        return self.Wperp @ self.Wperp.T @ mat

    def add(self, vecs: List[np.ndarray]):
        return SubspaceFloat(self.vecs + vecs)

    def project_on_on(self, mat):
        return proj_mat(mat, self.W, self.W)

    def project_on_orth(self, mat):
        return proj_mat(mat, self.W, self.Wperp)

    def project_orth_on(self, mat):
        return proj_mat(mat, self.Wperp, self.W)

    def project_orth_orth(self, mat):
        return proj_mat(mat, self.Wperp, self.Wperp)


class MatrixSubspaceFloat:
    def __init__(self, mats, vecs=None):
        new_vecs = [get_rank_one_factors(mat)[0] for mat in mats]
        existing_vecs = [] if vecs is None else vecs
        self.vecs = new_vecs + existing_vecs
        self.subspace = SubspaceFloat(self.vecs)

    def add(self, mats):
        return MatrixSubspaceFloat(mats, vecs=self.vecs) 

    def project_on_on(self, mat):
        return proj_mat(mat, self.subspace.W, self.subspace.W)

    def project_on_orth(self, mat):
        return proj_mat(mat, self.subspace.W, self.subspace.Wperp)

    def project_orth_on(self, mat):
        return proj_mat(mat, self.subspace.Wperp, self.subspace.W)

    def project_orth_orth(self, mat):
        return proj_mat(mat, self.subspace.Wperp, self.subspace.Wperp)



class Subspace:
    def __init__(self, vecs: List[np.ndarray]):
        """
        A class representing a linear subspace.

        Params
        ------
        vecs: a list of vectors
        """
        self.vecs = Matrix(vecs).T

        # === REPRESENT THE SUBSPACE
        aa = GramSchmidt([Matrix(v) for v in vecs], orthonormal=True)
        self.W = Matrix(aa)

        # === REPRESENT THE ORTHOCOMPLEMENT OF THE SUBSPACE
        nullspace = self.vecs.nullspace()
        bb = GramSchmidt(nullspace, orthonormal=True)
        self.Wperp = Matrix.hstack(*bb)

    @property
    def dim(self):
        return self.W.shape[1]

    def add(self, vecs):
        pass

    def project(self, vec):
        pass


class MatrixSubspace:
    def __init__(self, mats, vecs=None):
        print("Rank one factors")
        new_vecs = [get_rank_one_factors_sympy(mat)[0] for mat in mats]
        existing_vecs = [] if vecs is None else vecs
        self.vecs = new_vecs + existing_vecs
        self.subspace = Subspace(self.vecs)

    def add(self, mats):
        return MatrixSubspace(mats, vecs=self.vecs) 

    def project_on_on(self, mat):
        return proj_mat(mat, self.subspace.W, self.subspace.W)

    def project_on_orth(self, mat):
        return proj_mat(mat, self.subspace.W, self.subspace.Wperp)

    def project_orth_on(self, mat):
        return proj_mat(mat, self.subspace.Wperp, self.subspace.W)

    def project_orth_orth(self, mat):
        return proj_mat(mat, self.subspace.Wperp, self.subspace.Wperp)


def get_rank_one_factors(M):
    u_mat, s, v_mat = svd(M)
    ix = np.argmax(s)
    sval = s[ix]
    a1, a2 = u_mat[:, ix], v_mat[ix]
    return a1 * np.sqrt(sval), a2 * np.sqrt(sval)


def get_rank_one_factors_sympy(M: Matrix):
    print("SVD in rank_one_factors")
    print(M.shape)
    uu, ss, vv = M.singular_value_decomposition()
    print("Done")
    ix = np.argmax(ss)
    sval = ss[ix]
    a1, a2 = uu[:, ix], vv[ix]
    return a1 * sqrt(sval), a2 * sqrt(sval)


def proj_vec(v, W):
    return W @ W.T @ v


def proj_mat(M, W1, W2):
    """
    Given a `p*p` matrix `M`, a `p*r1` matrix `W1`, and a `p*r2` matrix `W2`,
    project the rows of `M` onto `W1` and the columns of `M` onto `W2`
    """
    p1, p2, p3 = M.shape[0], W1.shape[0], W2.shape[0]
    assert p1 == p2 and p2 == p3
    return W1 @ W1.T @ M @ W2 @ W2.T



if __name__ == "__main__":
    # H = np.random.uniform(size=(4, 4))
    # partial_order = cd.DAG(arcs={(3, 0), (2, 0), (2, 1)})
    # R, Q = RQ_partial_order(H, partial_order)
    # print(R)
    # print(np.isclose(Q @ Q.T, 0).astype(int))

    H = np.random.uniform(size=(4, 4))
    partial_order = cd.DAG(arcs={(3, 2), (2, 1), (1, 0)})
    R, Q = RQ_partial_order(H, partial_order)
    print(R)
    R2, Q2 = rq(H)
    print(np.isclose(Q @ Q.T, 0).astype(int))

    # H = np.random.uniform(size=(4, 4))
    # partial_order = cd.DAG(nodes={0, 1, 2, 3})
    # R, Q = RQ_partial_order(H, partial_order)
    # print(np.isclose(Q @ Q.T, 0).astype(int))