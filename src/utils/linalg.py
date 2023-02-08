# === IMPORTS: THIRD-PARTY ===
import numpy as np
from numpy.linalg import svd, lstsq
import causaldag as cd
from scipy.linalg import rq


def orthogonal_projection_matrix(vecs, mat):
    if len(vecs) == 0:
        return mat
    vmat = np.vstack(vecs).T
    xx2, _, _, _  = lstsq(vmat, mat, rcond=None)
    oo1 = mat - vmat @ xx2

    xx2, _, _, _ = lstsq(vmat, oo1.T, rcond=None)
    oo2 = oo1 - (vmat @ xx2).T
    
    return oo2


def RQ_partial_order(
    H: np.ndarray, 
    partial_order: cd.DAG
):
    d, p = H.shape
    R = np.zeros((d, d))
    Q = np.zeros((d, p))
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
    print("diff RQ-H", mdiff)

    return R, Q


def normalize(vec):
    return vec / np.linalg.norm(vec)


def normalize_H(H, return_scaling=False):
    p = H.shape[0]
    argmax_ixs = np.argmax(np.abs(H), axis=1)
    argmax_vals = H[list(range(p)), argmax_ixs]
    Hnew = H / argmax_vals[:, None]
    if return_scaling:
        return Hnew, argmax_vals
    else:
        return Hnew


def get_rank_one_factors(M):
    u_mat, s, v_mat = svd(M)
    ix = np.argmax(s)
    sval = s[ix]
    a1, a2 = u_mat[:, ix], v_mat[ix]
    return a1 * np.sqrt(sval), a2 * np.sqrt(sval)


def proj_vec(v, W):
    return W @ W.T @ v