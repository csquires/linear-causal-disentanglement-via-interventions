# === IMPORTS: BUILT-IN ===
import random

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import causaldag as cd
from scipy.linalg import orth

# === IMPORTS: LOCAL ===
from src.utils.permutations import get_inverse_permutation, get_permutation_matrix
from src.dataset import Dataset
from src.utils.linalg import normalize_H



def get_intervention(
    B_obs: np.ndarray, 
    ix: int, 
    iv_type="hard"
):
    p = B_obs.shape[0]
    Bk = B_obs.copy()
    new_vals = np.zeros(p)
    new_vals[ix] = np.round(np.random.uniform(6, 8), 2)

    if iv_type == "soft":
        new_vals[ix+1:] = np.round(np.random.uniform(.25, 1, size=p-ix-1), 2)
    
    Bk[ix, :] = new_vals
    return Bk


def rand_model(
    latent_dag: cd.DAG,
    nodes2num_ivs: dict,
    perm: list = None,
    unipotent: bool = False,
    observed: bool = False,
    upper_triangular_h: bool = True,
    orthogonal_h: bool = False,
    seed: int = None,
    no_perm: bool = False,
    shuffle_targets: bool = False
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    g = cd.rand.rand_weights(latent_dag)
    p = latent_dag.nnodes

    # === CREATE OBSERVATIONAL B MATRIX ===
    I = np.eye(p)
    A = g.to_amat()
    if unipotent:
        Omega_half = np.diag(np.ones(p))
    else:
        Omega_half = np.diag(np.random.uniform(2, 4, p))
    
    B_obs = np.round(Omega_half @ (I - A), 3)

    # === CREATE H MATRIX ===
    if observed:
        H = np.eye(p, dtype=int)
    else:
        if upper_triangular_h:
            H_ut = np.triu(np.round(np.random.uniform(-2, 2, size=(p, p)), 2), k=1)
            H = H_ut + np.eye(p)
        else:
            H = np.round(np.random.uniform(-2, 2, size=(p, p)), 2)
            H = normalize_H(H)
        if orthogonal_h:
            H = orth(H)

    # === CREATE P MATRIX ===
    perm = list(range(p)) if perm is None else perm
    invperm = get_inverse_permutation(perm)
    P = get_permutation_matrix(perm)
    if no_perm:
        P = np.eye(p, dtype=int)

    # === CREATE INTERVENTIONAL B MATRICES ===
    targets = []
    for node, num_ivs in nodes2num_ivs.items():
        ix = invperm[node]
        targets += [ix for _ in range(num_ivs)]
    if shuffle_targets:
        random.shuffle(targets)
    ix2target = dict(enumerate(targets))

    Bs = []
    for ix in targets:
        B = get_intervention(B_obs, ix)
        Bs.append(B)

    return Dataset(
        B_obs,
        P,
        H,
        Bs,
        ix2target
    )








