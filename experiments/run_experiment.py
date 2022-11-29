# === IMPORTS: BUILT-IN ===
import os
import pickle
import random
from time import time
from typing import List
import itertools as itr

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import causaldag as cd
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns

# === IMPORTS: LOCAL ===
from src.solver_partial_order import IterativeProjectionPartialOrderSolver
from src.rank_tester import SVDRankOneScorer
from src.rand import rand_model
from src.dataset import Dataset
from src.utils.permutations import get_permutation_matrix


def standardize(Q: np.ndarray):
    ppp = Q.shape[0]
    first_nonzero_ixs = np.argmax(Q != 0, axis=1)
    first_nonzero_vals = Q[(list(range(ppp)), first_nonzero_ixs)]
    signs = np.sign(first_nonzero_vals)
    return Q * signs[:, None]


def permute_solution(H_est, B0_est, ix2target_est, perm):
    P = get_permutation_matrix(perm)
    H_est_perm = P @ H_est
    B0_est_perm = P @ B0_est @ P.T
    ix2target_est_perm = {ix: perm[target] for ix, target in ix2target_est.items()}

    return H_est_perm, B0_est_perm, ix2target_est_perm


def find_best_permutation_match(run_results: dict):
    # === TRUE DATA
    ds: Dataset = run_results["ds"]
    B0_true = ds.B_obs
    H_true = ds.H
    B_trues = ds.Bs
    p = B0_true.shape[0]

    # === ESTIMATES
    H_est = run_results["H_est"]
    B0_est = run_results["B0_est"]
    B_ests = run_results["B_ests"]
    ix2target = run_results["ix2target"]

    min_B0_error = float('inf')
    best_solution = None
    for perm in itr.permutations(range(p), p):
        H_est_perm, B0_est_perm, ix2target_est_perm = permute_solution(
            H_est,
            B0_est,
            ix2target,
            perm
        )

        # === ERRORS
        H_error = np.linalg.norm(H_true - H_est_perm)
        B0_error = np.linalg.norm(B0_true - B0_est_perm)
        matches = [ds.ix2target[ix] == ix2target_est_perm[ix] for ix in ix2target]
        correct_order = all(matches)

        if B0_error < min_B0_error:
            min_B0_error = B0_error
            best_solution = (H_error, B0_error, correct_order)

    return best_solution


class ExperimentRunner:
    def __init__(
        self,
        nnodes: int,
        nsamples_list: List[int],
        nruns: int,
        seed: int = None,
        density: float = 0.5,
        rank_gamma: float = 1 - 1e-8,
        find_best_permutation=True
    ):
        self.nnodes = nnodes
        self.nsamples_list = nsamples_list
        self.nruns = nruns
        self.seed = seed if seed is not None else np.random.randint(0, 2**20)
        self.density = density
        self.rank_gamma = rank_gamma
        self.find_best_permutation = find_best_permutation

        self.result_filename = "experiments/experiment1/results/"
        os.makedirs(self.result_filename, exist_ok=True)
        self.result_filename += f"nnodes={nnodes},density={density},nruns={nruns},seed={self.seed}.pkl"

    def run(self, overwrite=False):
        np.random.seed(self.seed)
        random.seed(self.seed)


        # === CREATE DAGS AND DATASETS
        dags = [
            cd.rand.directed_erdos(self.nnodes, density=self.density, random_order=False)
            for _ in range(self.nruns)
        ]
        nodes2num_ivs = {i: 1 for i in range(self.nnodes)}
        datasets = [
            rand_model(
                dag, 
                nodes2num_ivs, 
                list(range(self.nnodes)), 
                upper_triangular_h=False,
                orthogonal_h=False,
                rational=False,
                no_perm=True,
                shuffle_targets=True
            )
            for dag in dags
        ]

        # === CREATE SOLVER
        solver = IterativeProjectionPartialOrderSolver(
            SVDRankOneScorer(), 
            rank_gamma=self.rank_gamma
        )

        # === RUN
        if not os.path.exists(self.result_filename) or overwrite:
            info = {
                "results": dict(),
                "metadata": dict(
                    nsamples_list=self.nsamples_list,
                    nruns=self.nruns
                )
            }
            for s_ix, nsamples in enumerate(self.nsamples_list):
                for r_ix in trange(self.nruns):
                    ds = datasets[r_ix]
                    obs_ds = ds.sample_thetas(nsamples)
                    # obs_ds = None

                    start_time = time()
                    sol = solver.solve(obs_ds, ds, verbose=True)
                    time_spent = time() - start_time

                    difference = ds.H - sol["H_est"]
                    l2 = np.linalg.norm(difference)
                    if np.isnan(l2):
                        breakpoint()

                    info["results"][(s_ix, r_ix)] = dict(
                        ds=ds,
                        obs_ds=obs_ds,
                        H_est=sol["H_est"],
                        B0_est=sol["B0_est"],
                        B_ests=sol["B_ests"],
                        Q_est=sol["Q_est"],
                        ix2target=sol["ix2target"],
                        time_spent=time_spent
                )

            pickle.dump(info, open(self.result_filename, "wb"))
        else:
            info = pickle.load(open(self.result_filename, "rb"))

        return info


    def load_info(self):
        info = pickle.load(open(self.result_filename, "rb"))
        return info


    def compute_errors(self):
        # === LOAD DATA
        info = self.load_info()
        metadata = info["metadata"]
        results = info["results"]
        nsamples_list = metadata["nsamples_list"]
        nruns = metadata["nruns"]

        # === ARRAYS FOR RESULTS
        H_errors = np.zeros((nruns, len(nsamples_list)))
        B0_errors = np.zeros((nruns, len(nsamples_list)))
        Q_errors = np.zeros((nruns, len(nsamples_list)))
        correct_orders = np.zeros((nruns, len(nsamples_list)))

        # === POPULATE ARRAYS
        for s_ix in range(len(nsamples_list)):
            for r_ix in trange(nruns):
                run_results = results[(s_ix, r_ix)]

                if self.find_best_permutation:
                    H_error, B0_error, correct_order = find_best_permutation_match(run_results)
                else:
                    # === TRUE DATA
                    ds: Dataset2 = run_results["ds"]
                    B0_true = ds.B_obs
                    H_true = ds.H
                    B_trues = ds.Bs

                    # === ESTIMATES
                    H_est = run_results["H_est"]
                    B0_est = run_results["B0_est"]
                    B_ests = run_results["B_ests"]
                    ix2target = run_results["ix2target"]

                    # === ERRORS
                    H_error = np.linalg.norm(H_true - H_est)
                    B0_error = np.linalg.norm(B0_true - B0_est)
                    matches = [ds.ix2target[ix] == ix2target[ix] for ix in ix2target]
                    correct_order = all(matches)

                # === ERRORS
                H_errors[r_ix, s_ix] = H_error
                B0_errors[r_ix, s_ix] = B0_error
                correct_orders[r_ix, s_ix] = correct_order

        return H_errors, B0_errors, Q_errors, correct_orders


    def plot(self, local=True):
        info = self.load_info()
        metadata = info["metadata"]
        nsamples_list = metadata["nsamples_list"]
        H_errors, B0_errors, Q_errors, correct_orders = self.compute_errors()

        avg_H_error = np.median(H_errors, axis=0)
        avg_Q_error = np.median(Q_errors, axis=0)
        avg_B0_error = np.median(B0_errors, axis=0)
        percent_correct_order = np.mean(correct_orders, axis=0)

        folder = f"experiments/figures/nnodes={self.nnodes}"
        folder += f",density={self.density}"
        folder += f",seed={self.seed}"
        folder += f",gamma={self.rank_gamma}"
        os.makedirs(folder, exist_ok=True)
        sns.set()
        plt.style.use("style.mplstyle")
        # === PLOT ERRORS IN Q ===
        plt.clf()
        plt.plot(nsamples_list, avg_Q_error)
        plt.xscale("log")
        plt.xlabel("Number of samples")
        plt.ylabel("Median Frobenius error in Q")
        plt.tight_layout()
        plt.savefig(f"{folder}/avg_Q_error.png")
        if local: plt.savefig(os.path.expanduser("~/Downloads/avg_Q_error.png"))

        # === PLOT ERRORS IN H ===
        plt.clf()
        plt.plot(nsamples_list, avg_H_error)
        plt.xscale("log")
        plt.xlabel("Number of samples")
        plt.ylabel("Median Frobenius error in $H$")
        plt.tight_layout()
        plt.savefig(f"{folder}/avg_H_error.png")
        if local: plt.savefig(os.path.expanduser("~/Downloads/avg_H_error.png"))

        # === PLOT ERRORS IN B0 ===
        plt.clf()
        plt.plot(nsamples_list, avg_B0_error)
        plt.xscale("log")
        plt.xlabel("Number of samples")
        plt.ylabel("Median Frobenius error in $B_0$")
        plt.tight_layout()
        plt.savefig(f"{folder}/avg_B0_error.png")
        if local: plt.savefig(os.path.expanduser("~/Downloads/avg_B0_error.png"))

        # === PLOT ERRORS IN ORDER ===
        plt.clf()
        plt.plot(nsamples_list, percent_correct_order)
        plt.xscale("log")
        plt.xlabel("Number of samples")
        plt.ylabel("Fraction with all \nintervention targets correct")
        plt.tight_layout()
        plt.savefig(f"{folder}/percent_correct_order.png")
        if local: plt.savefig(os.path.expanduser("~/Downloads/percent_correct_order.png"))