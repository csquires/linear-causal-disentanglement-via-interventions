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
from scipy.linalg import rq
import networkx as nx

# === IMPORTS: LOCAL ===
from src.solver_partial_order import IterativeProjectionPartialOrderSolver
from src.rank_testers import SVDRankOneScorer
from src.rand import rand_model
from src.dataset import Dataset2
from src.utils.permutations import get_permutation_matrix
from src.matching import IntegerProgram


def standardize(Q):
    ppp = Q.shape[0]
    first_nonzero_ixs = np.argmax(Q != 0, axis=1)
    first_nonzero_vals = Q[(list(range(ppp)), first_nonzero_ixs)]
    signs = np.sign(first_nonzero_vals)
    return Q * signs[:, None]


def permute_solution(H_est, B0_est, B_ests, ix2target_est, perm):
    P = get_permutation_matrix(perm)
    H_est_perm = P @ H_est
    B0_est_perm = P @ B0_est @ P.T
    B_ests_perm = {ix: P @ B_est @ P.T for ix, B_est in B_ests.items()}
    ix2target_est_perm = {ix: perm[target] for ix, target in ix2target_est.items()}

    return H_est_perm, B0_est_perm, B_ests_perm, ix2target_est_perm


def find_best_permutation_match_ilp(run_results: dict):
    # === TRUE DATA
    ds: Dataset2 = run_results["ds"]
    B0_true = ds.B_obs
    B1_true = ds.Bs[0]
    H_true = ds.H
    B_trues = ds.Bs
    p = B0_true.shape[0]

    # === ESTIMATES
    H_est = run_results["H_est"]
    B0_est = run_results["B0_est"]
    B1_est = run_results["B_ests"][0]
    B_ests = run_results["B_ests"]
    est_ix2target = run_results["ix2target"]

    true_graph = cd.DAG(
        nodes=list(range(p)),
        arcs={(i, j) for i, j in itr.combinations(range(p), 2) if ds.B_obs[i, j] != 0}
    )
    ip = IntegerProgram(true_graph, ds.ix2target, est_ix2target)
    inv_perm = ip.solve()
    perm = [inv_perm.index(ix) for ix in range(p)]

    H_est_perm, B0_est_perm, B_ests_perm, ix2target_est_perm = permute_solution(
        H_est,
        B0_est,
        B_ests,
        est_ix2target,
        perm
    )

    # === ERRORS
    H_error = np.linalg.norm(H_true - H_est_perm)
    B0_error = np.linalg.norm(B0_true - B0_est_perm)
    B1_error = np.linalg.norm(B1_true - B_ests_perm[0])
    matches = [ds.ix2target[ix] == ix2target_est_perm[ix] for ix in est_ix2target]
    nmatches = sum(matches)
    correct_order = all(matches)
    solution = (H_error, B0_error, B1_error, correct_order, perm, nmatches)
        
    return solution


def find_best_permutation_match_naive2(run_results: dict):
    # === TRUE DATA
    ds: Dataset2 = run_results["ds"]
    B0_true = ds.B_obs
    B1_true = ds.Bs[0]
    H_true = ds.H
    B_trues = ds.Bs
    p = B0_true.shape[0]

    # === ESTIMATES
    H_est = run_results["H_est"]
    B0_est = run_results["B0_est"]
    B1_est = run_results["B_ests"][0]
    B_ests = run_results["B_ests"]
    ix2target = run_results["ix2target"]

    best_solution = None
    true_graph = nx.DiGraph([(i, j) for i, j in itr.combinations(range(p), 2) if ds.B_obs[i, j] != 0])
    true_graph.add_nodes_from(list(range(p)))

    best_nmatches = -1
    for perm in nx.all_topological_sorts(true_graph):
        H_est_perm, B0_est_perm, B_ests_perm, ix2target_est_perm = permute_solution(
            H_est,
            B0_est,
            B_ests,
            ix2target,
            perm
        )

        # === ERRORS
        matches = [ds.ix2target[ix] == ix2target_est_perm[ix] for ix in ix2target]
        nmatches = sum(matches)
        correct_order = all(matches)

        if nmatches > best_nmatches:
            H_error = np.linalg.norm(H_true - H_est_perm)
            B0_error = np.linalg.norm(B0_true - B0_est_perm)
            B1_error = np.linalg.norm(B1_true - B_ests_perm[0])

            best_solution = (H_error, B0_error, B1_error, correct_order, perm, nmatches)
            best_nmatches = nmatches
        
    return best_solution


def find_best_permutation_match_naive(run_results: dict):
    # === TRUE DATA
    ds: Dataset2 = run_results["ds"]
    B0_true = ds.B_obs
    B1_true = ds.Bs[0]
    H_true = ds.H
    B_trues = ds.Bs
    p = B0_true.shape[0]

    # === ESTIMATES
    H_est = run_results["H_est"]
    B0_est = run_results["B0_est"]
    B1_est = run_results["B_ests"][0]
    B_ests = run_results["B_ests"]
    ix2target = run_results["ix2target"]

    min_B0_error = float('inf')
    best_solution = None
    true_graph = nx.DiGraph([(i, j) for i, j in itr.combinations(range(p), 2) if ds.B_obs[i, j] != 0])
    true_graph.add_nodes_from(list(range(p)))
    for perm in nx.all_topological_sorts(true_graph):
        H_est_perm, B0_est_perm, B_ests_perm, ix2target_est_perm = permute_solution(
            H_est,
            B0_est,
            B_ests,
            ix2target,
            perm
        )

        # === ERRORS
        H_error = np.linalg.norm(H_true - H_est_perm)
        B0_error = np.linalg.norm(B0_true - B0_est_perm)
        B1_error = np.linalg.norm(B1_true - B_ests_perm[0])
        matches = [ds.ix2target[ix] == ix2target_est_perm[ix] for ix in ix2target]
        correct_order = all(matches)

        if B0_error < min_B0_error:
            min_B0_error = B0_error
            best_solution = (H_error, B0_error, B1_error, correct_order, perm)

            # record the below for debugging
            best_perm = perm
            best_ix2target_preperm = ix2target
            best_H_est = np.round(H_est_perm, 3)
            best_B0_est = np.round(B0_est_perm, 3)
            best_B_ests_perm = {ix: np.round(B_est, 3) for ix, B_est in B_ests_perm.items()}
            best_ix2target = ix2target_est_perm
    
    print("=========")
    print(best_ix2target)
    print(ds.ix2target)
    
    # if not best_solution[-1]:
    #     print("True B0:")
    #     print(np.round(B0_true, 3))
    #     print("Estimated B0:")
    #     print(best_B0_est)

    #     print("True ix2target:")
    #     print(ds.ix2target)
    #     print("Estimated ix2target:")
    #     print(best_ix2target)

    #     for ix, B_est in best_B_ests_perm.items():
    #         print("=========")
    #         print(f"True B{ix}:")
    #         print(B_trues[ix])
    #         print(f"Estimated B{ix}:")
    #         print(B_est)
        
    #     breakpoint()
        
    return best_solution


class ExperimentRunnerHelper:
    def __init__(
        self, 
        datasets: List[Dataset2],
        nsamples_list: List[int],
        num_latent: int,
        result_filename: str,
        plot_folder: str,
        seed = 0,
        rank_gamma = 0.99,
        find_best_permutation = "ilp"
    ):
        self.datasets = datasets
        self.num_latent = num_latent
        self.nruns = len(self.datasets)
        self.result_filename = result_filename
        self.seed = seed
        self.rank_gamma = rank_gamma
        self.nsamples_list = nsamples_list
        self.find_best_permutation = find_best_permutation
        self.plot_folder = plot_folder

    def run(self, noiseless=False, verbose=False, overwrite=False):
        np.random.seed(self.seed)
        random.seed(self.seed)

        # === CREATE SOLVER
        solver = IterativeProjectionPartialOrderSolver(
            SVDRankOneScorer(), 
            rank_gamma=self.rank_gamma,
            num_latent=self.num_latent
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
                print(nsamples)
                for r_ix in trange(self.nruns):
                    ds = self.datasets[r_ix]
                    obs_ds = ds.sample_thetas(nsamples)
                    if noiseless:
                        print("NOISELESS")
                        obs_ds = ds

                    start_time = time()
                    sol = solver.solve(obs_ds, ds, verbose=verbose)
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
        B1_errors = np.zeros((nruns, len(nsamples_list)))
        Q_errors = np.zeros((nruns, len(nsamples_list)))
        correct_orders = np.zeros((nruns, len(nsamples_list)))

        # === POPULATE ARRAYS
        for s_ix in range(len(nsamples_list)):
            print(f"Computing errors for {nsamples_list[s_ix]}")
            for r_ix in trange(nruns):
                run_results = results[(s_ix, r_ix)]

                if self.find_best_permutation == "naive":
                    H_error, B0_error, B1_error, correct_order, perm, nmatches = find_best_permutation_match_naive2(run_results)
                elif self.find_best_permutation == "ilp":
                    H_error, B0_error, B1_error, correct_order, perm, nmatches = find_best_permutation_match_ilp(run_results)
                else:
                    # === TRUE DATA
                    ds: Dataset2 = run_results["ds"]
                    B0_true = ds.B_obs
                    H_true = ds.H
                    B_trues = ds.Bs
                    B1_true = B_trues[0]

                    # === ESTIMATES
                    H_est = run_results["H_est"]
                    B0_est = run_results["B0_est"]
                    B_ests = run_results["B_ests"]
                    B1_est = B_ests[0]
                    ix2target = run_results["ix2target"]

                    # === ERRORS
                    H_error = np.linalg.norm(H_true - H_est)
                    B0_error = np.linalg.norm(B0_true - B0_est)
                    B1_error = np.linalg.norm(B1_true - B1_est)
                    matches = [ds.ix2target[ix] == ix2target[ix] for ix in ix2target]
                    correct_order = all(matches)

                # === ERRORS
                H_errors[r_ix, s_ix] = H_error
                B0_errors[r_ix, s_ix] = B0_error
                B1_errors[r_ix, s_ix] = B0_error
                correct_orders[r_ix, s_ix] = correct_order

        return H_errors, B0_errors, B1_errors, Q_errors, correct_orders


    def plot(self, local=True):
        info = self.load_info()
        metadata = info["metadata"]
        nsamples_list = metadata["nsamples_list"]
        H_errors, B0_errors, B1_errors, Q_errors, correct_orders = self.compute_errors()

        avg_H_error = np.mean(H_errors, axis=0)
        avg_Q_error = np.mean(Q_errors, axis=0)
        avg_B0_error = np.mean(B0_errors, axis=0)
        avg_B1_error = np.mean(B1_errors, axis=0)
        percent_correct_order = np.mean(correct_orders, axis=0)

        median_H_error = np.median(H_errors, axis=0)
        median_Q_error = np.median(Q_errors, axis=0)
        median_B0_error = np.median(B0_errors, axis=0)
        median_B1_error = np.median(B1_errors, axis=0)
        percent_correct_order = np.mean(correct_orders, axis=0)
        
        os.makedirs(self.plot_folder, exist_ok=True)
        sns.set()
        plt.style.use("style.mplstyle")
        # === PLOT ERRORS IN Q ===
        plt.clf()
        plt.plot(nsamples_list, avg_Q_error)
        plt.xscale("log")
        plt.xlabel("Number of samples")
        plt.ylabel("Mean Frobenius error in Q")
        plt.tight_layout()
        plt.savefig(f"{self.plot_folder}/avg_Q_error.png")
        if local: plt.savefig(os.path.expanduser("~/Downloads/avg_Q_error.png"))

        # === PLOT ERRORS IN H ===
        plt.clf()
        plt.plot(nsamples_list, avg_H_error)
        plt.xscale("log")
        plt.xlabel("Number of samples")
        plt.ylabel("Mean Frobenius error in $H$")
        plt.tight_layout()
        plt.savefig(f"{self.plot_folder}/avg_H_error.png")
        plt.savefig(f"{self.plot_folder}/avg_H_error.pdf")
        if local: plt.savefig(os.path.expanduser("~/Downloads/avg_H_error.png"))

        plt.clf()
        plt.plot(nsamples_list, median_H_error)
        plt.xscale("log")
        plt.xlabel("Number of samples")
        plt.ylabel("Median Frobenius error in $H$")
        plt.tight_layout()
        plt.savefig(f"{self.plot_folder}/median_H_error.png")
        if local: plt.savefig(os.path.expanduser("~/Downloads/median_H_error.png"))

        # === PLOT ERRORS IN B0 ===
        plt.clf()
        plt.plot(nsamples_list, avg_B0_error)
        plt.xscale("log")
        plt.xlabel("Number of samples")
        plt.ylabel("Mean Frobenius error in $B_0$")
        plt.tight_layout()
        plt.savefig(f"{self.plot_folder}/avg_B0_error.png")
        plt.savefig(f"{self.plot_folder}/avg_B0_error.pdf")
        if local: plt.savefig(os.path.expanduser("~/Downloads/avg_B0_error.png"))

        plt.clf()
        plt.plot(nsamples_list, median_B0_error)
        plt.xscale("log")
        plt.xlabel("Number of samples")
        plt.ylabel("Median Frobenius error in $B_0$")
        plt.tight_layout()
        plt.savefig(f"{self.plot_folder}/median_B0_error.png")
        if local: plt.savefig(os.path.expanduser("~/Downloads/median_B0_error.png"))

        # === PLOT ERRORS IN B1 ===
        plt.clf()
        plt.plot(nsamples_list, avg_B1_error)
        plt.xscale("log")
        plt.xlabel("Number of samples")
        plt.ylabel("Mean Frobenius error in $B_1$")
        plt.tight_layout()
        plt.savefig(f"{self.plot_folder}/avg_B1_error.png")
        if local: plt.savefig(os.path.expanduser("~/Downloads/avg_B1_error.png"))

        plt.clf()
        plt.plot(nsamples_list, median_B1_error)
        plt.xscale("log")
        plt.xlabel("Number of samples")
        plt.ylabel("Median Frobenius error in $B_1$")
        plt.tight_layout()
        plt.savefig(f"{self.plot_folder}/median_B1_error.png")
        if local: plt.savefig(os.path.expanduser("~/Downloads/median_B1_error.png"))

        # === PLOT ERRORS IN ORDER ===
        plt.clf()
        plt.plot(nsamples_list, percent_correct_order)
        plt.xscale("log")
        plt.xlabel("Number of samples")
        plt.ylabel("Fraction with all \nintervention targets correct")
        plt.tight_layout()
        plt.savefig(f"{self.plot_folder}/percent_correct_order.png")
        plt.savefig(f"{self.plot_folder}/percent_correct_order.pdf")
        if local: plt.savefig(os.path.expanduser("~/Downloads/percent_correct_order.png"))


class ExperimentRunner2:
    def __init__(
        self,
        num_latent: int,
        nsamples_list: List[int],
        nruns: int,
        seed: int = None,
        density: float = 0.5,
        rank_gamma: float = 1 - 1e-8,
        find_best_permutation = "ilp",
        nnodes_obs: int = None,
        iv_type: str = "hard",
        experiment_name: str = "experiment1"
    ):
        self.num_latent = num_latent
        self.nnodes_obs = nnodes_obs if nnodes_obs is not None else num_latent
        self.nsamples_list = nsamples_list
        self.nruns = nruns
        self.seed = seed if seed is not None else np.random.randint(0, 2**20)
        self.density = density
        self.rank_gamma = rank_gamma
        self.find_best_permutation = find_best_permutation
        self.iv_type = iv_type
        
        self.result_filename = f"experiments/{experiment_name}/results/"
        os.makedirs(self.result_filename, exist_ok=True)
        self.result_filename += f"num_latent={num_latent}"
        self.result_filename += f",num_observed={self.nnodes_obs}"
        self.result_filename += f",iv_type={self.iv_type}"
        self.result_filename += f",density={density}"
        self.result_filename += f",nruns={nruns}"
        self.result_filename += f",seed={self.seed}.pkl"

        self.plot_folder = f"experiments/{experiment_name}/figures/num_latent={self.num_latent}"
        self.plot_folder += f",num_obs={self.nnodes_obs}"
        self.plot_folder += f",density={self.density}"
        self.plot_folder += f",seed={self.seed}"
        self.plot_folder += f",gamma={self.rank_gamma}"

    def run(self, overwrite=False):
        np.random.seed(self.seed)
        random.seed(self.seed)

        # === CREATE DAGS AND DATASETS
        dags = [
            cd.rand.directed_erdos(self.num_latent, density=self.density, random_order=False)
            for _ in range(self.nruns)
        ]
        nodes2num_ivs = {i: 1 for i in range(self.num_latent)}
        datasets = [
            rand_model(
                dag, 
                nodes2num_ivs, 
                list(range(self.num_latent)), 
                upper_triangular_h=False,
                orthogonal_h=False,
                rational=False,
                no_perm=True,
                shuffle_targets=True,
                nnodes_obs=self.nnodes_obs,
                iv_type=self.iv_type
            )
            for dag in dags
        ]

        self.experiment_runner_helper = ExperimentRunnerHelper(
            datasets,
            self.nsamples_list,
            self.num_latent,
            self.result_filename,
            self.plot_folder,
            seed=self.seed,
            rank_gamma=self.rank_gamma,
            find_best_permutation=self.find_best_permutation
        )

        self.experiment_runner_helper.run(overwrite=overwrite)

    def load_info(self):
        return self.experiment_runner_helper.load_info()

    def compute_errors(self):
        return self.experiment_runner_helper.compute_errors()

    def plot(self, local=True):
        self.experiment_runner_helper.plot()