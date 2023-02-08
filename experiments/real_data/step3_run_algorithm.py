# === IMPORTS: BUILT-IN ===
import os
import pickle
from argparse import ArgumentParser

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import pandas as pd

# === IMPORTS: LOCAL ===
from src.solver_partial_order import IterativeProjectionPartialOrderSolver
from src.rank_tester import SVDRankOneScorer
from src.dataset import ObservedDataset
from experiments.real_data.config import ncontexts2variants, get_solution_filename
from experiments.real_data.config import PROCESSED_DATA_FOLDER, DISTRIBUTION_FOLDER

# === ARGUMENT PARSING
parser = ArgumentParser()
parser.add_argument("--num_obs", type=int, default=100)
parser.add_argument("--edge_cutoff", type=float, default=0.04)
parser.add_argument("--num_contexts", type=int, default=10)
args = parser.parse_args()
NUM_OBS = args.num_obs
CUTOFF = args.edge_cutoff
NUM_CONTEXTS = args.num_contexts

np.set_printoptions(suppress=True)

# === LOAD DATA
data = pickle.load(open(f"{PROCESSED_DATA_FOLDER}/thetas_num_obs={NUM_OBS}.pkl", "rb"))
Theta_dict = data["Theta_dict"]
Theta_obs = data["Theta_obs"]

# === PICK THETAS 
variants = pd.read_csv(ncontexts2variants[NUM_CONTEXTS], header=None, index_col=0)
Thetas = [Theta_dict[variant] for variant in variants.index]

# === RUN ALGORITHM
nlatent = len(Thetas)
solver = IterativeProjectionPartialOrderSolver(
    SVDRankOneScorer(), 
    num_latent=nlatent,
    rank_gamma=0.99
)
dataset = ObservedDataset(Theta_obs, Thetas)
sol = solver.solve(dataset)

H_est = sol["H_est"]
B0_est = sol["B0_est"]
B_ests = sol["B_ests"]
ix2target = sol["ix2target"]
partial_order = sol["partial_order"]

# === TRUNCATE MATRICES
sorted_offdiags = np.sort(np.abs(B0_est[np.triu_indices_from(B0_est, k=1)]))
median_offdiagonal = np.median(sorted_offdiags)
mean_offdiagonal = np.mean(sorted_offdiags)
print(f"Median off-diagonal entry: {median_offdiagonal}")
print(f"Mean off-diagonal entry: {mean_offdiagonal}")

percent_above_cutoff = (sorted_offdiags > CUTOFF).mean()
print(f"Percent above cutoff: {percent_above_cutoff}")

B0_est_new = B0_est.copy()
B0_est_new[np.abs(B0_est_new) < CUTOFF] = 0
B_ests_new = dict()
for ix, B_est in B_ests.items():
    B_est_new = B_est.copy()
    B_est_new[np.abs(B_est_new) < CUTOFF] = 0
    B_ests_new[ix] = B_est_new
sol["B0_est"] = B0_est_new
sol["B_ests"] = B_ests_new

# === SAVE_RESULTS
os.makedirs(DISTRIBUTION_FOLDER, exist_ok=True)
pickle.dump(sol, open(get_solution_filename(NUM_OBS, NUM_CONTEXTS), "wb"))
