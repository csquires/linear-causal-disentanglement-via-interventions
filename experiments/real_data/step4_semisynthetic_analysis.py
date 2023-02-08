# === IMPORTS: BUILT-IN ===
from argparse import ArgumentParser

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.dataset import Dataset
from src.run_experiment import ExperimentRunnerHelper
from experiments.real_data.config import get_solution

# === ARGUMENT PARSING
parser = ArgumentParser()
parser.add_argument("--num_obs", type=int, default=100)
parser.add_argument("--num_contexts", type=int, default=10)
args = parser.parse_args()
NUM_OBS = args.num_obs
NUM_CONTEXTS = args.num_contexts
NRUNS = 50

nsamples_list = [
    int(1e6),
    int(5e6),
    int(1e7), 
    int(5e7), 
    int(1e8), 
    # int(5e8)
]
seed = 1
rank_gamma = 0.99

# === LOAD DATA
sol = get_solution(NUM_OBS, NUM_CONTEXTS)

# === CREATE DATASETS
H = sol["H_est"]
B_obs = sol["B0_est"]
Bs = [B for ix, B in sol["B_ests"].items()]
num_latent = len(Bs)
ix2target = sol["ix2target"]
datasets = [
    Dataset(B_obs, np.eye(num_latent), H, Bs, ix2target) 
    for _ in range(NRUNS)
]

# === RUN EXPERIMENTS
setting = f"seed={seed},num_obs={NUM_OBS},num_context={NUM_CONTEXTS}"
er = ExperimentRunnerHelper(
    datasets,
    nsamples_list,
    num_latent,
    result_filename=f"experiments/real_data/semisynthetic_outputs/{setting}.pkl",
    plot_folder=f"experiments/real_data/semisynthetic_plots/{setting}",
    seed=seed,
    rank_gamma=rank_gamma,
    find_best_permutation="ilp"
)
er.run(overwrite=True)
er.plot()