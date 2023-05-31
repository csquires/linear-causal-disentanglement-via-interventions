# === IMPORTS: BUILT-IN ===
import random

# === IMPORTS: THIRD-PARTY ===
import causaldag as cd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

# === IMPORTS: LOCAL ===
from src.rand import rand_model, rand_model_multinode_intervention
from src.dataset import Dataset

n_latent = 5
n_obs = 10
density = 0.75

nruns = 500
nsamples = 2500
datasets = []

# === GENERATE IN-MODEL DATASETS
for _ in trange(nruns):
    dag = cd.rand.directed_erdos(n_latent, density)
    nodes2num_ivs = {node: 1 for node in range(n_latent)}
    ds = rand_model(
        dag,
        nodes2num_ivs,
        nnodes_obs=n_obs
    )
    obs_ds = ds.sample_thetas(nsamples)
    datasets.append(obs_ds)

# === GENERATE OUT-OF-MODEL DATASETS
for _ in trange(nruns):
    dag = cd.rand.directed_erdos(n_latent, density)
    ix2target = [random.sample(list(range(n_latent)), 2) for ix in range(n_latent)]
    ds = rand_model_multinode_intervention(
        dag,
        ix2target,
        nnodes_obs=n_obs
    )
    obs_ds = ds.sample_thetas(nsamples)
    datasets.append(obs_ds)


def compute_statistics(dataset: Dataset):
    Theta_obs = dataset.Theta_obs
    all_stats = []
    for Theta in dataset.Thetas:
        u, s, v = np.linalg.svd(Theta - Theta_obs)
        s_sorted = np.sort(s)
        stat = (s_sorted[-1]**2 + s_sorted[-2]**2) / np.sum(s_sorted ** 2)
        all_stats.append(stat)
    min_stat = np.min(all_stats)
    return min_stat


stats = np.zeros(len(datasets))
for ix, dataset in enumerate(datasets):
    stat = compute_statistics(dataset)
    stats[ix] = stat
in_model_stats = stats[:50]
out_model_stats = stats[50:]


thresholds = np.linspace(0.97, 0.999, 20)
false_positive_rates = np.array([np.mean(out_model_stats > thresh) for thresh in thresholds])
true_positive_rates = np.array([np.mean(in_model_stats > thresh) for thresh in thresholds])
sort_ixs = np.argsort(false_positive_rates)

plt.clf()
sns.set()
plt.style.use("style.mplstyle")
plt.plot(false_positive_rates[sort_ixs], true_positive_rates[sort_ixs], marker=".")
plt.plot([0, 1], [0, 1], color="gray", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()
plt.savefig("experiments/model_membership/rank_test_roc.png")