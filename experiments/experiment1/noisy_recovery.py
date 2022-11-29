# === IMPORTS: BUILT-IN ===
from argparse import ArgumentParser, BooleanOptionalAction

# === IMPORTS: LOCAL ===
from experiments.run_experiment import ExperimentRunner

# === ARGUMENT PARSING ===
parser = ArgumentParser()
parser.add_argument("--nnodes", type=int, default=5)
parser.add_argument("--nsamples_list", type=int, nargs="+", default=[2500, 5000, 10000, 25000, 50000, 100000])
parser.add_argument("--nruns", type=int, default=500)
parser.add_argument("--density", type=float, default=0.75)
parser.add_argument("--rank_gamma", type=float, default=0.99)
parser.add_argument("--seed", type=int, default=689729)
parser.add_argument("--overwrite", type=BooleanOptionalAction, default=False)
args = parser.parse_args()
# ===============================================


er = ExperimentRunner(
    args.nnodes,
    args.nsamples_list,
    args.nruns,
    args.seed,
    args.density,
    args.rank_gamma
)
er.run()
er.plot()
print(er.seed)


