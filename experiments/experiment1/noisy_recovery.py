# === IMPORTS: BUILT-IN ===
from argparse import ArgumentParser, BooleanOptionalAction

# === IMPORTS: LOCAL ===
from src.run_experiment import ExperimentRunner2

# === ARGUMENT PARSING ===
parser = ArgumentParser()
parser.add_argument("--nnodes", type=int, default=5)
parser.add_argument("--nnodes_obs", type=int, default=10)
parser.add_argument("--nsamples_list", type=int, nargs="+", default=[
    2500, 
    5000, 
    10000, 
    25000, 
    50000, 
    100000,
    250000
])
parser.add_argument("--nruns", type=int, default=100)
parser.add_argument("--density", type=float, default=0.75)
parser.add_argument("--rank_gamma", type=float, default=0.99)
parser.add_argument("--seed", type=int, default=8164)
parser.add_argument("--iv_type", type=str, default="hard")
parser.add_argument("--overwrite", type=BooleanOptionalAction, default=False)
args = parser.parse_args()
# ===============================================

er = ExperimentRunner2(
    args.nnodes,
    args.nsamples_list,
    args.nruns,
    args.seed,
    args.density,
    args.rank_gamma,
    nnodes_obs=args.nnodes_obs,
    iv_type=args.iv_type,
    find_best_permutation="ilp"
)
er.run(overwrite=args.overwrite)
er.plot()
print(er.seed)


