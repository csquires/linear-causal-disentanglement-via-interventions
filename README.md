# Linear Causal Disentanglement via Interventions


## Setup 

To set up a virtual environment with the required packages, run:
```
bash setup.sh
```
Note that this project depends on two packages which can be difficult to install,
`pygraphviz` and `gurobipy`.
The dependence on `pygraphviz` is only necessary for plotting the learned latent DAG over the real data, so almost everything can be run without this dependency.
The dependence on `gurobipy` is required for efficient computation to see how well two permutations match, up to the partial order defined by a DAG.
This dependence can be removed by changing `DEFAULT_PERMUTATION_METHOD` in `src/run_experiment.py` from `"ilp"` to `"naive"`.

## Synthetic Data Result Reproduction
To reproduce the synthetic data results, run:
```
python3 -m experiments.experiment1.noisy_recovery --seed 8164 --nnodes 5 --nnodes_obs 10
```

## Real Data Result Reproduction
To reproduce the real data results, first download the data:
```
python3 -m experiments.real_data.step1_download_and_pickle
```
For the semi-synthetic results, run
```
bash experiments/real_data/semisynthetic.sh
```
For real-data results, run
```
bash experiments/real_data/real.sh
```