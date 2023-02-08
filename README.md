# Linear Causal Disentanglement via Interventions

To set up a virtual environment with the required packages, run:
```
bash setup.sh
```

To reproduce the synthetic data results, run:
```
python3 -m experiments.experiment1.noisy_recovery --seed 8164 --nnodes 5 --nnodes_obs 10
```

To reproduce the real data results, first download the data:
```
python3 -m experiments.real_data.step1_download_and_pickle
```
For the semi-synthetic results, run
```
bash experiments/real_data/semisynthetic.sh
```