#!/usr/bin/bash

python3 -m experiments.real_data.step2_preprocessing --num_obs 83
python3 -m experiments.real_data.step3_run_algorithm --num_obs 83 --num_contexts 83
python3 -m experiments.real_data.step5_compare_tcga --num_obs 83 --num_contexts 83
python3 -m experiments.real_data.step6_plot_graph --num_obs 83 --num_contexts 83