#!/usr/bin/bash

python3 -m experiments.real_data.step2_preprocessing --num_obs 100
python3 -m experiments.real_data.step3_run_algorithm --num_obs 100 --num_contexts 10
python3 -m experiments.real_data.step4_semisynthetic_analysis --num_obs 100 --num_contexts 10