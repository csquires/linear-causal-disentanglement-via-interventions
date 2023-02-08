#!/usr/bin/bash

python -m venv venv
source venv/bin/activate

# ALGORITHMS
pip3 install causaldag
pip3 install gurobipy

# SURVIVAL ANALYSIS
pip3 install lifelines
pip3 install statsmodels

# PLOTTING
pip3 install seaborn
pip3 install --global-option=build_ext --global-option="-I/usr/local/include/" --global-option="-L/usr/local/lib/" pygraphviz  # required for plotting the real-data graph