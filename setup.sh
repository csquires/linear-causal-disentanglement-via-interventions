#!/usr/bin/bash

python -m venv venv
source venv/bin/activate

pip3 install numpy
pip3 install causaldag
pip3 install xgboost pgmpy  # causaldag requirements
pip3 install seaborn
pip3 install lifelines
pip3 install statsmodels