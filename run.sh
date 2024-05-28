#!/bin/sh
#
# Run all script to generate the results presented in the paper "Power Control
# for Resilient Communication Systems With a Secret-Key Budget" (Karl-L.
# Besser, Rafael Schaefer, and Vincent Poor, IEEE International Symposium on
# Personal, Indoor and Mobile Radio Communications (PIMRC), Sep. 2024).
#
# Copyright (C) 2024 Karl-Ludwig Besser
# License: MIT

INIT_BUDGET=70
PROB_TX="0.35"
NUM_SAMP=1000000
NUM_TIME=250

for P_TX in {0..20..2}
do
python constant_power.py -b "$INIT_BUDGET" -v -p "$PROB_TX" --num_samples="$NUM_SAMP" -P "$P_TX" -T "$NUM_TIME" --export --plot
done
