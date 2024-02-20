#!/bin/sh
#

INIT_BUDGET=70
PROB_TX="0.35"
NUM_SAMP=1000000
NUM_TIME=250

for P_TX in {0..20..2}
do
python constant_power.py -b "$INIT_BUDGET" -v -p "$PROB_TX" --num_samples="$NUM_SAMP" -P "$P_TX" -T "$NUM_TIME" --export --plot
done
