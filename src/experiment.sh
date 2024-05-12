#!/bin/bash

# Array of threshold values
thresholds=(0.5)

# Array of N values
Ns=(2 3 4 5 6 7 8 9 10)

# Array of stop_mechanism values
stop_mechanisms=("PositiveN")

# Iterate over threshold values
for threshold in "${thresholds[@]}"
do
    # Iterate over N values
    for N in "${Ns[@]}"
    do
        # Iterate over stop_mechanism values
        for stop_mechanism in "${stop_mechanisms[@]}"
        do
            # Call the Python script with the current threshold, N, and stop_mechanism values
            python CS_based_early_stopping.py "$threshold" "$N" "$stop_mechanism"
        done
    done
done
