#!/bin/bash

# Array of threshold values
thresholds=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Array of N values
Ns=(1 2 3 4 5 6 7 8 9 10)


# Iterate over threshold values
for threshold in "${thresholds[@]}"
do
    # Iterate over N values
    for N in "${Ns[@]}"
    do
            # Call the Python script with the current threshold, N, and stop_mechanism values
        python CS_based_early_stopping.py "$threshold" "$N"
    done
done
