#!/bin/bash

# Array of threshold values
thresholds=(0.4 0.5)

# Array of N values
Ns=(2 3 4)

# Iterate over threshold values
for threshold in "${thresholds[@]}"
do
    # Iterate over N values
    for N in "${Ns[@]}"
    do
        # Call the Python script with the current threshold and N values
        python CS_based_early_stopping.py "$threshold" "$N"
    done
done
