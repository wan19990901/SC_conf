#!/bin/bash

# Array of threshold values
thresholds=(0.3 0.4 0.5 0.6 0.7)

# Array of N values
Ns=(2 3 4 5)

# Iterate over threshold values
for threshold in "${thresholds[@]}"
do
    # Iterate over N values
    for N in "${Ns[@]}"
    do
        # Call the Python script with the current threshold and N values
        python your_script.py "$threshold" "$N"
    done
done