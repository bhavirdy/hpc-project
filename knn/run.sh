#!/bin/bash

make

# Define values of k
ks=(3 5 7)

# Function to run the program with warmup and 3 actual runs
run_with_warmup() {
    program=$1
    k=$2
    label=$3

    echo "[$label] Warmup run with k = $k"
    $program "$k" > /dev/null

    for i in {1..3}; do
        echo "[$label] Run $i with k = $k"
        $program "$k"
    done
    echo "--------------------------------------"
}

# Serial
for k in "${ks[@]}"; do
    run_with_warmup ./bin/knn_serial "$k" "Serial"
done

# Parallel Sections
for k in "${ks[@]}"; do
    run_with_warmup ./bin/knn_parallel_sections "$k" "Parallel Sections"
done

# Parallel Tasks
for k in "${ks[@]}"; do
    run_with_warmup ./bin/knn_parallel_tasks "$k" "Parallel Tasks"
done
