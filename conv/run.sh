#!/bin/bash

# Build the program
make

# List of kernels to test
kernels=("sharpen" "emboss" "avg3" "avg5" "avg7")

# Function to run with warmup and 3 actual runs
run_with_warmup_conv() {
    program=$1
    kernel=$2
    label=$3
    output_prefix=$4

    echo "[$label] Warmup run with kernel: $kernel"
    $program "$kernel" > /dev/null

    for i in {1..3}; do
        echo "[$label] Run $i with kernel: $kernel"
        $program "$kernel"
        echo "Output saved to data/${output_prefix}_${kernel}.pgm"
    done
    echo "--------------------------------------"
}

# Serial
for kernel in "${kernels[@]}"; do
    run_with_warmup_conv ./bin/conv_serial "$kernel" "Serial" "output_serial"
done

# Parallel Global Memory
for kernel in "${kernels[@]}"; do
    run_with_warmup_conv ./bin/conv_parallel_glb_mem "$kernel" "Parallel Global Memory" "output_parallel_glb_mem"
done

# Parallel Shared Memory
for kernel in "${kernels[@]}"; do
    run_with_warmup_conv ./bin/conv_parallel_shr_mem "$kernel" "Parallel Shared Memory" "output_parallel_shr_mem"
done
