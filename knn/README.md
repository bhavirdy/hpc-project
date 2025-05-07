# K-Nearest Neighbors (KNN) Parallelization Project

This project implements the K-Nearest Neighbors algorithm with both serial and parallel versions using OpenMP.

## Directory Structure

- `src/`: Source files for the KNN algorithm.
- `bin/`: Executables.
- `data/`: Datasets for training/testing.

## Source Files

- `knn_serial.cpp`: Serial implementation of the KNN algorithm.
- `knn_parallel_sections.cpp`: Parallel version using OpenMP `sections`.
- `knn_parallel_tasks.cpp`: Parallel version using OpenMP `tasks`.

## Build Instructions

```bash
make
```

## Run Instructions
- Use the run.sh script for automated execution.
```bash
./run.sh
```