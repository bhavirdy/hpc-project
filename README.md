This repository contains two main subprojects demonstrating parallel computing concepts:

1. **Convolution (CUDA)** - under `conv/`
2. **K-Nearest Neighbors (OpenMP)** - under `knn/`

Each subproject includes:
- Serial and parallel implementations
- Makefile for compilation
- `run.sh` for simplified execution

## Requirements

- CUDA-capable GPU for `conv/`

## Getting Started

```bash
cd conv
make
./run.sh

cd ../knn
make
./run.sh
