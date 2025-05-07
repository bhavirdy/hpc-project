# CUDA Convolution Project

This project demonstrates different implementations of 2D convolution using CUDA. It includes both serial and parallel versions utilising global and shared memory.

## Directory Structure

- `src/`: Source files containing CUDA implementations.
- `bin/`: Compiled binaries.
- `common/`: Shared utilities or headers.
- `data/`: Input data or test images.

## Source Files

- `conv_serial.cu`: Baseline serial implementation of 2D convolution.
- `conv_parallel_glb_mem.cu`: CUDA parallel implementation using global memory.
- `conv_parallel_shr_mem.cu`: CUDA parallel implementation using shared memory.

### Run Instructions
- Use the run.sh script for automated execution.
```bash
./run.sh
```

### Notes
- Ensure you have a CUDA-compatible GPU.