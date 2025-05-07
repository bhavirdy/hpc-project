#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <chrono>

using namespace std;

__global__
void convolveKernelShared(const unsigned char* input,
                          unsigned char* output,
                          int width, int height,
                          const float* kernel,
                          int kWidth, int kHeight)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    int halfKW = kWidth / 2;
    int halfKH = kHeight / 2;

    // Shared memory tile dimensions
    int tileWidth = blockDim.x + kWidth - 1;
    int tileHeight = blockDim.y + kHeight - 1;

    extern __shared__ unsigned char shared[];

    int sharedX = tx + halfKW;
    int sharedY = ty + halfKH;

    // Load image into shared memory (with halo)
    for (int dy = ty; dy < tileHeight; dy += blockDim.y) {
        for (int dx = tx; dx < tileWidth; dx += blockDim.x) {
            int globalX = blockIdx.x * blockDim.x + dx - halfKW;
            int globalY = blockIdx.y * blockDim.y + dy - halfKH;

            globalX = min(max(globalX, 0), width - 1);
            globalY = min(max(globalY, 0), height - 1);

            shared[dy * tileWidth + dx] = input[globalY * width + globalX];
        }
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    for (int ky = 0; ky < kHeight; ++ky) {
        for (int kx = 0; kx < kWidth; ++kx) {
            int sX = sharedX + kx - halfKW;
            int sY = sharedY + ky - halfKH;
            float imageVal = shared[sY * tileWidth + sX];
            float kernelVal = kernel[ky * kWidth + kx];
            sum += imageVal * kernelVal;
        }
    }

    output[y * width + x] = static_cast<unsigned char>(min(max(int(roundf(sum)), 0), 255));
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <kernel_name>" << endl;
        cerr << "Available kernels: sharpen, emboss, avg3, avg5, avg7" << endl;
        return 1;
    }

    const string kernel_name = argv[1];
    const char* inputFile = "./data/input.pgm";
    string outputFile = "./data/output_parallel_shr_mem_" + kernel_name + ".pgm";

    vector<float> kernel;
    int kWidth = 0, kHeight = 0;

    if (kernel_name == "sharpen") {
        kernel = {
            -1, -1, -1,
            -1,  9, -1,
            -1, -1, -1
        };
        kWidth = kHeight = 3;
    } else if (kernel_name == "emboss") {
        kernel = {
            1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, -1, 0,
            0, 0, 0, 0, -1
        };
        kWidth = kHeight = 5;
    } else if (kernel_name == "avg3") {
        kernel = vector<float>(9, 1.0f / 9.0f);
        kWidth = kHeight = 3;
    } else if (kernel_name == "avg5") {
        kernel = vector<float>(25, 1.0f / 25.0f);
        kWidth = kHeight = 5;
    } else if (kernel_name == "avg7") {
        kernel = vector<float>(49, 1.0f / 49.0f);
        kWidth = kHeight = 7;
    } else {
        cerr << "Unknown kernel name: " << kernel_name << endl;
        return 1;
    }

    unsigned char* image_flat = nullptr;
    unsigned int width, height;
    sdkLoadPGM(inputFile, &image_flat, &width, &height);
    unsigned char* output_flat = new unsigned char[width * height];

    // Allocate device memory
    unsigned char *d_input, *d_output;
    float *d_kernel;
    checkCudaErrors(cudaMalloc(&d_input, width * height));
    checkCudaErrors(cudaMalloc(&d_output, width * height));
    checkCudaErrors(cudaMalloc(&d_kernel, kernel.size() * sizeof(float)));

    // Copy data to device
    checkCudaErrors(cudaMemcpy(d_input, image_flat, width * height, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch setup
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Running on GPU: " << prop.name << endl;

    // Calculate shared memory size
    int sharedMemSize = (threadsPerBlock.x + kWidth - 1) *
                        (threadsPerBlock.y + kHeight - 1) *
                        sizeof(unsigned char);

    // Launch kernel and time it
    auto start = chrono::high_resolution_clock::now();
    convolveKernelShared<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_input, d_output, width, height, d_kernel, kWidth, kHeight);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    cout << "CUDA Convolution (Shared Memory) Time: " << duration.count() << " ms\n";

    // Copy result back
    checkCudaErrors(cudaMemcpy(output_flat, d_output, width * height, cudaMemcpyDeviceToHost));

    // Save result
    sdkSavePGM(outputFile.c_str(), output_flat, width, height);

    // Free memory
    delete[] image_flat;
    delete[] output_flat;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}
