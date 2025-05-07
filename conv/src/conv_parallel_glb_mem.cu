#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <chrono>

using namespace std;

// CUDA Kernel
__global__
void convolveKernel(const unsigned char* input,
                    unsigned char* output,
                    int width, int height,
                    const float* kernel,
                    int kWidth, int kHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfKW = kWidth / 2;
    int halfKH = kHeight / 2;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    for (int ky = -halfKH; ky <= halfKH; ++ky) {
        for (int kx = -halfKW; kx <= halfKW; ++kx) {
            int ix = x + kx;
            int iy = y + ky;

            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                float imageVal = input[iy * width + ix];
                float kernelVal = kernel[(ky + halfKH) * kWidth + (kx + halfKW)];
                sum += imageVal * kernelVal;
            }
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
    string outputFile = "./data/output_parallel_glb_mem_" + kernel_name + ".pgm";

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

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Running on GPU: " << prop.name << endl;

    auto start = chrono::high_resolution_clock::now();
    convolveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height, d_kernel, kWidth, kHeight);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    cout << "CUDA Convolution (Global Memory) Time: " << duration.count() << " ms\n";

    // Copy result back
    checkCudaErrors(cudaMemcpy(output_flat, d_output, width * height, cudaMemcpyDeviceToHost));

    // Save output image
    sdkSavePGM(outputFile.c_str(), output_flat, width, height);

    // Free memory
    delete[] image_flat;
    delete[] output_flat;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}
