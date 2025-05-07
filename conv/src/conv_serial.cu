#include <helper_functions.h>
#include <vector>
#include <chrono>

using namespace std;

void convolve(const unsigned char* input,
              unsigned char* output,
              int width,
              int height,
              const vector<vector<float>>& kernel)
{
    int kHeight = kernel.size();
    int kWidth = kernel[0].size();
    int a = kHeight / 2;
    int b = kWidth / 2;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0.0f;
            for (int l = -a; l <= a; ++l) {
                for (int p = -b; p <= b; ++p) {
                    int y = i + l;
                    int x = j + p;

                    int imageVal = 0;
                    if (y >= 0 && y < height && x >= 0 && x < width) {
                        imageVal = input[y * width + x];
                    }

                    float kernelVal = kernel[l + a][p + b];
                    sum += imageVal * kernelVal;
                }
            }
            output[i * width + j] = static_cast<unsigned char>(min(max(int(round(sum)), 0), 255));
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <kernel_name>" << endl;
        cerr << "Available kernels: sharpen, emboss, avg3, avg5, avg7" << endl;
        return 1;
    }

    const string kernel_name = argv[1];
    const char* inputFile = "./data/input.pgm";
    string outputFile = "./data/output_serial_" + kernel_name + ".pgm";

    vector<vector<float>> kernel;

    if (kernel_name == "sharpen") {
        kernel = {
            {-1, -1, -1},
            {-1,  9, -1},
            {-1, -1, -1}
        };
    } else if (kernel_name == "emboss") {
        kernel = {
            {1, 0, 0, 0, 0},
            {0, 1, 0, 0, 0},
            {0, 0, 0, 0, 0},
            {0, 0, 0, -1, 0},
            {0, 0, 0, 0, -1}
        };
    } else if (kernel_name == "avg3") {
        kernel = vector<vector<float>>(3, vector<float>(3, 1.0f / 9));
    } else if (kernel_name == "avg5") {
        kernel = vector<vector<float>>(5, vector<float>(5, 1.0f / 25));
    } else if (kernel_name == "avg7") {
        kernel = vector<vector<float>>(7, vector<float>(7, 1.0f / 49));
    } else {
        cerr << "Unknown kernel: " << kernel_name << endl;
        return 1;
    }

    unsigned char* image_flat = nullptr;
    unsigned int width, height;
    sdkLoadPGM(inputFile, &image_flat, &width, &height);
    unsigned char* output_flat = new unsigned char[width * height];

    auto start = chrono::high_resolution_clock::now();
    convolve(image_flat, output_flat, width, height, kernel);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    cout << "Serial Convolution Time: " << duration.count() << " ms\n";

    sdkSavePGM(outputFile.c_str(), output_flat, width, height);

    delete[] image_flat;
    delete[] output_flat;

    return 0;
}