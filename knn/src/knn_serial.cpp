#include <cmath>  
#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <utility>
#include <omp.h>

using namespace std;

struct Input {
    vector<float> features;
};

struct TrainingDataPoint {
    vector<float> features;
    int label;
};

class KNN {
private:
    int _k;
    vector<TrainingDataPoint> _trainingData;
    
    double euclidean_distance(const Input& input, const TrainingDataPoint& trainingDataPoint) const {
        double sum = 0.0;
        for (size_t i = 0; i < input.features.size(); ++i) {
            double diff = input.features[i] - trainingDataPoint.features[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    };
    
    int partition(vector<pair<double, int>>& vec, int l, int h) {
        double pivot = vec[l].first;
        int i = l + 1;
        int j = h;
    
        while (i <= j) {
            while (i <= h && vec[i].first <= pivot) i++;
            while (j >= l && vec[j].first > pivot) j--;
            if (i < j) swap(vec[i], vec[j]);
        }
    
        if (j >= l) swap(vec[l], vec[j]);
        return j;
    }
    
    void serial_quicksort(vector<pair<double, int>>& vec, int l, int h) {
        if (l < h) {
            int j = partition(vec, l, h);
            serial_quicksort(vec, l, j - 1);
            serial_quicksort(vec, j + 1, h);
        }
    }

public:
    KNN(int k) : _k(k) {};
    void train(const vector<TrainingDataPoint>& trainingData) {
        _trainingData = trainingData;
    };

    int predict(const Input& input, double& distance_time, double& sort_time) {
        // 1. Calculate all distances
        double start_time = omp_get_wtime();
        vector<pair<double, int>> distances; // pair of (distance, label)
        for (const auto& trainingDataPoint : _trainingData) {
            double dist = euclidean_distance(input, trainingDataPoint);
            distances.push_back({dist, trainingDataPoint.label});
        }
        double end_time = omp_get_wtime();
        distance_time = (end_time - start_time);

        // 2. Sort distances
        start_time = omp_get_wtime();
        serial_quicksort(distances, 0, distances.size() - 1);
        end_time = omp_get_wtime();
        sort_time = (end_time - start_time);

        // 3. Vote among k nearest neighbors
        map<int, int> labelCounts;
        for (int i = 0; i < _k && i < static_cast<int>(distances.size()); ++i) {
            int label = distances[i].second;
            labelCounts[label]++;
        }

        // 4. Find label with maximum count
        int bestLabel = -1;
        int maxCount = -1;
        for (const auto& entry : labelCounts) {
            if (entry.second > maxCount) {
                maxCount = entry.second;
                bestLabel = entry.first;
            }
        }

        return bestLabel;
    };
};

// Reads a flattened (NUM_SAMPLES x FEATURE_DIM) binary file of floats
vector<vector<float>> read_features(const string &filename, size_t num_samples, size_t feature_dim) {
    size_t total_elements = num_samples * feature_dim;
    vector<float> flat(total_elements);
    
    ifstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    file.read(reinterpret_cast<char*>(flat.data()), total_elements * sizeof(float));
    if (!file)
        throw runtime_error("Error reading file: " + filename);
    file.close();
    
    // Reshape the flat vector into a vector of vectors.
    vector<vector<float>> features;
    features.reserve(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        vector<float> sample(flat.begin() + i * feature_dim, flat.begin() + (i + 1) * feature_dim);
        features.push_back(move(sample));
    }
    
    return features;
}

// Reads a binary file containing labels.
vector<int> read_labels(const string &filename, size_t num_samples) {
    vector<int> labels(num_samples);
    
    ifstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    file.read(reinterpret_cast<char*>(labels.data()), num_samples * sizeof(int));
    if (!file)
        throw runtime_error("Error reading file: " + filename);
    file.close();
    
    return labels;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <K>" << endl;
        return 1;
    }

    int K;
    K = stoi(argv[1]);

    constexpr size_t TRAIN_NUM_SAMPLES = 50000;
    constexpr size_t TEST_NUM_SAMPLES = 10000;
    constexpr size_t FEATURE_DIM = 512;
    
    vector<vector<float>> features = read_features("./data/train/train_features.bin", TRAIN_NUM_SAMPLES, FEATURE_DIM);
    vector<int> labels = read_labels("./data/train/train_labels.bin", TRAIN_NUM_SAMPLES);

    vector<TrainingDataPoint> trainingData;
    for (size_t i = 0; i < features.size(); ++i) {
        trainingData.push_back({features[i], labels[i]});
    }

    KNN knn(K);
    knn.train(trainingData);

    vector<vector<float>> inputs = read_features("./data/test/test_features.bin", TEST_NUM_SAMPLES, FEATURE_DIM);
    vector<int> expected_labels = read_labels("./data/test/test_labels.bin", TEST_NUM_SAMPLES);

    size_t correct = 0;
    double total_distance_time = 0.0, total_sort_time = 0.0;
    double total_runtime_start = omp_get_wtime();

    for (size_t i = 0; i < inputs.size(); ++i) {
        Input input{inputs[i]};
        double distance_time, sort_time;

        int predicted = knn.predict(input, distance_time, sort_time);
        total_distance_time += distance_time;
        total_sort_time += sort_time;

        if (predicted == expected_labels[i]) {
            ++correct;
        }

        if (i % 1000 == 0) {
            cout << "Predicted " << i << " / " << TEST_NUM_SAMPLES << " samples..." << endl;
        }
    }
    
    double total_runtime_end = omp_get_wtime();
    double total_runtime = total_runtime_end - total_runtime_start;

    double accuracy = static_cast<double>(correct) / inputs.size();
    cout << "Accuracy: " << (accuracy * 100.0) << "%" << endl;
    cout << "Total Distance Time: " << total_distance_time << " seconds" << endl;
    cout << "Total Sort Time: " << total_sort_time << " seconds" << endl;
    cout << "Total Runtime: " << total_runtime << " seconds" << endl;
    
    return 0;
}