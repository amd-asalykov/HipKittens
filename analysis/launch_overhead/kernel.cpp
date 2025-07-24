#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include "pyutils/pyutils.cuh"
#include "kittens.cuh"

using namespace kittens;

__global__ void emptyKernel() {
    // Empty kernel - does nothing
    // This allows us to measure pure launch overhead
}

__global__ void delayKernel() {
    // Simulate 10 microseconds of work
    // Use clock64() for timing on AMD GPUs
    long long start = clock64();
    long long clock_rate = 1000000; // Approximate clock rate in kHz (1 GHz)
    long long cycles_to_wait = (10 * clock_rate) / 1000; // 10 microseconds worth of cycles
    
    while ((clock64() - start) < cycles_to_wait) {
        // Busy wait
    }
}

void checkHipError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "HIP Error: " << msg << " - " << hipGetErrorString(err) << std::endl;
        exit(1);
    }
}

double measureRegularLaunch(int numLaunches) {
    // Typical launch configuration
    dim3 gridDim(256);   // 256 blocks
    dim3 blockDim(256);  // 256 threads per block = 65536 total threads
    
    // Warmup
    for (int i = 0; i < 100; i++) {
        emptyKernel<<<gridDim, blockDim>>>();
    }
    checkHipError(hipDeviceSynchronize(), "Warmup sync");

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numLaunches; i++) {
        emptyKernel<<<gridDim, blockDim>>>();
    }
    checkHipError(hipDeviceSynchronize(), "Regular launch sync");
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    
    return duration.count() / numLaunches; // Average time per launch in microseconds
}

double measureGraphLaunch(int numLaunches) {
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipStream_t stream;
    
    // Create stream for graph capture
    checkHipError(hipStreamCreate(&stream), "Stream creation");
    
    // Typical launch configuration
    dim3 gridDim(256);   // 256 blocks
    dim3 blockDim(256);  // 256 threads per block
    
    // Start graph capture
    checkHipError(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), "Begin capture");
    
    // Launch kernels into the stream during capture
    for (int i = 0; i < numLaunches; i++) {
        emptyKernel<<<gridDim, blockDim, 0, stream>>>();
    }
    
    // End capture and create graph
    checkHipError(hipStreamEndCapture(stream, &graph), "End capture");
    
    // Instantiate graph
    checkHipError(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), 
                  "Graph instantiation");
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        checkHipError(hipGraphLaunch(graphExec, stream), "Graph warmup launch");
        checkHipError(hipStreamSynchronize(stream), "Graph warmup sync");
    }
    
    // Measure single graph execution containing all kernels
    auto start = std::chrono::high_resolution_clock::now();
    
    checkHipError(hipGraphLaunch(graphExec, stream), "Graph launch");
    checkHipError(hipStreamSynchronize(stream), "Graph launch sync");
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    
    // Cleanup
    checkHipError(hipGraphExecDestroy(graphExec), "Graph exec destroy");
    checkHipError(hipGraphDestroy(graph), "Graph destroy");
    checkHipError(hipStreamDestroy(stream), "Stream destroy");
    
    return duration.count() / numLaunches; // Average time per kernel in the graph
}

void printStats(const std::string& name, const std::vector<double>& times) {
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    
    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    
    double median = sorted_times[sorted_times.size() / 2];
    double min = sorted_times.front();
    double max = sorted_times.back();
    
    double sq_sum = 0;
    for (auto t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double stddev = std::sqrt(sq_sum / times.size());
    
    std::cout << "\n" << name << " Statistics (microseconds):" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Mean:   " << mean << " μs" << std::endl;
    std::cout << "  Median: " << median << " μs" << std::endl;
    std::cout << "  Min:    " << min << " μs" << std::endl;
    std::cout << "  Max:    " << max << " μs" << std::endl;
    std::cout << "  StdDev: " << stddev << " μs" << std::endl;
}

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> in;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(64); } 
    size_t dynamic_shared_memory() { return 0; }
};

void dispatch_micro(micro_globals g) {
    // Initialize HIP
    checkHipError(hipInit(0), "HIP initialization");
    
    // Get device properties
    hipDeviceProp_t props;
    checkHipError(hipGetDeviceProperties(&props, 0), "Get device properties");
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    
    const int numRuns = 10;
    const int kernelsPerGraph = 1000;  // Number of kernels in each graph
    
    std::vector<double> regularTimes;
    std::vector<double> graphTimes;
    
    std::cout << "\nBenchmark configuration:" << std::endl;
    std::cout << "- Kernels per measurement: " << kernelsPerGraph << std::endl;
    std::cout << "- Grid dimensions: 256 blocks" << std::endl;
    std::cout << "- Block dimensions: 256 threads (65,536 total threads)" << std::endl;
    std::cout << "- Number of runs: " << numRuns << std::endl;
    std::cout << "\nRunning benchmark..." << std::endl;
    
    // Run multiple times to get statistics
    for (int run = 0; run < numRuns; run++) {
        std::cout << "Run " << (run + 1) << "/" << numRuns << "..." << std::endl;
        
        // Measure regular kernel launches
        double regularTime = measureRegularLaunch(kernelsPerGraph);
        regularTimes.push_back(regularTime);
        
        // Measure graph with same number of kernels
        double graphTime = measureGraphLaunch(kernelsPerGraph);
        graphTimes.push_back(graphTime);
    }
    
    // Print results
    printStats("Regular Kernel Launch", regularTimes);
    printStats("HIP Graph Launch", graphTimes);
    
    // Calculate speedup
    double avgRegular = std::accumulate(regularTimes.begin(), regularTimes.end(), 0.0) / regularTimes.size();
    double avgGraph = std::accumulate(graphTimes.begin(), graphTimes.end(), 0.0) / graphTimes.size();
    double speedup = avgRegular / avgGraph;
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Average Regular Launch: " << avgRegular << " μs" << std::endl;
    std::cout << "Average Graph Launch:   " << avgGraph << " μs" << std::endl;
    std::cout << "Graph Speedup:          " << speedup << "x" << std::endl;
    std::cout << "Overhead Reduction:     " << (1.0 - avgGraph/avgRegular) * 100 << "%" << std::endl;
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::in);
}