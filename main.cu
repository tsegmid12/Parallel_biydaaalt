#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <cuda_runtime.h>

void insertionSort(int* arr, int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void openmp(int* arr, int n) {
    #pragma omp parallel for
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void thread_worker(int* arr, int start, int end) {
    for (int i = start + 1; i < end; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= start && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void thread_sort(int* arr, int n, int num_threads) {
    int block = n / num_threads;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; t++) {
        int start = t * block;
        int end = (t == num_threads - 1) ? n : start + block;
        threads.emplace_back(thread_worker, arr, start, end);
    }
    for (auto& th : threads) th.join();
}

__global__ void insertionSortKernel(int* arr, int n, int chunkSize) {
    int tid = blockIdx.x;

    int start = tid * chunkSize;
    int end = start + chunkSize;

    if (end > n) end = n;

    // local insertion sort
    for (int i = start + 1; i < end; i++) {
        int key = arr[i];
        int j = i - 1;

        while (j >= start && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

#include <cuda_runtime.h>

void cuda_sort(int* h_arr, int n) {
    int* d_arr;

    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1;
    int chunkSize = 1024;
    int blocks = (n + chunkSize - 1) / chunkSize;

    insertionSortKernel<<<blocks, threadsPerBlock>>>(d_arr, n, chunkSize);

    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}

int main() {
    int sizes[] = {10000, 100000, 1000000}; 
    int num_sizes = 3;
    int num_threads = std::thread::hardware_concurrency();    

    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        int* arr = (int*)malloc(N * sizeof(int));

        srand(42);
        for (int i = 0; i < N; i++)
            arr[i] = rand() % 1000000;

        auto seq_start = std::chrono::steady_clock::now();

        insertionSort(arr, N);

        auto seq_end = std::chrono::steady_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(seq_end - seq_start).count();

        printf("Sequential Insertion Sort | N = %7d | Time: %6.2f ms\n", N, ms);

        int ok = 1;
        for (int i = 1; i < N; i++)
            if (arr[i] < arr[i-1]) { ok = 0; break; }
        printf("  Sorted correctly: %s\n", ok ? "YES" : "NO");

        auto omp_start = std::chrono::steady_clock::now();
        openmp(arr, N);
        auto omp_end = std::chrono::steady_clock::now();
        ms = std::chrono::duration<double, std::milli>(omp_end - omp_start).count();
        printf("OpenMP Insertion Sort   | N = %7d | Time: %6.2f ms\n", N, ms);

        ok = 1;
        for (int i = 1; i < N; i++)            if (arr[i] < arr[i-1]) { ok = 0; break; }
        printf("  Sorted correctly: %s\n", ok ? "YES" : "NO");  

        auto thread_start = std::chrono::steady_clock::now();
        thread_sort(arr, N, num_threads);
        auto thread_end = std::chrono::steady_clock::now();
        ms = std::chrono::duration<double, std::milli>(thread_end - thread_start).count();
        printf("Threaded Insertion Sort | N = %7d | Time: %6.2f ms\n", N, ms);
        ok = 1;
        for (int i = 1; i < N; i++)
            if (arr[i] < arr[i-1]) { ok = 0; break; }
        printf("  Sorted correctly: %s\n", ok ? "YES" : "NO");

        auto cuda_start = std::chrono::steady_clock::now();
        cuda_sort(arr, N);
        auto cuda_end = std::chrono::steady_clock::now();
        ms = std::chrono::duration<double, std::milli>(cuda_end - cuda_start).count();
        printf("CUDA Insertion Sort     | N = %7d | Time: %6.2f ms\n", N, ms);  
        ok = 1;
        for (int i = 1; i < N; i++)            
        if (arr[i] < arr[i-1]) { ok = 0; break; }
        printf("  Sorted correctly: %s\n", ok ? "YES" : "NO");
        free(arr);
            
    }
    return 0;
}