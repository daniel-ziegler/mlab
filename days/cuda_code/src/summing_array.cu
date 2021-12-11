#include <iostream>
#include <vector>
#include <random>
#include <utility>
#include <thrust/reduce.h>

#include "utils.h"

__global__ void reduce_sum_kernel(float* dest, float* A, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        atomicAdd(dest, A[tid]);
    }
}

float cuda_reduce_sum(float* gpu_A, size_t size) {
    float* gpu_dest;
    CUDA_ERROR_CHK(cudaMalloc(&gpu_dest, sizeof(float)));
    CUDA_ERROR_CHK(cudaMemset(gpu_dest, 0, sizeof(float)));

    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    reduce_sum_kernel<<<numBlocks, threadsPerBlock>>>(gpu_dest, gpu_A, size);

    CUDA_ERROR_CHK(cudaPeekAtLastError());
    CUDA_ERROR_CHK(cudaDeviceSynchronize());
    float dest;
    CUDA_ERROR_CHK(cudaMemcpy(&dest, gpu_dest, sizeof(float), cudaMemcpyDeviceToHost));
    return dest;
}

float thrust_reduce_sum(float* gpu_A, size_t size) {
    return thrust::reduce(thrust::device, gpu_A, gpu_A + size);
}

float cpu_reduce_sum(const std::vector<float>& A) {
    return std::accumulate(A.begin(), A.end(), 0.0);
}

constexpr int BLOCKSIZE = 512;

__global__ void reduce_sum_tree(float* dest, float* A, int size) {
    int readSize = 2 * BLOCKSIZE;
    int blockStart = blockIdx.x * readSize;
    int idx = blockStart + threadIdx.x;
    __shared__ float scratch[BLOCKSIZE];
    int stride = BLOCKSIZE;
    if (idx < size) {
        float acc = A[idx];
        if (idx + stride < size) {
            acc += A[idx + stride];
        }
        scratch[threadIdx.x] = acc;
    }
    __syncthreads();
    while (stride > 1) {
        stride /= 2;
        if (threadIdx.x < stride && idx + stride < size) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // write into dest
    if (threadIdx.x == 0) {
        dest[blockIdx.x] = scratch[0];
    }
}

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

float cuda_reduce_sum_tree(float* gpu_B, float* gpu_A, size_t size) {
    while (size > 1) {
        int readSize = 2*BLOCKSIZE;
        int numBlocks = ceil_div(size, readSize);
        reduce_sum_tree<<<numBlocks, BLOCKSIZE>>>(gpu_B, gpu_A, size);

        CUDA_ERROR_CHK(cudaPeekAtLastError());

        std::swap(gpu_A, gpu_B);
        size = ceil_div(size, readSize);
    }
    CUDA_ERROR_CHK(cudaDeviceSynchronize());

    float dest;
    CUDA_ERROR_CHK(cudaMemcpy(&dest, gpu_A, sizeof(float), cudaMemcpyDeviceToHost));
    return dest;
}

int main() {
    std::vector<size_t> sizes = {200, 300, 512, 2048, 301, 65536, 1<<20, 10000, 100000, 1000000, 2000000, 5000000, 10000000};
    for (auto size : sizes) {
        std::vector<float> A;
        std::mt19937_64 rng(42);
        std::normal_distribution<> normal_dist(0.0, 1);
        for (size_t i = 0; i < size; i++) {
            A.push_back(normal_dist(rng));
        }

        float *gpu_A;
        CUDA_ERROR_CHK(cudaMalloc(&gpu_A, A.size() * sizeof(float)));
        CUDA_ERROR_CHK(cudaMemcpy(gpu_A, A.data(), A.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));

        int B_size = ceil_div(A.size(), BLOCKSIZE);
        float *gpu_B;
        CUDA_ERROR_CHK(cudaMalloc(&gpu_B, B_size * sizeof(float)));

        float sum = cuda_reduce_sum_tree(gpu_B, gpu_A, A.size());
        std::cout << "GPU sum: " << sum << std::endl;

        float cpu_sum = cpu_reduce_sum(A);
        std::cout << "CPU sum: " << cpu_sum << std::endl;

        int n_iters = 50;
        Timer timer;
        float total = 0;
        for (int i = 0; i < n_iters; i++) {
            cuda_reduce_sum_tree(gpu_B, gpu_A, A.size());
            CUDA_ERROR_CHK(cudaDeviceSynchronize());
        }
        timer.stop();
        // std::cerr << total << "\n";
        std::cout << size << "," << timer.elapsed() / (n_iters * size) << std::endl;
        std::cout << std::endl;
    }
}
