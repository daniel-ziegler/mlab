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
constexpr int WARPSIZE = 32;

__global__ void reduce_sum_tree(float* dest, float* A, int size) {
    int blockStart = blockIdx.x * BLOCKSIZE;
    int idx = blockStart + threadIdx.x;
    
    float v = idx < size ? A[idx] : 0.0;

    int stride = WARPSIZE;
    while (stride > 1) {
        stride /= 2;
        v += __shfl_down_sync(0xffffffff, v, stride);
    }

    const int SCRATCHSIZE = BLOCKSIZE / WARPSIZE;
    __shared__ float scratch[SCRATCHSIZE];
    if (threadIdx.x % WARPSIZE == 0) {
        scratch[threadIdx.x / WARPSIZE] = v;
    }
    __syncthreads();
    if (threadIdx.x < WARPSIZE) {
        v = scratch[threadIdx.x];
        int stride = SCRATCHSIZE;
        while (stride > 1) {
            stride /= 2;
            v += __shfl_down_sync(0xffffffff, v, stride);
        }

        if (threadIdx.x == 0) {
            dest[blockIdx.x] = v;
        }
    }
}

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

float cuda_reduce_sum_tree(float* gpu_B, float* gpu_A, size_t size) {
    while (size > 1) {
        int numBlocks = ceil_div(size, BLOCKSIZE);
        reduce_sum_tree<<<numBlocks, BLOCKSIZE>>>(gpu_B, gpu_A, size);

        CUDA_ERROR_CHK(cudaPeekAtLastError());

        std::swap(gpu_A, gpu_B);
        size = ceil_div(size, BLOCKSIZE);
    }
    CUDA_ERROR_CHK(cudaDeviceSynchronize());

    float dest;
    CUDA_ERROR_CHK(cudaMemcpy(&dest, gpu_A, sizeof(float), cudaMemcpyDeviceToHost));
    return dest;
}

int main() {
    std::vector<size_t> sizes = {32, 200, 512, 65536, 1<<20, 2000000, 5000000, 10000000, 100000000};
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
