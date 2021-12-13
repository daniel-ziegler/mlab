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
    int totalSize = BLOCKSIZE*gridDim.x;
    int idx = blockStart + threadIdx.x;
    float v = 0.0;
    for (int i = idx; i<size; i+=totalSize){
        v += A[i];
    }

    for (int stride = WARPSIZE/2; stride>0; stride/=2) {
        v += __shfl_down_sync(0xffffffff, v, stride);
    }

    const int SCRATCHSIZE = BLOCKSIZE / WARPSIZE;
    __shared__ float scratch[SCRATCHSIZE];
    if (threadIdx.x % WARPSIZE == 0) {
        scratch[threadIdx.x / WARPSIZE] = v;
    }
    __syncthreads();
    if (threadIdx.x < SCRATCHSIZE) {
        v = scratch[threadIdx.x];
        for (int stride = SCRATCHSIZE/2; stride>0; stride/=2) {
            v += __shfl_down_sync(0xffff, v, stride);
        }

        if (threadIdx.x == 0) {
            atomicAdd(dest, v);
        }
    }
}

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

float cuda_reduce_sum_tree(float* gpu_Dest, float* gpu_A, size_t size) {
    int blocks = min(ceil_div(size, BLOCKSIZE), 1024);
    reduce_sum_tree<<<blocks, BLOCKSIZE>>>(gpu_Dest, gpu_A, size);

    CUDA_ERROR_CHK(cudaPeekAtLastError());

    CUDA_ERROR_CHK(cudaDeviceSynchronize());

    float dest;
    CUDA_ERROR_CHK(cudaMemcpy(&dest, gpu_Dest, sizeof(float), cudaMemcpyDeviceToHost));
    return dest;
}

int main() {
    std::vector<size_t> sizes = {32, 200, 512, 65536, 100000, 1<<20, 2000000, 10000000, 100000000, 2<<29};
    for (auto size : sizes) {
        std::vector<float> A;
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<> dist(0.0, 1.0);
        for (size_t i = 0; i < size; i++) {
            A.push_back(dist(rng));
        }

        float *gpu_A;
        CUDA_ERROR_CHK(cudaMalloc(&gpu_A, A.size() * sizeof(float)));
        CUDA_ERROR_CHK(cudaMemcpy(gpu_A, A.data(), A.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));

        float *gpu_dest;
        CUDA_ERROR_CHK(cudaMalloc(&gpu_dest, sizeof(float)));
        CUDA_ERROR_CHK(cudaMemset(gpu_dest, 0, sizeof(float)));

        float sum = cuda_reduce_sum_tree(gpu_dest, gpu_A, A.size());
        std::cout << "GPU sum: " << sum << std::endl;

        if (size < 1<<23) {
            float cpu_sum = cpu_reduce_sum(A);
            std::cout << "CPU sum: " << cpu_sum << std::endl;
        }

        int n_iters = min(50, ceil_div(1<<30, size));
        Timer timer;
        float total = 0;
        for (int i = 0; i < n_iters; i++) {
            // cuda_reduce_sum_tree(gpu_dest, gpu_A, A.size());
            thrust_reduce_sum(gpu_A, A.size());

            CUDA_ERROR_CHK(cudaDeviceSynchronize());
        }
        timer.stop();
        // std::cerr << total << "\n";
        std::cout << size << "," << timer.elapsed() / (n_iters * size) << std::endl;
        std::cout << std::endl;

        CUDA_ERROR_CHK(cudaFree(gpu_A));
        CUDA_ERROR_CHK(cudaFree(gpu_dest));
    }
}
