#include <iostream>
#include <vector>
#include <random>
#include <thrust/reduce.h>

#include "utils.h"

__global__ void reduce_sum_kernel(float* dest, float* A, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        atomicAdd(dest, A[tid]);
    }
}

float cuda_reduce_sum(float* gpu_A, size_t size) {
    float *gpu_dest;
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

int main() {
    std::vector<size_t> sizes = {10000, 100000, 1000000, 2000000, 5000000, 10000000, 100000000};
    for (auto size : sizes) {
        std::vector<float> A;
        std::mt19937_64 rng(42);
        std::normal_distribution<> normal_dist(0.0, 1);
        for (size_t i = 0; i < size; i++) {
            A.push_back(normal_dist(rng));
        }

        float cpu_sum = cpu_reduce_sum(A);

        float *gpu_A;
        CUDA_ERROR_CHK(cudaMalloc(&gpu_A, A.size() * sizeof(float)));
        CUDA_ERROR_CHK(cudaMemcpy(gpu_A, A.data(), A.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));


        float sum = cuda_reduce_sum(gpu_A, A.size());

        int n_iters = 50;
        Timer timer;
        float total = 0;
        for (int i = 0; i < n_iters; i++) {
            total += cuda_reduce_sum(gpu_A, A.size());
        }
        timer.stop();
        std::cerr << total << "\n";
        std::cout << size << "," << timer.elapsed() / (n_iters * size) << std::endl;
    }
}
