#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int matrix_size = 512;


__global__ void matmul(const float* a, const float* b, float* c) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= matrix_size || col >= matrix_size) return;

  float sum = 0.0f;
  for (int k = 0; k < matrix_size; ++k) {
    sum += a[row * matrix_size + k] * b[k * matrix_size + col];
  }
  c[row * matrix_size + col] = sum;
}

int main(int argc, char** argv) {
  const int kernel_count = argc > 1 ? std::atoi(argv[1]) : 2000;
  if (kernel_count <= 0) {
    std::fprintf(stderr, "usage: %s [kernel_count=2000]\n", argv[0]);
    return EXIT_FAILURE;
  }

  constexpr size_t matrix_bytes = matrix_size * matrix_size * sizeof(float);
  std::vector<float> host_matrix(matrix_size * matrix_size, 1.0f);
  float *a = nullptr, *b = nullptr, *c = nullptr;
  cudaMalloc(&a, matrix_bytes);
  cudaMalloc(&b, matrix_bytes);
  cudaMalloc(&c, matrix_bytes);
  cudaMemcpy(a, host_matrix.data(), matrix_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(b, host_matrix.data(), matrix_bytes, cudaMemcpyHostToDevice);

  const dim3 threads(16, 16);
  const dim3 blocks((matrix_size + threads.x - 1) / threads.x,
                    (matrix_size + threads.y - 1) / threads.y);

  // Warm up before timing the batch.
  matmul<<<blocks, threads>>>(a, b, c);
  cudaDeviceSynchronize();

  const auto submit_start = std::chrono::steady_clock::now();
  for (int i = 0; i < kernel_count; ++i) {
    // use the default stream
    matmul<<<blocks, threads>>>(a, b, c);
  }
  const auto submit_end = std::chrono::steady_clock::now();
  cudaDeviceSynchronize();
  const auto finish = std::chrono::steady_clock::now();

  float result = 0.0f;
  cudaMemcpy(&result, c, sizeof(result), cudaMemcpyDeviceToHost);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  const auto submit_ms = std::chrono::duration<double, std::milli>(submit_end - submit_start).count();
  const auto total_ms = std::chrono::duration<double, std::milli>(finish - submit_start).count();
  std::printf("Queued %d naive %dx%d matmul kernels in the default stream.\n",
              kernel_count, matrix_size, matrix_size);
  std::printf("Host submission: %.3f ms; submission through completion: %.3f ms.\n", submit_ms, total_ms);
  std::printf("c[0] = %.1f (expected %d)\n", result, matrix_size);
}
