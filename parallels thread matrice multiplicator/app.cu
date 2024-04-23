#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void matrixMulKernel(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

int main()
{
    int N = 3; // Matrices size
    float A[N*N] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B[N*N] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    float C[N*N] = {0};

    int size = N * N * sizeof(float);
    float *A_device, *B_device, *C_device;

    cudaMalloc((void**) &A_device, size);
    cudaMalloc((void**) &B_device, size);
    cudaMalloc((void**) &C_device, size);

    cudaMemcpy(A_device, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(1, 1);
    if (N*N > 512) {
        printf("Matrix too large for the GPU!\n");
        return 1;
    }

    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(A_device, B_device, C_device, N);

    cudaMemcpy(C, C_device, size, cudaMemcpyDeviceToHost);

            //print result

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", C[i*N + j]);
        }
        printf("\n");
    }

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

}