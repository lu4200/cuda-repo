#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void mykernel(float *A1, float *A2, float *R)
{
    int p = threadIdx.x;
    R[p] = A1[p] + A2[p];
}
 
int main()
{
    float A1[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    float A2[] = { 10, 20, 30, 40, 50, 60, 70, 80, 90 };
    float R[9];
    int taille_mem = sizeof(float) * 9;
    float *a1_device;
    float *a2_device;
    float *r_device;

    cudaMalloc((void**) &a1_device, taille_mem);
    cudaMalloc((void**) &a2_device, taille_mem);
    cudaMalloc((void**) &r_device, taille_mem);

    cudaMemcpy(a1_device, A1, taille_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(a2_device, A2, taille_mem, cudaMemcpyHostToDevice);
        
    mykernel<<<1, 9>>>(a1_device, a2_device, r_device);
 	
    cudaMemcpy(R, r_device, taille_mem, cudaMemcpyDeviceToHost);
    //output
    for(int i = 0; i < 9; i++) {
        printf("%f\n", R[i]);
    }
}
