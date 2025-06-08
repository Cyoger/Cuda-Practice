#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <cuda_runtime.h>

/*
Kernel function to add vectors.
__global__ is used to mark function as a kernel (callable from CPU, executed on GPU)
 */
__global__ void vecAddKernel(float *A, float *B, float *C, int n){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < n){
		C[i] = A[i] + B[i];
	}
}

void vecAdd(float *A_h, float *B_h, float *C_h, int n){
	int size = n * sizeof(float);
	float *A_d, *B_d, *C_d;
	cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
	cudaMalloc((void**) &C_d, size);
	
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = ceil((float)n / threadsPerBlock); // calculates number of blocks needed to have a thread for every element in the array 
	vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);

	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
	
}

int main() {
	float A[4] = {1, 2, 3, 4};
	float B[4] = {1, 2, 3, 4};
	float C[4];
	vecAdd(A, B, C, 4);
	for (int i = 0; i < 4; i++) {
		printf("%f ", C[i]);
	}
	printf("\n");
	return 0;
}