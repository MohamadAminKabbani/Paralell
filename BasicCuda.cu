#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 1024  // Define the size of the matrix

// Kernel function for matrix multiplication
__global__ void matrixMultiplication(float *A, float *B, float *C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    // Allocate memory for matrices on the host
    float *h_A = (float*)malloc(WIDTH * WIDTH * sizeof(float));
    float *h_B = (float*)malloc(WIDTH * WIDTH * sizeof(float));
    float *h_C = (float*)malloc(WIDTH * WIDTH * sizeof(float));

    // Allocate memory for matrices on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, WIDTH * WIDTH * sizeof(float));
    cudaMalloc(&d_B, WIDTH * WIDTH * sizeof(float));
    cudaMalloc(&d_C, WIDTH * WIDTH * sizeof(float));

    // Initialize matrices A and B with some values
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        h_A[i] = 0.01f * i;
        h_B[i] = 0.02f * i;
    }

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((WIDTH + 15) / 16, (WIDTH + 15) / 16);

    // Execute the matrix multiplication kernel
    matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, WIDTH);

    // Copy result matrix C from device to host
    cudaMemcpy(h_C, d_C, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
