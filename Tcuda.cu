#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 1024   // Define the size of the matrix
#define TILE_WIDTH 16  // Define the size of the tile

__global__ void matrixMultiplicationTiled(float *A, float *B, float *C, int width) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float sum = 0.0f;

    for (int m = 0; m < (width-1)/TILE_WIDTH+1; ++m) {
        if (m*TILE_WIDTH + tx < width && row < width)
            tile_A[ty][tx] = A[row*width + m*TILE_WIDTH + tx];
        else
            tile_A[ty][tx] = 0.0f;

        if (m*TILE_WIDTH + ty < width && col < width)
            tile_B[ty][tx] = B[(m*TILE_WIDTH + ty)*width + col];
        else
            tile_B[ty][tx] = 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += tile_A[ty][k] * tile_B[k][tx];
        __syncthreads();
    }
    if (row < width && col < width)
        C[row*width + col] = sum;
}

int main() {
    // Allocate memory for matrices on host
    float *h_A = (float*)malloc(WIDTH * WIDTH * sizeof(float));
    float *h_B = (float*)malloc(WIDTH * WIDTH * sizeof(float));
    float *h_C = (float*)malloc(WIDTH * WIDTH * sizeof(float));

    // Allocate memory for matrices on device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, WIDTH * WIDTH * sizeof(float));
    cudaMalloc(&d_B, WIDTH * WIDTH * sizeof(float));
    cudaMalloc(&d_C, WIDTH * WIDTH * sizeof(float));

    // Initialize matrices A and B with some values
    for (int i = 0; i < WIDTH*WIDTH; i++) {
        h_A[i] = 0.01f * i;
        h_B[i] = 0.02f * i;
    }

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH, (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH);

    // Execute the matrix multiplication kernel
    matrixMultiplicationTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, WIDTH);

    // Copy result matrix C from device to host
    cudaMemcpy(h_C, d_C, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
