#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 1024      // Size of the matrices
#define TILE_SIZE 32          // Size of the tile

int main() {
    // Allocate memory for matrices
    float *matrix_A = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float *matrix_B = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    float *matrix_C = (float*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // Initialize matrices with some values
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        matrix_A[i] = 0.01f * i;
        matrix_B[i] = 0.02f * i;
    }

    clock_t start_time = clock();  // Start timing

    // Perform matrix multiplication using tiling technique for optimization
    #pragma acc data copyin(matrix_A[0:MATRIX_SIZE*MATRIX_SIZE], matrix_B[0:MATRIX_SIZE*MATRIX_SIZE]) copyout(matrix_C[0:MATRIX_SIZE*MATRIX_SIZE])
    {
        #pragma acc parallel loop tile(TILE_SIZE, TILE_SIZE)
        for (int i = 0; i < MATRIX_SIZE; i += TILE_SIZE) {
            for (int j = 0; j < MATRIX_SIZE; j += TILE_SIZE) {
                float C_tile[TILE_SIZE][TILE_SIZE] = {0};

                #pragma acc loop seq
                for (int k = 0; k < MATRIX_SIZE; k++) {
                    float A_tile[TILE_SIZE], B_tile[TILE_SIZE];

                    #pragma acc loop seq
                    for (int kk = 0; kk < TILE_SIZE; kk++) {
                        A_tile[kk] = matrix_A[(i + kk) * MATRIX_SIZE + k];
                        B_tile[kk] = matrix_B[k * MATRIX_SIZE + (j + kk)];
                    }

                    #pragma acc loop vector
                    for (int ii = 0; ii < TILE_SIZE; ii++) {
                        for (int jj = 0; jj < TILE_SIZE; jj++) {
                            C_tile[ii][jj] += A_tile[ii] * B_tile[jj];
                        }
                    }
                }

                #pragma acc loop seq
                for (int ii = 0; ii < TILE_SIZE; ii++) {
                    for (int jj = 0; jj < TILE_SIZE; jj++) {
                        matrix_C[(i + ii) * MATRIX_SIZE + (j + jj)] = C_tile[ii][jj];
                    }
                }
            }
        }
    }

    clock_t end_time = clock();  // End timing
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3f seconds\n", elapsed_time);

    // Free allocated memory
    free(matrix_A);
    free(matrix_B);
    free(matrix_C);

    return 0;
}
