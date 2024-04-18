#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 11

int totalsolutions = 0;

bool isSafe(int board[], int row) {
    for (int i = 0; i < row; i++) {
        if (board[i] == board[row] || abs(board[i] - board[row]) == (row - i))
            return false; // same diagonal
    }
    return true;
}

void queens(int board[], int rows, int queenValue) {
    if (rows == N) {
        #pragma omp atomic
        totalsolutions++;
    } else {
        board[0] = queenValue;
        #pragma omp parallel for
        for (int col = 0; col < N; col++) {
            board[rows] = col;
            if (isSafe(board, rows)) {
                queens(board, rows + 1, queenValue);
            }
        }
    }
}

int main() {
    double start, finish, total_time;

    start = omp_get_wtime();

omp_set_num_threads(11);
   
    #pragma omp parallel 
    {
     printf("%d",omp_get_num_threads());
    #pragma omp for reduction(+:totalsolutions)
    for (int queenValue = 0; queenValue < N; queenValue++) {
        int board[N] = {0}; // Initialize board for each queen value
        queens(board, 1, queenValue);
    }}

    finish = omp_get_wtime();
    total_time = finish - start;
    printf("\nTotal Solutions: %d\nRun Time: %lfs\n", totalsolutions, total_time);

    return 0;
}

