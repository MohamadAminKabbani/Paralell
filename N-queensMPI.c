#include <stdio.h> 
#include <time.h>
#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdbool.h> 

#define MASTER 0
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
        totalsolutions++;
    } else {
        board[0] = queenValue; //set static for parallelization
        for (int col = 0; col < N; col++) {
            board[rows] = col;
            if (isSafe(board, rows)){
                queens(board, rows+1, queenValue);
        	}
        }
    }
}


int main(int argc, char * argv[]) {
    double start, finish, total_time;//Elements to calculate the time
    int myrank, source, size;
    int board[N];
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (myrank == MASTER)  {
        start = MPI_Wtime();
        for (source = 0; source < size - 1; source++) {
            int answer;
            MPI_Recv(&answer, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            totalsolutions += answer;
        }
        finish = MPI_Wtime();
        total_time = finish - start;
        printf("\nTotal Solutions: %d\nRun Time: %lfs\n", totalsolutions, total_time);
    }
    else{
        queens(board, 1, myrank - 1);
        MPI_Send(&totalsolutions, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
