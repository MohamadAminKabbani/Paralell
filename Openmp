#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_VALUE 1000
#define NUM_BUCKETS 100

void bucketSort(int arr[], int size) {
    int bucket[MAX_VALUE + 1] = {0};
    int tempArr[size];

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        #pragma omp atomic
        bucket[arr[i]]++;
    }

    for (int i = 1; i <= MAX_VALUE; i++) {
        bucket[i] += bucket[i - 1];
    }

    #pragma omp parallel for
    for (int i = size - 1; i >= 0; i--) {
        int value = arr[i];
        int index = --bucket[value];
        tempArr[index] = value;
    }

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        arr[i] = tempArr[i];
    }
}

int main() {
    srand(time(NULL));

    int *arr = malloc(NUM_BUCKETS * sizeof(int));
    for (int i = 0; i < NUM_BUCKETS; i++) {
        arr[i] = rand() % MAX_VALUE;
    }

    clock_t start = clock();

    bucketSort(arr, NUM_BUCKETS);

    clock_t end = clock();

    double elapsed_time = ((double) (end - start)) * 1000 / CLOCKS_PER_SEC;

    printf("Sorted Array:\n");
    for (int i = 0; i < NUM_BUCKETS; i++) {
        printf("%d ", arr[i]);
    }
    printf("\nElapsed time: %.5f ms\n", elapsed_time);

    free(arr);
    return 0;
}
