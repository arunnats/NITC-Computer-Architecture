#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>  // For CPU timing

#define CUDA_CHK(...) { \
  cudaError_t cuda_err_code = __VA_ARGS__; \
  if (cuda_err_code != cudaSuccess) { \
    printf("%s failed with code %d\n", #__VA_ARGS__, cuda_err_code); \
    abort(); \
  } \
}

#define TINY 1.0e-40
#define a(i,j,N) a[(i)*(N)+(j)]



// Function to print matrices to a file in row-major order
void print_output(FILE* output_file, int N, float* L, float* U, float* X) {
    fprintf(output_file, "%d\n", N); // First line is N

    // Printing the lower triangular matrix L
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i > j) {
                fprintf(output_file, "%f ", L[i * N + j]);
            } else if (i == j) {
                fprintf(output_file, "1.000000 ");  // Diagonal is 1 in L
            } else {
                fprintf(output_file, "0.000000 ");  // Upper part is 0
            }
        }
        fprintf(output_file, "\n");
    }

    // Printing the upper triangular matrix U
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i <= j) {
                fprintf(output_file, "%f ", U[i * N + j]);
            } else {
                fprintf(output_file, "0.000000 ");  // Lower part is 0
            }
        }
        fprintf(output_file, "\n");
    }

    // Printing the solution vector X
    for (int i = 0; i < N; i++) {
        fprintf(output_file, "%f\n", X[i]);
    }
}

int main(){
    int N;
    float *a, *b;
    FILE *file = fopen("input.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return EXIT_FAILURE;
    }

    // Start timing for reading matrices from file
    clock_t start_read = clock();

    fscanf(file, "%d", &N);

    a = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            fscanf(file, "%f", &a[i * N + j]);

    for (int i = 0; i < N; i++)
        fscanf(file, "%f", &b[i]);

    fclose(file);

    // End timing for reading matrices
    clock_t end_read = clock();
    double time_read = ((double)(end_read - start_read)) / CLOCKS_PER_SEC;

    cudaSetDevice(0);
    float* d_A;
    float* d_b;
    float* d_L;
    float* d_U;

    CUDA_CHK(cudaMalloc((void**)&d_A, sizeof(float) * N * N));
    CUDA_CHK(cudaMalloc((void**)&d_b, sizeof(float) * N));
    CUDA_CHK(cudaMalloc((void**)&d_L, sizeof(float) * N * N));
    CUDA_CHK(cudaMalloc((void**)&d_U, sizeof(float) * N * N));

    CUDA_CHK(cudaMemcpy(d_A, a, sizeof(float) * N * N, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice));

    // Start GPU timing
    cudaEvent_t start_decomp, stop_decomp;
    cudaEvent_t start_solve, stop_solve;

    cudaEventCreate(&start_decomp);
    cudaEventCreate(&stop_decomp);
    cudaEventCreate(&start_solve);
    cudaEventCreate(&stop_solve);

    // Start timing for L and U computation
    cudaEventRecord(start_decomp, 0);

    int shared_size = (N * N + N) * sizeof(float);  // Memory for shared A and B
    solve<<<1, 1, shared_size>>>(d_A, d_b, 1, N, d_L, d_U);

    // Stop timing for L and U computation
    cudaEventRecord(stop_decomp, 0);
    cudaEventSynchronize(stop_decomp);

    // Time taken for L and U computation
    float time_decomp = 0;
    cudaEventElapsedTime(&time_decomp, start_decomp, stop_decomp);

    // Start timing for solving equations
    cudaEventRecord(start_solve, 0);

    // Assume solve function is part of the kernel solve, hence timed together

    // Stop timing for solving equations
    cudaEventRecord(stop_solve, 0);
    cudaEventSynchronize(stop_solve);

    // Time taken for solving the equations
    float time_solve = 0;
    cudaEventElapsedTime(&time_solve, start_solve, stop_solve);

    // Total time in milliseconds
    float total_time = time_decomp + time_solve;

    // Copy results back to host
    CUDA_CHK(cudaMemcpy(b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost));

    // Allocate space for L and U matrices
    float* L = (float*)malloc(N * N * sizeof(float));
    float* U = (float*)malloc(N * N * sizeof(float));
    CUDA_CHK(cudaMemcpy(L, d_L, sizeof(float) * N * N, cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaMemcpy(U, d_U, sizeof(float) * N * N, cudaMemcpyDeviceToHost));

    FILE *output_file = fopen("output.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error opening output file.\n");
        return EXIT_FAILURE;
    }

    print_output(output_file, N, L, U, b);
    fclose(output_file);

    // Output timing information to a separate file
    FILE *timing_file = fopen("timing.txt", "w");
    if (timing_file == NULL) {
        fprintf(stderr, "Error opening timing file.\n");
        return EXIT_FAILURE;
    }

    fprintf(timing_file, "Time taken to read matrices from file: %f seconds\n", time_read);
    fprintf(timing_file, "Time taken to compute L and U: %f milliseconds\n", time_decomp);
    fprintf(timing_file, "Time taken to solve system: %f milliseconds\n", time_solve);
    fprintf(timing_file, "Total GPU time: %f milliseconds\n", total_time);

    fclose(timing_file);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_L);
    cudaFree(d_U);
    free(a);
    free(b);
    free(L);
    free(U);

    // Destroy events
    cudaEventDestroy(start_decomp);
    cudaEventDestroy(stop_decomp);
    cudaEventDestroy(start_solve);
    cudaEventDestroy(stop_solve);

    return 0;
}
