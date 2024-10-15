#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define TINY 1e-20
#define MAX_SIZE 100 // Maximum size for the matrix

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(EXIT_FAILURE); }} while(0)

// GPU kernel for row swaps
__global__ void swap_rows(float* a, int* p, int k, int pi, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        float temp = a[k * N + j];
        a[k * N + j] = a[pi * N + j];
        a[pi * N + j] = temp;
    }
}

// GPU kernel for column swaps
__global__ void swap_cols(float* a, int* q, int k, int pj, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float temp = a[i * N + k];
        a[i * N + k] = a[i * N + pj];
        a[i * N + pj] = temp;
    }
}

// Matrix normalization and subtraction kernel
__global__ void normalize_and_subtract(float* a, int k, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > k && i < N) {
        float ftmp = a[i * N + k] /= a[k * N + k];
        for (int j = k + 1; j < N; j++) {
            a[i * N + j] -= ftmp * a[k * N + j];
        }
    }
}

// Host function for pivot decomposition (run on CPU but offload heavy tasks to GPU)
void h_pivot_decomp(float* a, int* p, int* q, int N) {
    int i, j, k;
    int pi, pj, tmp;
    float max;

    // Allocate GPU memory
    float* d_a;
    int* d_p;
    int* d_q;
    CUDA_CALL(cudaMalloc(&d_a, N * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_p, N * sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_q, N * sizeof(int)));

    // Copy data to GPU
    CUDA_CALL(cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_p, p, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_q, q, N * sizeof(int), cudaMemcpyHostToDevice));

    for (k = 0; k < N; k++) {
        pi = -1, pj = -1, max = 0.0;
        // Pivot selection (run on CPU for simplicity)
        for (i = k; i < N; i++) {
            for (j = k; j < N; j++) {
                if (fabs(a[i * N + j]) > max) {
                    max = fabs(a[i * N + j]);
                    pi = i;
                    pj = j;
                }
            }
        }

        // Swap rows on GPU
        tmp = p[k];
        p[k] = p[pi];
        p[pi] = tmp;
        swap_rows<<<(N + 255) / 256, 256>>>(d_a, d_p, k, pi, N);

        // Swap columns on GPU
        tmp = q[k];
        q[k] = q[pj];
        q[pj] = tmp;
        swap_cols<<<(N + 255) / 256, 256>>>(d_a, d_q, k, pj, N);

        // Normalize and subtract on GPU
        normalize_and_subtract<<<(N + 255) / 256, 256>>>(d_a, k, N);
        CUDA_CALL(cudaDeviceSynchronize()); // Sync GPU
    }

    // Copy the result back to the host
    CUDA_CALL(cudaMemcpy(a, d_a, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_p);
    cudaFree(d_q);
}

// Forward/backward substitution kernel (solve phase)
__global__ void forward_substitution(float* a, float* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        for (int j = 0; j < i; j++) {
            b[i] -= a[i * N + j] * b[j];
        }
    }
}

__global__ void backward_substitution(float* a, float* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 0 && i < N) {
        for (int j = i + 1; j < N; j++) {
            b[i] -= a[i * N + j] * b[j];
        }
        b[i] /= a[i * N + i];
    }
}

// Solve function (parallelized using CUDA)
void h_solve(float* a, float* b, int* p, int* q, int N) {
    // Allocate GPU memory for a, b
    float* d_a;
    float* d_b;
    CUDA_CALL(cudaMalloc(&d_a, N * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_b, N * sizeof(float)));

    // Copy data to GPU
    CUDA_CALL(cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

    // Perform forward substitution on GPU
    forward_substitution<<<(N + 255) / 256, 256>>>(d_a, d_b, N);
    CUDA_CALL(cudaDeviceSynchronize()); // Sync GPU

    // Perform backward substitution on GPU
    backward_substitution<<<(N + 255) / 256, 256>>>(d_a, d_b, N);
    CUDA_CALL(cudaDeviceSynchronize()); // Sync GPU

    // Copy the solution back to host
    CUDA_CALL(cudaMemcpy(b, d_b, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
}

int main() {
    int N;
    float *a, *b;
    int *p_pivot, *q_pivot;

    // Read input from the file
    FILE *file = fopen("input.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return EXIT_FAILURE;
    }

    // Read the size of the system
    fscanf(file, "%d", &N);
    if (N > MAX_SIZE) {
        fprintf(stderr, "Matrix size exceeds maximum allowed size (%d).\n", MAX_SIZE);
        fclose(file);
        return EXIT_FAILURE;
    }

    // Allocate memory for matrix A and vector B
    a = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    p_pivot = (int *)malloc(N * sizeof(int));
    q_pivot = (int *)malloc(N * sizeof(int));

    // Read matrix A
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            fscanf(file, "%f", &a[i * N + j]);

    // Read vector B
    for (int i = 0; i < N; i++)
        fscanf(file, "%f", &b[i]);

    fclose(file);

    // Initialize pivot arrays
    for (int i = 0; i < N; i++) {
        p_pivot[i] = i;
        q_pivot[i] = i;
    }

    // Perform pivot decomposition
    h_pivot_decomp(a, p_pivot, q_pivot, N);

    // Solve the system
    h_solve(a, b, p_pivot, q_pivot, N);

    // Output results to file
    FILE* output = fopen("output.txt", "w");
    fprintf(output, "%d\n", N);

    // Output lower triangular matrix L
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i > j)
                fprintf(output, "%.6f ", a[i * N + j]);
            else if (i == j)
                fprintf(output, "1.000000 ");
            else
                fprintf(output, "0.000000 ");
        }
        fprintf(output, "\n");
    }

    // Output upper triangular matrix U
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i <= j)
                fprintf(output, "%.6f ", a[i * N + j]);
            else
                fprintf(output, "0.000000 ");
        }
        fprintf(output, "\n");
    }

    // Output solution vector X
    for (int i = 0; i < N; i++) {
        fprintf(output, "%.6f\n", b[i]);
    }

    fclose(output);

    // Free allocated memory
    free(a);
    free(b);
    free(p_pivot);
    free(q_pivot);

    return EXIT_SUCCESS;
}