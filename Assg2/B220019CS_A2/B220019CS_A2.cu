#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <fstream>

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
                fprintf(output_file, "%f\n", L[i * N + j]);
            } else if (i == j) {
                fprintf(output_file, "1.000000\n");  // Diagonal is 1 in L
            } else {
                fprintf(output_file, "0.000000\n");  // Upper part is 0
            }
        }
        //fprintf(output_file, "\n");
    }

    // Printing the upper triangular matrix U
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i <= j) {
                fprintf(output_file, "%f\n", U[i * N + j]);
            } else {
                fprintf(output_file, "0.000000\n");  // Lower part is 0
            }
        }
        //fprintf(output_file, "\n");
    }

    // Printing the solution vector X
    for (int i = 0; i < N; i++) {
        fprintf(output_file, "%f\n", X[i]);
    }
}

__device__ void d_pivot_decomp(float *a, int *p, int *q, int N, float *L, float *U){
    int i,j,k;
    int pi,pj,tmp;
    float max;
    float ftmp;

    // Initialize U with values of A and L as 0
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            U[i * N + j] = a[i * N + j];
            L[i * N + j] = (i == j) ? 1.0f : 0.0f;  // Initialize diagonal of L to 1
        }
    }

    for (k=0;k<N;k++){
        pi=-1,pj=-1,max=0.0;
        for (i=k;i<N;i++) {
            for (j=k;j<N;j++) {
                if (fabs(a(i,j,N))>max){
                    max = fabs(a(i,j,N));
                    pi=i;
                    pj=j;
                }
            }
        }
        tmp=p[k];
        p[k]=p[pi];
        p[pi]=tmp;
        for (j=0;j<N;j++){
            ftmp=a(k,j,N);
            a(k,j,N)=a(pi,j,N);
            a(pi,j,N)=ftmp;
        }
        tmp=q[k];
        q[k]=q[pj];
        q[pj]=tmp;
        for (i=0;i<N;i++){
            ftmp=a(i,k,N);
            a(i,k,N)=a(i,pj,N);
            a(i,pj,N)=ftmp;
        }
        if ((fabs(a(k,k,N))>TINY)){
            for (i=k+1;i<N;i++){
                ftmp=a(i,k,N)/=a(k,k,N);
                L[i * N + k] = ftmp;  // Store lower triangular values in L
                for (j=k+1;j<N;j++){
                    a(i,j,N)-=(ftmp*a(k,j,N));
                    U[i * N + j] = a(i,j,N);  // Store upper triangular values in U
                }
            }
        }
    }
}

__device__ void d_solve(float *a, float *x, int *p, int *q, int N){
    int i, ii = 0, j;
    float ftmp;
    float *xtmp = new float[N];  
    int *inverse_q = new int[N];  

    for (i = 0; i < N; i++) {
        inverse_q[q[i]] = i;
    }

    for (i = 0; i < N; i++) {
        xtmp[i] = x[p[i]]; 
    }

    for (i = 0; i < N; i++) {
        ftmp = xtmp[i];
        if (ii != 0)
            for (j = ii - 1; j < i; j++)
                ftmp -= a(i,j,N) * xtmp[j];
        else if (ftmp != 0.0)
            ii = i + 1;
        xtmp[i] = ftmp;
    }

    xtmp[N - 1] /= a(N-1, N-1, N);
    for (i = N - 2; i >= 0; i--) {
        ftmp = xtmp[i];
        for (j = i + 1; j < N; j++) {
            ftmp -= a(i,j,N) * xtmp[j];
        }
        xtmp[i] = ftmp / a(i,i,N);
    }

    for (i = 0; i < N; i++) {
        x[i] = xtmp[inverse_q[i]];
    }

    delete[] xtmp;    
    delete[] inverse_q;  
}

__global__ void solve(float *A, float *B, int max, int N, float *L, float *U){
  int id = blockDim.x*blockIdx.x + threadIdx.x;
  extern __shared__ float shared_mem[];  // Shared memory
  
  int *p_pivot = new int[N];  
  int *q_pivot = new int[N];

  float *local_A = shared_mem;  // Shared memory for matrix A
  float *local_B = &shared_mem[N * N];  // Shared memory for vector B
  
  if ((id < max)){
    for (int i = 0; i < N; i++) {
        p_pivot[i] = q_pivot[i] = i;
    }

    for (int i = 0; i < N * N; i++) {
        local_A[i] = A[id * N * N + i];  // Load data into shared memory
    }

    for (int i = 0; i < N; i++) {
        local_B[i] = B[id * N + i];
    }

    d_pivot_decomp(local_A, p_pivot, q_pivot, N, L, U);
    d_solve(local_A, local_B, p_pivot, q_pivot, N);

    for (int i = 0; i < N; i++) {
        B[id * N + i] = local_B[i];  // Write back the result
    }
  }

  delete[] p_pivot;  
  delete[] q_pivot;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {  // Third argument for timing output file
        fprintf(stderr, "Usage: %s <inputfile> <outputfile> <timingfile>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    const char *timing_filename = argv[3];

    int N;
    float *a, *b;

    // Using <chrono> for timing
    using namespace std::chrono;
    high_resolution_clock::time_point start, end;
    duration<double> time_taken;

    // Timing for reading matrices
    start = high_resolution_clock::now();

    FILE *file = fopen(input_filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s.\n", input_filename);
        return EXIT_FAILURE;
    }

    // fscanf(file, "%d", &N);

    if (fscanf(file, "%d", &N) != 1) {
        fprintf(stderr, "Error reading N from file.\n");
        return EXIT_FAILURE;
    }

    a = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));

    // for (int i = 0; i < N; i++)
    //     for (int j = 0; j < N; j++)
    //         fscanf(file, "%f", &a[i * N + j]);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fscanf(file, "%f", &a[i * N + j]) != 1) {
                fprintf(stderr, "Error reading matrix A from file.\n");
                return EXIT_FAILURE;
            }
        }
    }

    for (int i = 0; i < N; i++) {
        if (fscanf(file, "%f", &b[i]) != 1) {
            fprintf(stderr, "Error reading vector B from file.\n");
            return EXIT_FAILURE;
        }
    }

    
    fclose(file);

    end = high_resolution_clock::now();
    time_taken = duration_cast<duration<double>>(end - start);

    // Open timing file
    std::ofstream timing_file(timing_filename);
    if (!timing_file) {
        std::cerr << "Error opening timing file " << timing_filename << std::endl;
        return EXIT_FAILURE;
    }

    timing_file << "Time taken to read A and B: " << time_taken.count() << " seconds\n";

    // CUDA events for timing GPU computations
    cudaEvent_t cuda_start, cuda_stop;
    float milliseconds = 0;

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

    // Start timing for L and U computation
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);
    cudaEventRecord(cuda_start);

    int shared_size = (N * N + N) * sizeof(float);  // Memory for shared A and B
    int M = 100;  // Assume we have 10 systems to solve in parallel
    int threadsPerBlock = N;  // Number of threads per block, each handling one row
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;  // Number of blocks
    solve<<<blocksPerGrid, threadsPerBlock, shared_size>>>(d_A, d_b, 1, N, d_L, d_U);  // Kernel with shared memory

    cudaEventRecord(cuda_stop);
    cudaEventSynchronize(cuda_stop);
    cudaEventElapsedTime(&milliseconds, cuda_start, cuda_stop);
    timing_file << "Time taken in computing L and U: " << milliseconds / 1000 << " seconds\n";

    CUDA_CHK(cudaMemcpy(b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost));

    // Allocate space for L and U matrices
    float* L = (float*)malloc(N * N * sizeof(float));
    float* U = (float*)malloc(N * N * sizeof(float));
    CUDA_CHK(cudaMemcpy(L, d_L, sizeof(float) * N * N, cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaMemcpy(U, d_U, sizeof(float) * N * N, cudaMemcpyDeviceToHost));

    FILE *output_file = fopen(output_filename, "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error opening output file %s.\n", output_filename);
        return EXIT_FAILURE;
    }

    print_output(output_file, N, L, U, b);
    fclose(output_file);

    // Report total time taken to solve system
    timing_file << "Total time taken to solve system: " << milliseconds / 1000 << " seconds\n";

    // Clean up
    timing_file.close();

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_L);
    cudaFree(d_U);
    free(a);
    free(b);
    free(L);
    free(U);

    return 0;
}
