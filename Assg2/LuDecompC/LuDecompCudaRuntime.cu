#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHK(...) { \
  cudaError_t cuda_err_code = __VA_ARGS__; \
  if (cuda_err_code != cudaSuccess) { \
    printf("%s failed with code %d\n", #__VA_ARGS__, cuda_err_code); \
    abort(); \
  } \
}

#define TINY 1.0e-40
#define a(i,j,N) a[(i)*(N)+(j)]

__device__ void d_pivot_decomp(float *a, int *p, int *q, int N){
    int i,j,k;
    int pi,pj,tmp;
    float max;
    float ftmp;
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
                for (j=k+1;j<N;j++){
                    a(i,j,N)-=(ftmp*a(k,j,N));
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

__global__ void solve(float *A, float *B, int max, int N){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
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

    d_pivot_decomp(local_A, p_pivot, q_pivot, N);
    d_solve(local_A, local_B, p_pivot, q_pivot, N);

    for (int i = 0; i < N; i++) {
        B[id * N + i] = local_B[i];  // Write back the result
    }
  }

  delete[] p_pivot;  
  delete[] q_pivot;
}

int main(int argc, char *argv[]){
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    int N;
    float *a, *b;
    FILE *file = fopen(input_file, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening input file: %s\n", input_file);
        return EXIT_FAILURE;
    }

    fscanf(file, "%d", &N);

    a = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            fscanf(file, "%f", &a[i * N + j]);

    for (int i = 0; i < N; i++)
        fscanf(file, "%f", &b[i]);

    fclose(file);

    cudaSetDevice(0);
    float* d_A;
    float* d_b;
    CUDA_CHK(cudaMalloc((void**)&d_A, sizeof(float) * N * N));
    CUDA_CHK(cudaMalloc((void**)&d_b, sizeof(float) * N));

    CUDA_CHK(cudaMemcpy(d_A, a, sizeof(float) * N * N, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice));

    int shared_size = (N * N + N) * sizeof(float);  // Memory for shared A and B
    int M = 10;  // Assume we have 10 systems to solve in parallel
    int threadsPerBlock = N;  // Number of threads per block, each handling one row
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;  // Number of blocks
    solve<<<blocksPerGrid, threadsPerBlock, shared_size>>>(d_A, d_b, M, N);  // Kernel with shared memory
    cudaDeviceSynchronize();

    CUDA_CHK(cudaMemcpy(b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost));

    FILE *outfile = fopen(output_file, "w");
    if (outfile == NULL) {
        fprintf(stderr, "Error opening output file: %s\n", output_file);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; i++) {
        fprintf(outfile, "%f\n", b[i]);
    }

    fclose(outfile);

    free(a);
    free(b);
    CUDA_CHK(cudaFree(d_A));
    CUDA_CHK(cudaFree(d_b));

    return 0;
}
