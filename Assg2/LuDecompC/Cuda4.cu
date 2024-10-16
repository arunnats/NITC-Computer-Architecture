#include <utility>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define CUDA_CHK(...) { \
  cudaError_t cuda_err_code = __VA_ARGS__; \
  if (cuda_err_code != cudaSuccess) { \
    printf("%s failed with code %d\n", #__VA_ARGS__, cuda_err_code); \
    abort(); \
  } \
}

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif
#define TINY 1.0e-40
#define a(i,j,N) a[(i)*(N)+(j)]  // Updated for dynamic size `N`

#define GO 1
#define NOGO 0

void Check_Kernel(const char *message){
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess){
    fprintf(stderr,"Error: %s:%s\n",message, cudaGetErrorString(error));
  }
}

__device__ void d_pivot_decomp(float *a, int *p, int *q, int N){
    int i,j,k;
    int pi,pj,tmp;
    float max;
    float ftmp;
    for (k=0;k<N;k++){
        pi=-1,pj=-1,max=0.0;
        // Find pivot in submatrix a(k:n,k:n)
        for (i=k;i<N;i++) {
            for (j=k;j<N;j++) {
                if (fabs(a(i,j,N))>max){
                    max = fabs(a(i,j,N));
                    pi=i;
                    pj=j;
                }
            }
        }
        // Swap Row
        tmp=p[k];
        p[k]=p[pi];
        p[pi]=tmp;
        for (j=0;j<N;j++){
            ftmp=a(k,j,N);
            a(k,j,N)=a(pi,j,N);
            a(pi,j,N)=ftmp;
        }
        // Swap Col
        tmp=q[k];
        q[k]=q[pj];
        q[pj]=tmp;
        for (i=0;i<N;i++){
            ftmp=a(i,k,N);
            a(i,k,N)=a(i,pj,N);
            a(i,pj,N)=ftmp;
        }
        // Check pivot size and decompose
        if ((fabs(a(k,k,N))>TINY)){
            for (i=k+1;i<N;i++){
                // Column normalisation
                ftmp=a(i,k,N)/=a(k,k,N);
                for (j=k+1;j<N;j++){
                    // a(ik)*a(kj) subtracted from lower right submatrix elements
                    a(i,j,N)-=(ftmp*a(k,j,N));
                }
            }
        }
    }
}

__device__ void d_solve(float *a, float *x, int *p, int *q, int N){
    int i,ii=0,j;
    float ftmp;
    float *xtmp = new float[N];  // Dynamically allocate memory for xtmp

    // Swap rows (x=Px)
    for (i=0; i<N; i++){
        xtmp[i]=x[p[i]]; // Value that should be here
    }

    // Lx=x
    for (i=0;i<N;i++){
        ftmp=xtmp[i];
        if (ii != 0)
            for (j=ii-1;j<i;j++)
                ftmp-=a(i,j,N)*xtmp[j];
        else if (ftmp!=0.0)
            ii=i+1;
        xtmp[i]=ftmp;
    }

    // Backward substitution
    xtmp[N-1]/=a(N-1,N-1,N);
    for (i=N-2;i>=0;i--){
        ftmp=xtmp[i];
        for (j=i+1;j<N;j++){
            ftmp-=a(i,j,N)*xtmp[j];
        }
        xtmp[i]=ftmp/a(i,i,N);
    }

    // Swap columns (x=Qx)
    for (i=0;i<N;i++){
        x[i]=xtmp[q[i]];
    }

    delete[] xtmp;  // Free dynamically allocated memory
}

__global__ void solve(float *A, float *B, int max, int N){
  int id= blockDim.x*blockIdx.x + threadIdx.x;
  int *p_pivot = new int[N];  // Dynamically allocate memory for pivots
  int *q_pivot = new int[N];
  
  if ((GO==1) && (id < max)){
    for (int i=0;i<N;i++) {
        p_pivot[i]=q_pivot[i]=i;
    }

    d_pivot_decomp(&A[id*N*N], p_pivot, q_pivot, N);
    d_solve(&A[id*N*N], &B[id*N], p_pivot, q_pivot, N);
  }

  delete[] p_pivot;  // Free dynamically allocated memory
  delete[] q_pivot;
}

int main(){
    int N;
    float *a, *b;
    FILE *file = fopen("input.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return EXIT_FAILURE;
    }

    // Read the size of the system
    fscanf(file, "%d", &N);

    const unsigned int matsize=N*N;
    const unsigned int vecsize=N;

    // Allocate memory for matrix A and vector B
    a = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));

    // Read matrix A
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            fscanf(file, "%f", &a[i*N + j]);

    // Read vector B
    for (int i = 0; i < N; i++)
        fscanf(file, "%f", &b[i]);

    fclose(file);

    // CUDA setup
    cudaSetDevice(0);
    float* d_A;
    float* d_b;
    CUDA_CHK(cudaMalloc((void**)&d_A, sizeof(float)*matsize));
    CUDA_CHK(cudaMalloc((void**)&d_b, sizeof(float)*vecsize));

    // Copy input data to the device
    CUDA_CHK(cudaMemcpy(d_A, a, sizeof(float)*matsize, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_b, b, sizeof(float)*vecsize, cudaMemcpyHostToDevice));

    // Set up kernel execution parameters (use more threads and blocks for larger matrices)
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Execute the kernel
    solve<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_b, 1, N);
    cudaDeviceSynchronize();
    Check_Kernel("Solve");

    // Copy the results back to the host
    CUDA_CHK(cudaMemcpy(b, d_b, sizeof(float)*vecsize, cudaMemcpyDeviceToHost));

    // Output the solution
    printf("Solution vector:\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", b[i]);
    }

    // Clean up
    free(a);
    free(b);
    CUDA_CHK(cudaFree(d_A));
    CUDA_CHK(cudaFree(d_b));

    return 0;
}