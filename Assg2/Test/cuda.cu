#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAT1 3
#define TINY 1e-20
#define a(i,j) a[(i)*MAT1+(j)]

__global__ void pivot_decomposition(float *a, int *p, int *q, int N) {
    int k = threadIdx.x;
    
    if (k < N) {
        int i, j;
        int pi = -1, pj = -1;
        float max = 0.0, ftmp;
        
        // Find pivot in submatrix a(k:n, k:n)
        for (i = k; i < N; i++) {
            for (j = k; j < N; j++) {
                if (fabs(a[i * N + j]) > max) {
                    max = fabs(a[i * N + j]);
                    pi = i;
                    pj = j;
                }
            }
        }

        // Swap row
        int tmp = p[k];
        p[k] = p[pi];
        p[pi] = tmp;
        for (j = 0; j < N; j++) {
            ftmp = a[k * N + j];
            a[k * N + j] = a[pi * N + j];
            a[pi * N + j] = ftmp;
        }

        // Swap col
        tmp = q[k];
        q[k] = q[pj];
        q[pj] = tmp;
        for (i = 0; i < N; i++) {
            ftmp = a[i * N + k];
            a[i * N + k] = a[i * N + pj];
            a[i * N + pj] = ftmp;
        }

        __syncthreads(); // Synchronize before next steps

        // Check pivot size and decompose
        if (fabs(a[k * N + k]) > TINY) {
            for (i = k + 1; i < N; i++) {
                ftmp = a[i * N + k] /= a[k * N + k];
                for (j = k + 1; j < N; j++) {
                    a[i * N + j] -= (ftmp * a[k * N + j]);
                }
            }
        }
    }
}

__global__ void forward_substitution(float *a, float *x, int *p, int N) {
    int i = threadIdx.x;
    if (i < N) {
        float ftmp = 0.0;
        for (int j = 0; j < i; j++) {
            ftmp += a[i * N + j] * x[j];
        }
        x[i] = (x[p[i]] - ftmp);
    }
}

__global__ void backward_substitution(float *a, float *x, int *q, int N) {
    int i = threadIdx.x;
    if (i < N) {
        float ftmp = 0.0;
        for (int j = i + 1; j < N; j++) {
            ftmp += a[i * N + j] * x[j];
        }
        x[i] = (x[i] - ftmp) / a[i * N + i];
    }
}

int main() {
    // Host input
    float a[MAT1 * MAT1] = {1, 3, -2, 3, 5, 6, 2, 4, 3};
    float b[MAT1] = {5, 7, 8};
    int p[MAT1], q[MAT1];

    // Initialize pivot arrays
    for (int i = 0; i < MAT1; i++) {
        p[i] = i;
        q[i] = i;
    }

    // Device memory allocation
    float *d_a, *d_b;
    int *d_p, *d_q;
    cudaMalloc((void **)&d_a, MAT1 * MAT1 * sizeof(float));
    cudaMalloc((void **)&d_b, MAT1 * sizeof(float));
    cudaMalloc((void **)&d_p, MAT1 * sizeof(int));
    cudaMalloc((void **)&d_q, MAT1 * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_a, a, MAT1 * MAT1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, MAT1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, MAT1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, MAT1 * sizeof(int), cudaMemcpyHostToDevice);

    // Run pivot decomposition kernel
    pivot_decomposition<<<1, MAT1>>>(d_a, d_p, d_q, MAT1);

    // Run forward substitution kernel      
    forward_substitution<<<1, MAT1>>>(d_a, d_b, d_p, MAT1);

    // Run backward substitution kernel
    backward_substitution<<<1, MAT1>>>(d_a, d_b, d_q, MAT1);

    // Copy the result back to host
    cudaMemcpy(b, d_b, MAT1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Output solution
    for (int i = 0; i < MAT1; i++) {
        printf("x[%d] = %f\n", i, b[i]);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_p);
    cudaFree(d_q);

    return 0;
}
