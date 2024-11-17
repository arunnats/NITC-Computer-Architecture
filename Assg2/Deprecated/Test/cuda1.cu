#include <stdio.h>

__global__ void helloCUDA() {
    printf("Hello, CUDA!\n");
}

int main() {
    helloCUDA<<<1,1>>>();
    printf("Hello, CUDAaaaaaa!\n");
    cudaDeviceSynchronize();
    return 0;
}