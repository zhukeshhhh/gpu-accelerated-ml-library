#include <stdlib.h>
#include <stdio.h>
#include "arena-gpu.h"

b32 cuda_available() {
    i32 count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);

    if (count == 0 || error != cudaSuccess) { return 0; }
    return 1;
}

void* gpu_data_alloc(float* data, u64 size) {
    cudaMalloc(&data, size);
    return data;
}