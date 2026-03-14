#include "algebra.h"
#include "../arena/arena-gpu.h"

#define TILE_SIZE 16

matrix* mat_create_gpu(u32 rows, u32 cols) {
    matrix* mat = (matrix*)malloc(sizeof(matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->device = GPU;
    cudaMallocManaged((void**)&mat->data, sizeof(f32) * rows * cols);
    return mat;
}

void mat_fill_gpu(matrix* mat, f32 val) {
    u64 mat_size = mat->cols * mat->rows;
    u64 threads = 256;
    u64 blocks = (mat_size + threads - 1) / threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);
    f32* dst = mat->data;
    mat_fill_kernel<<<BLOCKS, THREADS>>>(dst, val, mat_size);
    cudaError err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

b32 mat_add_gpu(matrix* out, const matrix* a, const matrix* b) {
    u64 mat_size = out->cols * out->rows;
    u64 threads = 256;
    u64 blocks = (mat_size + threads - 1) / threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);
    f32* out_data = out->data;
    f32* a_data = a->data;
    f32* b_data = b->data;
    mat_add_kernel<<<BLOCKS, THREADS>>>(out_data, a_data, b_data, mat_size);
    cudaDeviceSynchronize();
    return true;
}

b32 mat_sub_gpu(matrix* out, const matrix* a, const matrix* b) {
    u64 mat_size = out->cols * out->rows;
    u64 threads = 256;
    u64 blocks = (mat_size + threads - 1) / threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);
    f32* out_data = out->data;
    f32* a_data = a->data;
    f32* b_data = b->data;
    mat_sub_kernel<<<BLOCKS, THREADS>>>(out_data, a_data, b_data, mat_size);
    cudaDeviceSynchronize();
    return true;
}

b32 mat_mul_gpu(matrix* out, const matrix* a, const matrix* b, b32 zero_out, b32 transpose_a, b32 transpose_b) {
    
}

void mat_scale_gpu(matrix* mat, f32 scale) {}
f32 mat_sum_gpu(matrix* mat) { return 1.0f; }
b32 mat_relu_gpu(matrix* out, const matrix* in) { return true; }
b32 mat_softmax_gpu(matrix* out, const matrix* in) { return true; }
b32 mat_cross_entropy_gpu(matrix* out, const matrix* p, const matrix* q) { return true; }
b32 mat_relu_add_grad_gpu(matrix* out, const matrix* in) { return true; }
b32 mat_softmax_add_grad_gpu(matrix* out, const matrix* softmax_out) { return true; }
b32 mat_cross_entropy_add_grad_gpu(matrix* out, const matrix* p, const matrix* q) { return true; }


__global__ void mat_add_kernel(f32* out_data, const f32* a_data, const f32* b_data, u64 mat_size) {
    u64 tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < mat_size) {
        out_data[tid] = a_data[tid] + b_data[tid];
    }
}

__global__ void mat_sub_kernel(f32* out_data, const f32* a_data, const f32* b_data, u64 mat_size) {
    u64 tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < mat_size) {
        out_data[tid] = a_data[tid] - b_data[tid];
    }
}

__global__ void mat_mul_kernel(f32* out_data, const f32* a_data, const f32* b_data, u64 N, u64 M, u64 K) {
    __shared__ f32 a_shared[TILE_SIZE][TILE_SIZE];
    __shared__ f32 b_shared[TILE_SIZE][TILE_SIZE];

    u64 by = blockIdx.y;
    u64 bx = blockIdx.x;

    u64 local_row = threadIdx.y;
    u64 local_col = threadIdx.x;

    u64 row = threadIdx.y + blockDim.y * blockIdx.y;
    u64 col = threadIdx.x + blockDim.x * blockIdx.x;

    f32 patrial_sum = 0.0f;
    for (u64 phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; phase++) {

        if ((row < N) && (phase * TILE_SIZE + local_col) < K) {
            a_shared[local_row][local_col] = a_data[row * K + phase * TILE_SIZE + local_col]; // constant row and changing col
        }
        else {
            a_shared[local_row][local_col] = 0.0f;
        }

        if ((col < M) && (phase * TILE_SIZE + local_row) < K) {
            b_shared[local_row][local_col] = b_data[(phase * TILE_SIZE + local_row) * K + col]; // constant col and changing row
        }
        else {
            b_shared[local_row][local_col] = 0.0f;
        }

        __syncthreads();

        for (u64 k = 0; k < TILE_SIZE; k++) {
            patrial_sum += a_shared[local_row][k] * b_shared[k][local_col];
        }
        __syncthreads();
    }
    out_data[row * K + col] = patrial_sum;
}

__global__ void mat_fill_kernel(f32* dst, f32 val, u64 mat_size) {
    u64 tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < mat_size) {
        dst[tid] = val;
    }
}
__global__ void mat_scale_kernel(matrix* mat, f32 scale);
__global__ void mat_sum_kernel(matrix* mat);
__global__ void mat_relu_kernel(matrix* out, const matrix* in);
__global__ void mat_softmax_kernel(matrix* out, const matrix* in);
__global__ void mat_cross_entropy_kernel(matrix* out, const matrix* p, const matrix* q);
__global__ void mat_relu_add_grad_kernel(matrix* out, const matrix* in);
__global__ void mat_softmax_add_grad_kernel(matrix* out, const matrix* softmax_out);
__global__ void mat_cross_entropy_add_grad_kernel(matrix* out, const matrix* p, const matrix* q);
