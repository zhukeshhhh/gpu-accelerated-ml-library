#include "algebra.h"
#include "../arena/arena-gpu.h"


// TODO: implement GPU WRAPPERS
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
    return true;
}

void mat_scale_gpu(matrix* mat, f32 scale) {}
f32 mat_sum_gpu(matrix* mat) { return 1.0f; }
b32 mat_relu_gpu(matrix* out, const matrix* in) { return true; }
b32 mat_softmax_gpu(matrix* out, const matrix* in) { return true; }
b32 mat_cross_entropy_gpu(matrix* out, const matrix* p, const matrix* q) { return true; }
b32 mat_relu_add_grad_gpu(matrix* out, const matrix* in) { return true; }
b32 mat_softmax_add_grad_gpu(matrix* out, const matrix* softmax_out) { return true; }
b32 mat_cross_entropy_add_grad_gpu(matrix* out, const matrix* p, const matrix* q) { return true; }


// TODO: IMPLEMENT CUDA KERNELS
__global__ void mat_add_kernel(float* out_data, const float* a_data, const float* b_data, u64 mat_size) {
    u64 tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < mat_size) {
        out_data[tid] = a_data[tid] + b_data[tid];
    }
}

__global__ void mat_sub_kernel(float* out_data, const float* a_data, const float* b_data, u64 mat_size) {
    u64 tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < mat_size) {
        out_data[tid] = a_data[tid] - b_data[tid];
    }
}

__global__ void mat_mul_kernel(matrix* out, const matrix* a, const matrix* b) {

}

__global__ void mat_fill_kernel(float* dst, f32 val, u64 mat_size) {
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
