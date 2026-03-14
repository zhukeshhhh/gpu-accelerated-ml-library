// Part of this code was adapted from a video of "Magicalbat" YouTube channel
// Video URL: "https://youtu.be/hL_n_GljC0I"
// Original GitHub repo: "https://github.com/Magicalbat/videos/tree/main/machine-learning"
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include "../arena/arena-cpu.h"
#include "../arena/arena-gpu.h"
#include "../rand/rand.h"

typedef enum {
    CPU,
    GPU
} deviceType;

typedef struct {
    u32 rows;
    u32 cols;
    f32* data;
    deviceType device;
} matrix;

/*<====================CPU ONLY====================>*/
size_t mat_size(matrix* mat);
u64 mat_elemnum(matrix* mat);
b32 mat_copy(matrix* dst, matrix* src);
void mat_clear(matrix* mat);
void print_matrix(matrix* mat);

/*<====================CPU & GPU====================>*/
matrix* mat_create(u32 rows, u32 cols, deviceType device);
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul(matrix* out, const matrix* a, const matrix* b, b32 zero_out, b32 transpose_a, b32 transpose_b);
void mat_fill(matrix* mat, f32 val);
void mat_scale(matrix* mat, f32 scale);
f32 mat_sum(matrix* mat);
b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad(matrix* out, const matrix* in);
b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out);
b32 mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q);

/*<====================CPU WRAPPERS====================>*/
matrix* mat_create_cpu(u32 rows, u32 cols);
b32 mat_add_cpu(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub_cpu(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul_cpu(matrix* out, const matrix* a, const matrix* b, b32 zero_out, b32 transpose_a, b32 transpose_b);
void mat_fill_cpu(matrix* mat, f32 val);
void mat_scale_cpu(matrix* mat, f32 scale);
f32 mat_sum_cpu(matrix* mat);
b32 mat_relu_cpu(matrix* out, const matrix* in);
b32 mat_softmax_cpu(matrix* out, const matrix* in);
b32 mat_cross_entropy_cpu(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad_cpu(matrix* out, const matrix* in);
b32 mat_softmax_add_grad_cpu(matrix* out, const matrix* softmax_out);
b32 mat_cross_entropy_add_grad_cpu(matrix* out, const matrix* p, const matrix* q);

/*<====================GPU WRAPPERS====================>*/
matrix* mat_create_gpu(u32 rows, u32 cols);
b32 mat_add_gpu(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub_gpu(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul_gpu(matrix* out, const matrix* a, const matrix* b, b32 zero_out, b32 transpose_a, b32 transpose_b);
void mat_fill_gpu(matrix* mat, f32 val);
void mat_scale_gpu(matrix* mat, f32 scale);
f32 mat_sum_gpu(matrix* mat);
b32 mat_relu_gpu(matrix* out, const matrix* in);
b32 mat_softmax_gpu(matrix* out, const matrix* in);
b32 mat_cross_entropy_gpu(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad_gpu(matrix* out, const matrix* in);
b32 mat_softmax_add_grad_gpu(matrix* out, const matrix* softmax_out);
b32 mat_cross_entropy_add_grad_gpu(matrix* out, const matrix* p, const matrix* q);

/*<====================CUDA KERNELS====================>*/
#ifdef __CUDACC__
    __global__ void mat_add_kernel(float* out, const float* a, const float* b, u64 mat_size);
    __global__ void mat_sub_kernel(float* out, const float* a, const float* b, u64 mat_size);
    __global__ void mat_mul_kernel(matrix* out, const matrix* a, const matrix* b);
    __global__ void mat_fill_kernel(float* dst, f32 val, u64 mat_size);
    __global__ void mat_scale_kernel(matrix* mat, f32 scale);
    __global__ void mat_sum_kernel(matrix* mat);
    __global__ void mat_relu_kernel(matrix* out, const matrix* in);
    __global__ void mat_softmax_kernel(matrix* out, const matrix* in);
    __global__ void mat_cross_entropy_kernel(matrix* out, const matrix* p, const matrix* q);
    __global__ void mat_relu_add_grad_kernel(matrix* out, const matrix* in);
    __global__ void mat_softmax_add_grad_kernel(matrix* out, const matrix* softmax_out);
    __global__ void mat_cross_entropy_add_grad_kernel(matrix* out, const matrix* p, const matrix* q);
#endif


#ifdef __cplusplus
}
#endif