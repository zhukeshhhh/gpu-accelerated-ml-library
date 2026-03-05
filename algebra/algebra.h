// Part of this code was adapted from a video of "Magicalbat" YouTube channel
// Video URL: "https://youtu.be/hL_n_GljC0I"
// Original GitHub repo: "https://github.com/Magicalbat/videos/tree/main/machine-learning"

#pragma once

#ifdef __cpulsplus
extern C {
#endif

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
matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);
size_t mat_size(matrix* mat);
u64 mat_elemnum(matrix* mat);
b32 mat_copy(matrix* dst, matrix* src);
void mat_clear(matrix* mat);

/*<====================CPU & GPU====================>*/
void mat_fill(matrix* mat, f32 val);
void mat_scale(matrix* mat, f32 scale);
f32 mat_sum(matrix* mat);
b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad(matrix* out, const matrix* in);
b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out);
b32 mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q);
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul(matrix* out, const matrix* a, const matrix* b, b32 zero_out, b32 transpose_a, b32 transpose_b);

/*<====================CPU WRAPPERS====================>*/
void mat_fill_cpu(matrix* mat, f32 val);
void mat_scale_cpu(matrix* mat, f32 scale);
f32 mat_sum_cpu(matrix* mat);
b32 mat_relu_cpu(matrix* out, const matrix* in);
b32 mat_softmax_cpu(matrix* out, const matrix* in);
b32 mat_cross_entropy_cpu(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad_cpu(matrix* out, const matrix* in);
b32 mat_softmax_add_grad_cpu(matrix* out, const matrix* softmax_out);
b32 mat_cross_entropy_add_grad_cpu(matrix* out, const matrix* p, const matrix* q);
b32 mat_add_cpu(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub_cpu(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul_cpu(matrix* out, const matrix* a, const matrix* b, b32 zero_out, b32 transpose_a, b32 transpose_b);

/*<====================GPU WRAPPERS====================>*/
void mat_fill_gpu(matrix* mat, f32 val);
void mat_scale_gpu(matrix* mat, f32 scale);
f32 mat_sum_gpu(matrix* mat);
b32 mat_relu_gpu(matrix* out, const matrix* in);
b32 mat_softmax_gpu(matrix* out, const matrix* in);
b32 mat_cross_entropy_gpu(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad_gpu(matrix* out, const matrix* in);
b32 mat_softmax_add_grad_gpu(matrix* out, const matrix* softmax_out);
b32 mat_cross_entropy_add_grad_gpu(matrix* out, const matrix* p, const matrix* q);
b32 mat_add_gpu(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub_gpu(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul_gpu(matrix* out, const matrix* a, const matrix* b, b32 zero_out, b32 transpose_a, b32 transpose_b);

/*<====================CUDA KERNELS====================>*/
// TODO: finish cuda kernel declarations
#ifdef __CUDACC__
    __global__ void mat_add_kernel(matrix* out, const matrix* a, const matrix* b);
    __global__ void mat_sub_kernel(matrix* out, const matrix* a, const matrix* b);
    __global__ void mat_mul_kernel(matrix* out, const matrix* a, const matrix* b, b32 zero_out, b32 transpose_a, b32 transpose_b);
#endif


#ifdef __cpulsplus
}
#endif