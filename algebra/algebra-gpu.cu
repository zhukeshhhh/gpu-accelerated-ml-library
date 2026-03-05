#include "algebra.h"
#include "../arena/arena-gpu.h"


// TODO: implement GPU WRAPPERS
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

// TODO: implement CUDA KERNELS