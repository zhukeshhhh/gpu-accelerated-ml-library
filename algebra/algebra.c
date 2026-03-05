// This is a unified API to connect CPU and GPU implemented code
// Contains C wrappers for CPU and GPU code
// Contains "CPU only" realizable code

#include "algebra.h"
#include "../arena/arena-gpu.h"

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols) {
    matrix* mat = PUSH_STRUCT(arena, matrix);
    mat->rows = rows;
    mat->cols = cols;
    mat->device = cuda_available() == true ? GPU : CPU;
    
    // TODO: solve question regarding mat->data allocation
    mat->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);
    return mat;
}

size_t mat_size(matrix* mat) { return sizeof(f32) * (u64)mat->cols * mat->rows; }

u64 mat_elemnum(matrix* mat) { return (u64)mat->rows * mat->cols; }

b32 mat_copy(matrix* dst, matrix* src) {
    if (dst->rows != src->rows || dst->cols != src->cols) { return false; }
    memcpy(dst->data, src->data, mat_size(dst));
    return true;
}

void mat_clear(matrix* mat) { memset(mat->data, 0, mat_size(mat)); }


// TODO: Dispatchers: call CPU or GPU implementation for each function
// specific CPU and GPU implementation can be found in corresponding algebra files
void mat_fill(matrix* mat, f32 val) {

}

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