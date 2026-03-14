#include "algebra.h"

matrix* mat_create(u32 rows, u32 cols, deviceType device) {
    if (device == GPU) {
        return mat_create_gpu(rows, cols);
    }
    else if (device == CPU) {
        return mat_create_cpu(rows, cols);
    }
    else {
        return NULL;
    }
}

size_t mat_size(matrix* mat) { 
    return sizeof(f32) * (u64)mat->cols * mat->rows; 
}

u64 mat_elemnum(matrix* mat) { 
    return (u64)mat->rows * mat->cols; 
}

b32 mat_copy(matrix* dst, matrix* src) {
    if (dst->rows != src->rows || dst->cols != src->cols) { return false; }
    memcpy(dst->data, src->data, mat_size(dst));
    return true;
}

void mat_clear(matrix* mat) { 
    memset(mat->data, 0, mat_size(mat)); 
}

void print_matrix(matrix* mat) {
    u64 mat_size = mat->rows * mat->cols;
    for (u64 i = 0; i < mat_size; i++) {
        printf("%f ", mat->data[i]);
        if (i % mat->cols == mat->cols - 1) {
            printf("\n");
        }
    }
}

// TODO: Dispatchers: call CPU or GPU implementation for each function
// specific CPU and GPU implementation can be found in corresponding algebra files
void mat_fill(matrix* mat, f32 val) {
    if (mat->device == GPU) {
        mat_fill_gpu(mat, val);
    }
    else if (mat->device == CPU) {
        mat_fill_cpu(mat, val);
    }
}

b32 mat_add(matrix* out, const matrix* a, const matrix* b) {
    if (a->cols != b->cols || a->rows != b->rows) { return false; }
    if (out->cols != a->cols || out->rows != out->rows) { return false; }

    if (out->device == CPU && a->device == CPU && b->device == CPU) {
        return mat_add_cpu(out, a, b);
    }
    else if (out->device == GPU && a->device == GPU && b->device == GPU) {
        return mat_add_gpu(out, a, b);
    }
    else {
        return false;
    }
}

b32 mat_sub(matrix* out, const matrix* a, const matrix* b) {
    if (a->cols != b->cols || a->rows != b->rows) { return false; }
    if (out->cols != a->cols || out->rows != out->rows) { return false; }

    if (out->device == CPU && a->device == CPU && b->device == CPU) {
        return mat_sub_cpu(out, a, b);
    }
    else if (out->device == GPU && a->device == GPU && b->device == GPU) {
        return mat_sub_gpu(out, a, b);
    }
    else {
        return false;
    }
}

b32 mat_mul(matrix* out, const matrix* a, const matrix* b, b32 zero_out, b32 transpose_a, b32 transpose_b) {
    if (out->device == CPU && a->device == CPU && b->device == CPU) {
        return mat_mul_cpu(out, a, b, zero_out, transpose_a, transpose_b);
    }
    else if (out->device == GPU && a->device == GPU && b->device == GPU) {
        return mat_mul_gpu(out, a, b, zero_out, transpose_a, transpose_b);
    }
    else {
        return false;
    }
}

void mat_scale(matrix* mat, f32 scale) {
    if (mat->device == GPU) {
        mat_scale_gpu(mat, scale);
    }
    else if (mat->device == CPU) {
        mat_scale_cpu(mat, scale);
    }
}

f32 mat_sum(matrix* mat) {
    if (mat->device == GPU) {
        mat_sum_gpu(mat);
    }
    else if (mat->device == CPU) {
        mat_sum_cpu(mat);
    }
}

b32 mat_relu(matrix* out, const matrix* in) { return true; }
b32 mat_softmax(matrix* out, const matrix* in) { return true; }
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q) { return true; }
b32 mat_relu_add_grad(matrix* out, const matrix* in) { return true; }
b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out) { return true; }
b32 mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q) { return true; }