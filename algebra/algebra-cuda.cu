#include "algebra.h"
#include "../arena/arena-cuda.h"


int main(void) {
    return 0;
}

__global__ void _mat_mul_nn(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 k = 0; k < a->cols; k++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] += 
                    a->data[k + i * a->cols] * 
                    b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_tn(matrix* out, const matrix* a, const matrix* b) {
    for (u64 k = 0; k < a->rows; k++) {
        for (u64 i = 0; i < out->rows; i++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] += 
                    a->data[i + k * a->cols] * 
                    b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_nt(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->cols; k++) {
                out->data[j + i * out->cols] += 
                    a->data[k + i * a->cols] * 
                    b->data[k + j * b->cols];
            }
        }
    }
}

void _mat_mul_tt(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->rows; k++) {
                out->data[i * out->cols + j] += a->data[k * a->cols + i] * b->data[b->rows * j + k];
            }
        }
    }
}


b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b, 
    b32 zero_out, b32 transpose_a, b32 transpose_b
) {
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows) { return false; }
    if (out->rows != a_rows || out->cols != b_cols) { return false; }

    if (zero_out) { mat_clear(out); }

    u32 transpose = (transpose_a << 1) | transpose_b;
    switch (transpose) {
        case 0b00: { _mat_mul_nn<<<1, 16>>>(out, a, b); } break;
        case 0b01: { _mat_mul_nt(out, a, b); } break;
        case 0b10: { _mat_mul_tn(out, a, b); } break;
        case 0b11: { _mat_mul_tt(out, a, b); } break;
    }
    
    return true;
}
