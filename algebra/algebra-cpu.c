// Part of this code was adapted from a video of "Magicalbat" YouTube channel
// Video URL: "https://youtu.be/hL_n_GljC0I"
// Original GitHub repo: "https://github.com/Magicalbat/videos/tree/main/machine-learning"

#include "algebra.h"
#include "../arena/arena-cpu.h"

void mat_fill_cpu(matrix* mat, f32 val) {
    u64 size = mat_elemnum(mat);

    for (u64 i = 0; i < size; i++) {
        mat->data[i] = val;
    }
}

void mat_scale_cpu(matrix* mat, f32 scale) {
    u64 size = mat_elemnum(mat);

    for (u64 i = 0; i < size; i++) {
        mat->data[i] *= scale;
    }
}

f32 mat_sum_cpu(matrix* mat) {
    u64 size = mat_elemnum(mat);

    f32 sum = 0;
    for (u64 i = 0; i < size; i++) {
        sum += mat->data[i];
    }

    return sum;
}

b32 mat_add_cpu(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) { return false; }
    if (out->rows != a->rows || out->cols != a->cols) { return false; }

    u64 size = mat_elemnum(out);

    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

b32 mat_sub_cpu(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) { return false; }
    if (out->rows != a->rows || out->cols != a->cols) { return false; }

    u64 size = mat_elemnum(out);

    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

void _mat_mul_nn_cpu(matrix* out, const matrix* a, const matrix* b) {
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

void _mat_mul_tn_cpu(matrix* out, const matrix* a, const matrix* b) {
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

void _mat_mul_nt_cpu(matrix* out, const matrix* a, const matrix* b) {
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

void _mat_mul_tt_cpu(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->rows; k++) {
                out->data[i * out->cols + j] += a->data[k * a->cols + i] * b->data[b->rows * j + k];
            }
        }
    }
}

b32 mat_mul_cpu(
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
        case 0b00: { _mat_mul_nn_cpu(out, a, b); } break;
        case 0b01: { _mat_mul_nt_cpu(out, a, b); } break;
        case 0b10: { _mat_mul_tn_cpu(out, a, b); } break;
        case 0b11: { _mat_mul_tt_cpu(out, a, b); } break;
    }
    
    return true;
}

b32 mat_relu_cpu(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    u64 size = mat_elemnum(out);
    for (u64 i = 0; i < size; i++) {
        out->data[i] = MAX(0, in->data[i]);
    }

    return true;
}

b32 mat_softmax_cpu(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    u64 size = mat_elemnum(out);
    f32 exp_sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        exp_sum += expf(in->data[i]);
    }

    mat_scale_cpu(out, 1.0f / exp_sum);

    return true;
}

b32 mat_cross_entropy_cpu(matrix* out, const matrix* p, const matrix* q) {
    if (p->cols != q->cols || q->rows != p-> rows) { return false; }
    if (out->cols != p->cols || out->rows != p->cols) { return false; }

    u64 size = mat_elemnum(out);
    for (u64 i = 0; i < size; i++) {
        if (p->data[i] == 0.0f) { out->data[i] = 0.0f; }
        else { out->data[i] = p->data[i] * -logf(q->data[i]); }
    }
}


// TODO: finish implementation of these functions
b32 mat_relu_add_grad_cpu(matrix* out, const matrix* in);
b32 mat_softmax_add_grad_cpu(matrix* out, const matrix* softmax_out);
b32 mat_cross_entropy_add_grad_cpu(matrix* out, const matrix* p, const matrix* q);