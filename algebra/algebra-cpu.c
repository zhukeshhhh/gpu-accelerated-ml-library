// This code was adapted from a video of "Magicalbat" YouTube channel
// Video URL: "https://youtu.be/hL_n_GljC0I"
// Original GitHub repo: "https://github.com/Magicalbat/videos/tree/main/machine-learning"

#include "algebra.h"
#include "../arena/arena.h"

int main(void) {
    printf("HELLO WORLD!\n");

    mem_arena* arena = arena_create(MiB(1), KiB(50));

    matrix* a = mat_create(arena, 10, 20);
    matrix* b = mat_create(arena, 20, 20);

    matrix* c = mat_create(arena, a->rows, b->cols);

    mat_fill(a, 1.0f);
    mat_fill(b, 2.0f);

    mat_fill(c, 0);

    b8 result = mat_mul(c, a, b, false, false, false);


    
    if (result == true){printf("Matmul Success!\n");}
    else if (result == false) {printf("Matmul Filed!\n");}

    printf("<========A=======>\n");

    for (i32 i = 0; i < a->rows; i++) {
        for (i32 j = 0; j < a->cols; j++) {
            printf("%f ", a->data[i * a->cols + j]);
            if (j % a->cols == a->cols - 1) { printf("\n"); }
        }
    }


    printf("<========B=======>\n");

    for (i32 i = 0; i < b->rows; i++) {
        for (i32 j = 0; j < b->cols; j++) {
            printf("%f ", b->data[i * b->cols + j]);
            if (j % b->cols == b->cols - 1) { printf("\n"); }
        }
    }


    printf("<========C=======>\n");

    for (i32 i = 0; i < c->rows; i++) {
        for (i32 j = 0; j < c->cols; j++) {
            printf("%f ", c->data[i * c->cols + j]);
            if (j % c->cols == c->cols - 1) { printf("\n"); }
        }
    }

    arena_destroy(arena);
    return 0;
}

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols) {
    matrix* mat = PUSH_STRUCT(arena, matrix);

    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);

    return mat;
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

void mat_fill(matrix* mat, f32 val) {
    u64 size = mat_elemnum(mat);

    for (u64 i = 0; i < size; i++) {
        mat->data[i] = val;
    }
}

void mat_scale(matrix* mat, f32 scale) {
    u64 size = mat_elemnum(mat);

    for (u64 i = 0; i < size; i++) {
        mat->data[i] *= scale;
    }
}

f32 mat_sum(matrix* mat) {
    u64 size = mat_elemnum(mat);

    f32 sum = 0;
    for (u64 i = 0; i < size; i++) {
        sum += mat->data[i];
    }

    return sum;
}

b32 mat_add(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) { return false; }
    if (out->rows != a->rows || out->cols != a->cols) { return false; }

    u64 size = mat_elemnum(out);

    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

b32 mat_sub(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) { return false; }
    if (out->rows != a->rows || out->cols != a->cols) { return false; }

    u64 size = mat_elemnum(out);

    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

void _mat_mul_nn(matrix* out, const matrix* a, const matrix* b) {
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
        case 0b00: { _mat_mul_nn(out, a, b); } break;
        case 0b01: { _mat_mul_nt(out, a, b); } break;
        case 0b10: { _mat_mul_tn(out, a, b); } break;
        case 0b11: { _mat_mul_tt(out, a, b); } break;
    }
    
    return true;
}

b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad(matrix* out, const matrix* in);
b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out);
b32 mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q);