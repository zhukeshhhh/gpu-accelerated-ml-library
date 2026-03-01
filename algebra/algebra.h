// This code was adapted from a video of "Magicalbat" YouTube channel
// Video URL: "https://youtu.be/hL_n_GljC0I"
// Original GitHub repo: "https://github.com/Magicalbat/videos/tree/main/machine-learning"

#include "../arena/arena.h"
#include "../random/rand.h"

typedef struct {
    u32 rows;
    u32 cols;
    f32* data;
} matrix;

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);
size_t mat_size(matrix* mat);
u64 mat_elemnum(matrix* mat);
b32 mat_copy(matrix* dst, matrix* src);
void mat_clear(matrix* mat);
void mat_fill(matrix* mat, f32 val);
void mat_scale(matrix* mat, f32 scale);
f32 mat_sum(matrix* mat);
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b, 
    b32 zero_out, b32 transpose_a, b32 transpose_b
);
b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad(matrix* out, const matrix* in);
b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out);
b32 mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q);
