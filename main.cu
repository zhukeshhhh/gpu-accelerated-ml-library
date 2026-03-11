#include "algebra/algebra.h"
#include "arena/arena-cpu.h"
#include "arena/arena-gpu.h"
#include <cuda_runtime.h>


int main(void) {
    matrix* a = mat_create(5, 5, GPU);
    matrix* b = mat_create(5, 5, GPU);
    matrix* c = mat_create(5, 5, GPU);

    mat_fill(a, 1.0f);
    mat_fill(b, 1.0f);

    mat_add(c, a, b);

    print_matrix(c);

    return 0;
}