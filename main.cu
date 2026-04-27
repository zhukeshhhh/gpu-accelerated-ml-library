#include "include/base.h"
#include "include/algebra.h"
#include <cuda_runtime.h>

// TODO: learn matrix coalesting for mat_mul
// TODO: learn how to do matmul for transposed matrices
// TODO: fix the parameters for each kernel
// TODO: add error checking code

int main(void) {
    matrix* ad = mat_create(100, 100, GPU);
    matrix* bd = mat_create(100, 100, GPU);
    matrix* sum_abd = mat_create(100, 100, GPU);
    matrix* mul = mat_create(100, 100, GPU);

    mat_fill(ad, 1.2f);
    mat_fill(bd, 1.0f);

    mat_add(sum_abd, ad, bd);
    mat_mul(mul, ad, sum_abd, false, false, true);

    printf("MULTIPLICATION COMPELED\n");
    print_matrix(mul);
    printf("SUM of FINAL matrix : %f\n", mat_sum(mul));

    return 0;
}