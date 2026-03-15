#include "include/base.h"
#include "include/algebra.h"
#include <cuda_runtime.h>

// TODO: learn matrix coalesting for mat_mul
// TODO: learn how to do matmul for transposed matrices
// TODO: fix the parameters for each kernel
// TODO: add error checking code

int main(void) {
    matrix* ad = mat_create(1000, 10, GPU);
    matrix* bd = mat_create(1000, 10, GPU);
    matrix* cd = mat_create(10, 10, GPU);


    mat_fill(ad, 1.1f);
    mat_fill(bd, 1.2f);

    if (mat_mul(cd, ad, bd, false, true, false) == true) {
        print_matrix(cd);
    }

    return 0;
}