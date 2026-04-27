#include "../include/base.h"
#include "../include/algebra.h"
#include <math.h>

#define TILE_SIZE 32

matrix* mat_create_gpu(u32 rows, u32 cols) {
    matrix* mat = (matrix*)malloc(sizeof(matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->device = GPU;
    cudaError_t err = cudaMallocManaged((void**)&mat->data, sizeof(f32) * rows * cols);
    if (err != cudaSuccess) {
        printf("mat_create_gpu(): failed to create a matrix on a GPU\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    printf("Matrix on a GPU was created successfully! Rows: %d | Cols: %d\n", mat->rows, mat->cols);
    return mat;
}

void mat_fill_gpu(matrix* mat, f32 val) {
    u64 mat_size = mat_elemnum(mat);
    u64 threads = 256;
    u64 blocks = (mat_size + threads - 1) / threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);
    f32* dst = mat->data;
    mat_fill_kernel<<<BLOCKS, THREADS>>>(dst, val, mat_size);
    cudaError err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("mat_fill_gpu(): failed to fill the matrix on a GPU!\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void mat_clear_gpu(matrix* mat) {
    u64 mat_size = mat_elemnum(mat);
    cudaError_t err = cudaMemset(&mat, 0, mat_size * sizeof(f32));
    if (err != cudaSuccess) {
        printf("mat_clear_gpu(): failed to clear the matrix on a GPU!\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

b32 mat_add_gpu(matrix* out, const matrix* a, const matrix* b) {
    u64 mat_size = out->cols * out->rows;
    u64 threads = 256;
    u64 blocks = (mat_size + threads - 1) / threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);
    f32* out_data = out->data;
    f32* a_data = a->data;
    f32* b_data = b->data;
    mat_add_kernel<<<BLOCKS, THREADS>>>(out_data, a_data, b_data, mat_size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("mat_add_gpu(): failed to add matrices!\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

b32 mat_sub_gpu(matrix* out, const matrix* a, const matrix* b) {
    u64 mat_size = mat_elemnum(out);
    u64 threads = 256;
    u64 blocks = (mat_size + threads - 1) / threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);
    f32* out_data = out->data;
    f32* a_data = a->data;
    f32* b_data = b->data;
    mat_sub_kernel<<<BLOCKS, THREADS>>>(out_data, a_data, b_data, mat_size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("mat_sub_gpu(): failed to subtract matrices!\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

b32 mat_mul_gpu(matrix* out, const matrix* a, const matrix* b, b32 zero_out, b32 transpose_a, b32 transpose_b) {

    u64 a_rows = transpose_a ? a->cols : a->rows;
    u64 a_cols = transpose_a ? a->rows : a->cols;
    u64 b_rows = transpose_b ? b->cols : b->rows;
    u64 b_cols = transpose_b ? b->rows : b->cols;

    u64 N = out->rows;
    u64 M = out->cols;
    u64 K = a_cols;

    if (a_cols != b_rows) {
        printf("mat_mul_gpu(): sizes of A and B are incompatible! \
            a_cols(%ld) != b_rows(%ld)\n", a_cols, b_rows);
        return false;
    }
    if (out->rows != a_rows || out->cols != b_cols) { 
        printf("mat_mul_gpu(): sizes of A or B with OUT matrix are incompatible! \
            out->rows(%d) != a_rows(%ld) or out->cols(%d) != b_cols(%ld)\n", out->rows, a_rows, out->cols, b_cols);
        return false;
    }

    if (zero_out) { 
        mat_clear_gpu(out);
    }

    dim3 THREADS(TILE_SIZE, TILE_SIZE);
    dim3 BLOCKS((out->cols + TILE_SIZE - 1) / TILE_SIZE, (out->rows + TILE_SIZE - 1) / TILE_SIZE);

    u32 transpose = (transpose_a << 1) | transpose_b;
    switch (transpose) {
        case 0b00: { mat_mul_kernel<<<BLOCKS, THREADS>>>(out->data, a->data, b->data, N, M, K, NN); } break;
        case 0b01: { mat_mul_kernel<<<BLOCKS, THREADS>>>(out->data, a->data, b->data, N, M, K, NT); } break;
        case 0b10: { mat_mul_kernel<<<BLOCKS, THREADS>>>(out->data, a->data, b->data, N, M, K, TN); } break;
        case 0b11: { mat_mul_kernel<<<BLOCKS, THREADS>>>(out->data, a->data, b->data, N, M, K, TT); } break;
    }    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("mat_mul_gpu(): failed to multiply matrices!\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

void mat_scale_gpu(matrix* mat, f32 scale) {
    u64 mat_size = mat_elemnum(mat);
    u64 threads = 256;
    u64 blocks = (mat_size + threads - 1) / threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);
    f32* mat_data = mat->data;
    mat_scale_kernel<<<BLOCKS, THREADS>>>(mat_data, scale, mat_size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("mat_sub_gpu(): failed to scale the matrix!\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

f32 mat_sum_gpu(matrix* mat) {
    u64 mat_size = mat_elemnum(mat);
    f32* out;
    cudaMallocManaged(&out, sizeof(f32));
    u32 threads = 128;
    mat_sum_kernel<<<1, threads, threads * sizeof(f32)>>>(mat->data, out, mat_size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("mat_sum_gpu(): failed to sum the matrix!\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    f32 result = *out;
    cudaFree(out);
    return result;
}

b32 mat_relu_gpu(matrix* out, const matrix* in) { return true; }
b32 mat_softmax_gpu(matrix* out, const matrix* in) { return true; }

__global__ void mat_cross_entropy_kernel(f32* out_data, const f32* p_data, const f32* q_data, u64 n) {
    u64 tid = (u64)threadIdx.x + (u64)blockIdx.x * (u64)blockDim.x;
    if (tid < n) {
        f32 pv = p_data[tid];
        f32 qv = q_data[tid];
        if (pv == 0.0f) {
            out_data[tid] = 0.0f;
        }
        else {
            out_data[tid] = pv * -logf(qv);
        }
    }
}

b32 mat_cross_entropy_gpu(matrix* out, const matrix* p, const matrix* q) {
    if (p->rows != q->rows || p->cols != q->cols) { return false; }
    if (out->rows != p->rows || out->cols != p->cols) { return false; }

    u64 n = mat_elemnum(out);
    u64 threads = 256;
    u64 blocks = (n + threads - 1) / threads;
    dim3 THREADS((u32)threads);
    dim3 BLOCKS((u32)blocks);
    mat_cross_entropy_kernel<<<BLOCKS, THREADS>>>(out->data, p->data, q->data, n);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("mat_cross_entropy_gpu(): failed!\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}
b32 mat_relu_add_grad_gpu(matrix* out, const matrix* in) { return true; }
b32 mat_softmax_add_grad_gpu(matrix* out, const matrix* softmax_out) { return true; }
b32 mat_cross_entropy_add_grad_gpu(matrix* out, const matrix* p, const matrix* q) { return true; }


__global__ void mat_add_kernel(f32* out_data, const f32* a_data, const f32* b_data, u64 N) {
    u64 tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        out_data[tid] = a_data[tid] + b_data[tid];
    }
}

__global__ void mat_sub_kernel(f32* out_data, const f32* a_data, const f32* b_data, u64 N) {
    u64 tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        out_data[tid] = a_data[tid] - b_data[tid];
    }
}


__global__ void mat_mul_kernel(f32* out_data, const f32* a_data, const f32* b_data, u64 N, u64 M, u64 K, matmulTransposeCase tcase) {
    __shared__ f32 a_shared[TILE_SIZE][TILE_SIZE];
    __shared__ f32 b_shared[TILE_SIZE][TILE_SIZE + 1];

    u64 local_row = threadIdx.y;
    u64 local_col = threadIdx.x;

    u64 row = threadIdx.y + blockDim.y * blockIdx.y;
    u64 col = threadIdx.x + blockDim.x * blockIdx.x;

    f32 partial_sum = 0.0f;
    for (u64 phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; phase++) {

        switch (tcase) {
        case NN: {
            if ((row < N) && (phase * TILE_SIZE + local_col) < K) {
                a_shared[local_row][local_col] = a_data[row * K + phase * TILE_SIZE + local_col]; // constant row and changing col
            }
            else {
                a_shared[local_row][local_col] = 0.0f;
            }

            if ((col < M) && (phase * TILE_SIZE + local_row) < K) {
                b_shared[local_row][local_col] = b_data[(phase * TILE_SIZE + local_row) * M + col]; // constant col and changing row
            }
            else {
                b_shared[local_row][local_col] = 0.0f;
            }
        } break;

        case NT: {
            if ((row < N) && (phase * TILE_SIZE + local_col) < K) {
                a_shared[local_row][local_col] = a_data[row * K + phase * TILE_SIZE + local_col]; // constant row and changing col
            }
            else {
                a_shared[local_row][local_col] = 0.0f;
            }

            if ((col < M) && (phase * TILE_SIZE + local_row) < K) {
                b_shared[local_row][local_col] = b_data[col * K + phase * TILE_SIZE + local_row]; // constant row and changing col
            }
            else {
                b_shared[local_row][local_col] = 0.0f;
            }
        } break;

        case TN: {
            if ((row < N) && (phase * TILE_SIZE + local_col) < K) {
                a_shared[local_row][local_col] = a_data[(phase * TILE_SIZE + local_col) * N + row]; // constant col and changing row
            }
            else {
                a_shared[local_row][local_col] = 0.0f;
            }

            if ((col < M) && (phase * TILE_SIZE + local_row) < K) {
                b_shared[local_row][local_col] = b_data[(phase * TILE_SIZE + local_row) * M + col]; // constant row and changing col
            }
            else {
                b_shared[local_row][local_col] = 0.0f;
            }
        } break;

        case TT: {
            if ((row < N) && (phase * TILE_SIZE + local_col) < K) {
                a_shared[local_row][local_col] = a_data[(phase * TILE_SIZE + local_col) * N + row]; // constant col and changing row
            }
            else {
                a_shared[local_row][local_col] = 0.0f;
            }

            if ((col < M) && (phase * TILE_SIZE + local_row) < K) {
                b_shared[local_row][local_col] = b_data[col * K + phase * TILE_SIZE + local_row];  // constant row and changing col
            }
            else {
                b_shared[local_row][local_col] = 0.0f;
            }
        } break;
        
        default:
            break;
        }
        __syncthreads();

        for (u64 k = 0; k < TILE_SIZE; k++) {
            partial_sum += a_shared[local_row][k] * b_shared[k][local_col];
        }
        __syncthreads();
    }
    if (row < N && col < M) {
        out_data[row * M + col] = partial_sum;
    }
}

__global__ void mat_fill_kernel(f32* dst, f32 val, u64 mat_size) {
    u64 tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < mat_size) {
        dst[tid] = val;
    }
}

__global__ void mat_scale_kernel(f32* mat, f32 scale, u64 mat_size) {
    u32 tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < mat_size) {
        mat[tid] += scale;
    }
}

__global__ void mat_sum_kernel(f32* mat, f32* out, u64 N) {
    u64 tid = threadIdx.x;
    f32 sum = 0.0f;
    for (u64 i = tid; i < N; i += blockDim.x) {
        sum += mat[i];
    }
    extern __shared__ f32 sdata[];
    sdata[tid] = sum;
    __syncthreads();
    for (u64 stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out = sdata[0];
    }
}


__global__ void mat_relu_kernel(matrix* out, const matrix* in);
__global__ void mat_softmax_kernel(matrix* out, const matrix* in);
__global__ void mat_relu_add_grad_kernel(matrix* out, const matrix* in);
__global__ void mat_softmax_add_grad_kernel(matrix* out, const matrix* softmax_out);
__global__ void mat_cross_entropy_add_grad_kernel(matrix* out, const matrix* p, const matrix* q);
