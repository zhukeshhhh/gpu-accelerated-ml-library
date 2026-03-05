#include "arena-cuda.h"
#include <stdlib.h>
#include <stdio.h>

int main() {
    printf("Enter\n");
    cuda_arena_create(GiB(1));

    for(;;) {}

    return 0;
}

cuda_arena* cuda_arena_create(u64 size) {
    cuda_arena* arena = (cuda_arena*)malloc(sizeof(cuda_arena));
    if (!arena) { return NULL; }

    cudaError_t error = cudaMalloc(&arena->base, size);

    if (error != cudaSuccess) {
        free(arena);
        return NULL;
    }

    arena->size = size;
    arena->offset = 0;

    return arena;
}

void cuda_arena_destroy(cuda_arena* arena) {
    cudaFree(arena->base);
    free(arena);
}

void* cuda_arena_push(cuda_arena* arena, u64 size) {
    u64 offset_aligned = ALIGN_UP(arena->offset, ARENA_ALIGN);
    u64 new_offset = offset_aligned + size;

    if (new_offset > arena->size) { return NULL; }
    void* out = (u8*)arena->base + offset_aligned;
    arena->offset = new_offset;

    return out;
}

void cuda_arena_pop(cuda_arena* arena, u64 size) {
    if (size > arena->offset) {
        arena->offset = 0;
    }
    arena->offset -= size;
}

void cuda_arena_clear(cuda_arena* arena) {
    arena->offset = 0;
}