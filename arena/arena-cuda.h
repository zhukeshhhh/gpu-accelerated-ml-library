#ifndef ARENA_CUDA_H
#define ARENA_CUDA_H

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef i8 b8;
typedef i32 b32;

#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) << 20)
#define GiB(n) ((u64)(n) << 30)
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define ALIGN_UP(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))
#define ARENA_ALIGN 16

typedef struct {
    void* base;
    u64 size;
    u64 offset;
} cuda_arena;

cuda_arena* cuda_arena_create(u64 size);
void cuda_arena_destroy(cuda_arena* arena);
void* cuda_arena_push(cuda_arena* arena, u64 size);
void cuda_arena_pop(cuda_arena* arena, u64 size);
void cuda_arena_pop_to(cuda_arena* arena, u64 new_pos);
void cuda_arena_clear(cuda_arena* arena);

#define PUSH_STRUCT(arena, T) (T*)cuda_arena_push((arena), sizeof(T), false)
#define PUSH_STRUCT_NZ(arena, T) (T*)cuda_arena_push((arena), sizeof(T), true)
#define PUSH_ARRAY(arena, T, n) (T*)cuda_arena_push((arena), sizeof(T) * (n), false)
#define PUSH_ARRAY_NZ(arena, T, n) (T*)cuda_arena_push((arena), sizeof(T) * (n), true)

#ifdef __cplusplus
}
#endif

#endif