// This code was adapted from a video of "Magicalbat" YouTube channel
// Video URL: "https://youtu.be/jgiMagdjA1s"
// Original GitHub repo: "https://github.com/Magicalbat/videos/blob/main/arena.c"

#pragma once

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

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
#define CPU_ALIGN_UP(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))
#define CPU_ARENA_BASE_POS (sizeof(cpu_arena))
#define CPU_ARENA_ALIGN (sizeof(void*))

typedef struct {
    u64 capacity;
    u64 pos;
} cpu_arena;

cpu_arena* cpu_arena_create(u64 capacity);
void cpu_arena_destroy(cpu_arena* arena);
void* cpu_arena_push(cpu_arena* arena, u64 size);
void cpu_arena_pop(cpu_arena* arena, u64 size);
void cpu_arena_pop_to(cpu_arena* arena, u64 new_pos);
u64 cpu_arena_available(cpu_arena* arena);
void cpu_arena_clear(cpu_arena* arena);

#ifdef __cplusplus
}
#endif