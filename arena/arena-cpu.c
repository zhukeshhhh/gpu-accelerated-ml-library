// This code was adapted from a video of "Magicalbat" YouTube channel
// Video URL: "https://youtu.be/jgiMagdjA1s"
// Original GitHub repo: "https://github.com/Magicalbat/videos/blob/main/arena.c"
#pragma once
#include "arena-cpu.h"

cpu_arena* cpu_arena_create(u64 capacity) {
    cpu_arena* arena = (cpu_arena*)malloc(capacity);
    arena->pos = CPU_ARENA_BASE_POS;

    return arena;
}

void cpu_arena_destroy(cpu_arena* arena) {
    free(arena);
}

void* cpu_arena_push(cpu_arena* arena, u64 size) {
    u64 aligned_pos = CPU_ALIGN_UP(arena->pos, CPU_ARENA_ALIGN);
    u64 new_pos = aligned_pos + size;

    if (new_pos > arena->capacity) { return NULL; }

    arena->pos = new_pos;

    u64* out = (u64*)arena + aligned_pos;
    return out;
}

void cpu_arena_pop(cpu_arena* arena, u64 size) {
    if (size > arena->pos - CPU_ARENA_BASE_POS) {
        size = arena->pos - CPU_ARENA_BASE_POS;
    }
    arena->pos -= size;
}

void cpu_arena_pop_to(cpu_arena* arena, u64 new_pos) {
    u64 size = new_pos < arena->pos ? arena->pos - new_pos : 0;
    arena_pop(arena, size);
}


void cpu_arena_clear(cpu_arena* arena) {
    arena_pop_to(arena, CPU_ARENA_BASE_POS);
}