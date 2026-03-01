// Based on pcg random number generator (https://www.pcg-random.org)
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

// This code was adapted from a video of "Magicalbat" YouTube channel
// Video URL: https://youtu.be/jgiMagdjA1s
// Original GitHub repo: "https://github.com/Magicalbat/videos/blob/main/rand.c"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;

#define PI 3.141592653589793238

typedef struct {
    u64 state;
    u64 inc;
    f32 prev_norm;
} prng_state;

void prng_seed_r(prng_state* rng, u64 initstate, u64 initseq);
void prng_seed(u64 initstate, u64 initseq);

u32 prng_rand_r(prng_state* rng);
u32 prng_rand(void);

f32 prng_randf_r(prng_state* rng);
f32 prng_randf(void);

f32 prng_rand_norm_r(prng_state* rng);
f32 prng_rand_norm(void);