// A per-frame draw-match key: translated shader hashes, replay operation, and occurrence within that group.
// This does not identify a draw across shader changes or changes to the order of otherwise identical draws.
#ifndef DKB_DRAW_ID_H
#define DKB_DRAW_ID_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t vs, gs, fs;
    uint32_t op, count;
} DrawIdSlot;

// The caller zeroes slots at the start of a capture and serializes access. Exact keys are compared after hashing,
// so a hash collision cannot merge two groups. Returns zero if the table is full.
static inline uint32_t draw_id_next(DrawIdSlot *slots, size_t capacity, uint64_t vs, uint64_t gs, uint64_t fs,
                                    uint32_t op) {
    if (!capacity) return 0;
    uint64_t h = 14695981039346656037ull;
    h = (h ^ vs) * 1099511628211ull;
    h = (h ^ gs) * 1099511628211ull;
    h = (h ^ fs) * 1099511628211ull;
    h = (h ^ op) * 1099511628211ull;
    size_t start = (size_t)(h % capacity);
    for (size_t n = 0; n < capacity; n++) {
        DrawIdSlot *s = &slots[(start + n) % capacity];
        if (!s->count) {
            *s = (DrawIdSlot){vs, gs, fs, op, 1};
            return 1;
        }
        if (s->vs == vs && s->gs == gs && s->fs == fs && s->op == op)
            return ++s->count;
    }
    return 0;
}

#endif
