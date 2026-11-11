// Small caches of the draw path (TOTK_DRAW_FASTPATH, vk_exec_draw.c), free of Vulkan so tests/spirv_opt_test.py can run them.
#pragma once
#include <stdint.h>

// A one-entry memo of a pure function of (a, b, c, d): get returns 1 and *v when the key repeats.
typedef struct {
    const void *a, *b, *c;
    uint64_t d;
    int v, valid;
} DcMemo;

static inline int dc_memo_get(const DcMemo *m, const void *a, const void *b, const void *c, uint64_t d, int *v) {
    if (!m->valid || m->a != a || m->b != b || m->c != c || m->d != d) return 0;
    *v = m->v;
    return 1;
}

static inline void dc_memo_put(DcMemo *m, const void *a, const void *b, const void *c, uint64_t d, int v) {
    m->a = a, m->b = b, m->c = c, m->d = d, m->v = v, m->valid = 1;
}

// The index buffer the command buffer being recorded has bound. A bind of the same buffer, offset and type is redundant while
// nothing else bound another since; the state lives in the draw path's per-command-buffer block, which is cleared when a
// command buffer starts, and bind sites that do not go through dc_index_redundant clear it with dc_index_reset.
typedef struct {
    uint64_t buffer, offset;
    int type, valid;
} DcIndex;

// 1 when binding (buffer, offset, type) would repeat the last bind; otherwise remembers it and returns 0.
static inline int dc_index_redundant(DcIndex *x, uint64_t buffer, uint64_t offset, int type) {
    if (x->valid && x->buffer == buffer && x->offset == offset && x->type == type) return 1;
    x->buffer = buffer, x->offset = offset, x->type = type, x->valid = 1;
    return 0;
}

static inline void dc_index_reset(DcIndex *x) { x->valid = 0; }
