// Slot allocator of the global sampled-texture table (TOTK_BINDLESS_TEX, runtime/vk_bindless.c). Free of Vulkan so
// tests/bindless_table_test.py can run it.
//
// The table is an array of combined image samplers per SPIR-V image class (BT_CLASSES, below). A slot holds what the
// replay's descriptor path would bind for one texture use: an image view, a sampler and a layout. The host resolves a
// game handle to that triple exactly as before (tex_info), and the table only keeps one descriptor per distinct triple
// instead of writing a descriptor per draw. Keying by the resolved triple, not by the game's texture and sampler ids,
// keeps everything that tex_info decides (view swizzle and mip range, the point sampler of a depth texture, the
// feedback-loop layout) out of the shader. Slot 0 of every class is the dummy descriptor and is never given out.
//
// A slot is freed when its view or sampler is destroyed (a handle to a re-initialised texture resolves to a new view
// and so to another triple). The caller rewrites a freed slot with the dummy descriptor.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define BT_CLASSES 32   // dim (1D 2D 3D Cube: bits 0-1) | array << 2 | multisample << 3 | shadow << 4

// The class of a shader texture (GfxShaderRes.type = Ryujinx SamplerType: 1D 1, buffer 2, 2D 3, 3D 4, cube 5; Array 0x100,
// Multisample 0x200, Shadow 0x400). -1 for a texture buffer and anything unknown. tools/xlate patch 0021 uses the same map.
static inline int bt_class_of_type(uint32_t type) {
    int dim;
    switch (type & 0xff) {
    case 1: dim = 0; break;
    case 3: dim = 1; break;
    case 4: dim = 2; break;
    case 5: dim = 3; break;
    default: return -1;
    }
    return dim | (type & 0x100 ? 4 : 0) | (type & 0x200 ? 8 : 0) | (type & 0x400 ? 16 : 0);
}

typedef struct { uint64_t view, samp; uint32_t layout; } BtKey;

typedef struct {
    BtKey key;
    uint32_t next;   // next slot of the same hash bucket, 0: end
    uint8_t used;
} BtSlot;

typedef struct {
    uint32_t cap[BT_CLASSES];       // slots per class including the dummy slot 0; 0: the class does not exist
    BtSlot *slot[BT_CLASSES];
    uint32_t *bucket[BT_CLASSES];   // slot of the first entry of each hash bucket, 0: none
    uint32_t nbucket[BT_CLASSES];
    uint32_t *freelist[BT_CLASSES];
    uint32_t nfree[BT_CLASSES], high[BT_CLASSES];   // high: next never used slot
    uint64_t acquired, fresh, hits, full, released;
    uint64_t gen;                                   // bumped on every release: a cached slot may have been freed
} BtTable;

static inline uint32_t bt_hash(const BtKey *k) {
    uint64_t x = k->view * 0x9E3779B97F4A7C15ull ^ (k->samp + 0x7F4A7C15ull) * 0xC2B2AE3D27D4EB4Full ^ k->layout * 0x165667B19E3779F9ull;
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ull;
    return (uint32_t)(x ^ x >> 32);
}

static inline int bt_init(BtTable *t, const uint32_t cap[BT_CLASSES]) {
    memset(t, 0, sizeof *t);
    for (int c = 0; c < BT_CLASSES; c++) {
        if (cap[c] < 2) continue;
        uint32_t nb = 16;
        while (nb < cap[c] * 2) nb *= 2;
        t->cap[c] = cap[c], t->nbucket[c] = nb, t->high[c] = 1;
        t->slot[c] = calloc(cap[c], sizeof(BtSlot));
        t->bucket[c] = calloc(nb, sizeof(uint32_t));
        t->freelist[c] = malloc(sizeof(uint32_t) * cap[c]);
        if (!t->slot[c] || !t->bucket[c] || !t->freelist[c]) return 0;
    }
    return 1;
}

static inline void bt_free(BtTable *t) {
    for (int c = 0; c < BT_CLASSES; c++) free(t->slot[c]), free(t->bucket[c]), free(t->freelist[c]);
    memset(t, 0, sizeof *t);
}

// The slot of (cls, key), or 0 when it has none.
static inline uint32_t bt_find(const BtTable *t, uint32_t cls, const BtKey *k) {
    if (cls >= BT_CLASSES || !t->cap[cls]) return 0;
    for (uint32_t s = t->bucket[cls][bt_hash(k) & (t->nbucket[cls] - 1)]; s; s = t->slot[cls][s].next) {
        const BtKey *e = &t->slot[cls][s].key;
        if (e->view == k->view && e->samp == k->samp && e->layout == k->layout) return s;
    }
    return 0;
}

// The slot of (cls, key), made when it has none (*fresh 1: the caller writes the descriptor). 0 when the class is full or
// absent: the caller uses the dummy slot 0.
static inline uint32_t bt_acquire(BtTable *t, uint32_t cls, const BtKey *k, int *fresh) {
    *fresh = 0;
    t->acquired++;
    uint32_t s = bt_find(t, cls, k);
    if (s) {
        t->hits++;
        return s;
    }
    if (cls >= BT_CLASSES || !t->cap[cls]) return t->full++, 0;
    if (t->nfree[cls]) s = t->freelist[cls][--t->nfree[cls]];
    else if (t->high[cls] < t->cap[cls]) s = t->high[cls]++;
    else return t->full++, 0;
    uint32_t b = bt_hash(k) & (t->nbucket[cls] - 1);
    t->slot[cls][s] = (BtSlot){*k, t->bucket[cls][b], 1};
    t->bucket[cls][b] = s;
    t->fresh++;
    *fresh = 1;
    return s;
}

static inline const BtKey *bt_key(const BtTable *t, uint32_t cls, uint32_t slot) {
    return cls < BT_CLASSES && slot && slot < t->cap[cls] && t->slot[cls][slot].used ? &t->slot[cls][slot].key : NULL;
}

static inline void bt_unlink(BtTable *t, uint32_t cls, uint32_t s) {
    uint32_t *p = &t->bucket[cls][bt_hash(&t->slot[cls][s].key) & (t->nbucket[cls] - 1)];
    while (*p && *p != s) p = &t->slot[cls][*p].next;
    if (*p) *p = t->slot[cls][s].next;
    t->slot[cls][s].used = 0;
    t->freelist[cls][t->nfree[cls]++] = s;
    t->released++, t->gen++;
}

// Frees every slot whose view (or, below, sampler) is v. cb(user, cls, slot) runs for each freed slot so the caller can write
// the dummy descriptor into it. Returns how many were freed.
static inline int bt_release_view(BtTable *t, uint64_t v, void (*cb)(void *, uint32_t, uint32_t), void *user) {
    int n = 0;
    for (uint32_t c = 0; c < BT_CLASSES; c++)
        for (uint32_t s = 1; s < t->high[c]; s++)
            if (t->slot[c][s].used && t->slot[c][s].key.view == v) {
                bt_unlink(t, c, s);
                if (cb) cb(user, c, s);
                n++;
            }
    return n;
}

static inline int bt_release_sampler(BtTable *t, uint64_t v, void (*cb)(void *, uint32_t, uint32_t), void *user) {
    int n = 0;
    for (uint32_t c = 0; c < BT_CLASSES; c++)
        for (uint32_t s = 1; s < t->high[c]; s++)
            if (t->slot[c][s].used && t->slot[c][s].key.samp == v) {
                bt_unlink(t, c, s);
                if (cb) cb(user, c, s);
                n++;
            }
    return n;
}

static inline uint32_t bt_live(const BtTable *t, uint32_t cls) {
    return cls < BT_CLASSES && t->cap[cls] ? t->high[cls] - 1 - t->nfree[cls] : 0;
}
