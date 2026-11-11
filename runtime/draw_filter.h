// Pure helpers of the draw diagnostics in vk_exec_draw.c (DKB_SKIP_ORIGINS, DKB_SKIP_BISECT), free of Vulkan so
// tests/spirv_opt_test.py can run them.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DF_MAX_ORIGINS 64

typedef struct {
    unsigned n;
    uint64_t value[DF_MAX_ORIGINS], mask[DF_MAX_ORIGINS];   // an origin matches when (origin & mask) == value
} DfOrigins;

static inline int df_hex(const char *s, size_t n, uint64_t *out) {
    uint64_t v = 0;
    if (!n || n > 16) return 0;
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (d < 0) return 0;
        v = v << 4 | (uint64_t)d;
    }
    *out = v;
    return 1;
}

// A comma-separated list of origin prefixes of 1..16 hex digits (a 16-digit token is a whole origin; a shorter one matches
// every origin that starts with it). Returns the count, or -1 when a token is empty, too long, not hex or the list is too long.
static inline int df_parse_origins(DfOrigins *o, const char *list) {
    o->n = 0;
    while (list && *list) {
        const char *e = strchr(list, ',');
        size_t n = e ? (size_t)(e - list) : strlen(list);
        uint64_t v;
        if (!df_hex(list, n, &v) || o->n >= DF_MAX_ORIGINS) return -1;
        unsigned shift = (unsigned)(64 - 4 * n);
        o->mask[o->n] = shift ? ~0ull << shift : ~0ull;
        o->value[o->n] = shift ? v << shift : v;
        o->n++;
        if (e && !e[1]) return -1;   // a trailing comma
        list = e ? e + 1 : list + n;
    }
    return (int)o->n;
}

// Like df_parse_origins, but every token must be a whole 16-digit origin (DKB_LQ_FOLD_ORIGINS).
static inline int df_parse_whole_origins(DfOrigins *o, const char *list) {
    for (const char *p = list; p && *p;) {
        const char *e = strchr(p, ',');
        if ((e ? (size_t)(e - p) : strlen(p)) != 16) return o->n = 0, -1;
        p = e ? e + 1 : p + 16;
    }
    return df_parse_origins(o, list);
}

// The index of the first entry that matches origin, or -1. An origin of 0 (not an NVN 2 stage) never matches.
static inline int df_match_origin(const DfOrigins *o, uint64_t origin) {
    for (unsigned i = 0; origin && i < o->n; i++)
        if ((origin & o->mask[i]) == o->value[i]) return (int)i;
    return -1;
}

// "mask:value" in hex. 1 on success. A value with bits outside the mask can never match and is refused.
static inline int df_parse_bisect(const char *s, uint64_t *mask, uint64_t *value) {
    const char *c = s ? strchr(s, ':') : NULL;
    if (!c || !df_hex(s, (size_t)(c - s), mask) || !df_hex(c + 1, strlen(c + 1), value)) return 0;
    return (*value & ~*mask) == 0;
}

#define DF_MAX_TERMS 16
// DKB_SKIP_BISECT / DKB_SKIP_BISECT_FS: up to 16 "mask:value" terms; an origin matches when any term does.
typedef struct { int on; unsigned n; uint64_t mask[DF_MAX_TERMS], value[DF_MAX_TERMS]; } DfBisect;

// "mask:value[,mask:value...]" in hex. Returns the term count, or -1 when a term is malformed (the set is then empty).
static inline int df_parse_bisect_list(DfBisect *b, const char *s) {
    b->on = 0, b->n = 0;
    while (s && *s) {
        const char *e = strchr(s, ',');
        size_t n = e ? (size_t)(e - s) : strlen(s);
        char term[40];
        if (n >= sizeof term || b->n >= DF_MAX_TERMS) return b->n = 0, -1;
        memcpy(term, s, n), term[n] = 0;
        if (!df_parse_bisect(term, &b->mask[b->n], &b->value[b->n])) return b->n = 0, -1;
        b->n++;
        if (e && !e[1]) return b->n = 0, -1;   // a trailing comma
        s = e ? e + 1 : s + n;
    }
    b->on = b->n > 0;
    return (int)b->n;
}

static inline int df_bisect_hit(const DfBisect *b, uint64_t origin) {
    for (unsigned i = 0; b->on && origin && i < b->n; i++)
        if ((origin & b->mask[i]) == b->value[i]) return 1;
    return 0;
}

// One draw's decision: 1 skip, 0 draw. The origins are those of its vertex and fragment stages (0: none or not NVN 2).
// *why: bit 0 the origin list matched, bit 1 the vertex bisect (DKB_SKIP_BISECT), bit 2 the fragment bisect (DKB_SKIP_BISECT_FS).
// Any of them skips the draw; a stage without an origin never matches a bisect.
static inline int df_skip(const DfOrigins *o, const DfBisect *vb, const DfBisect *fb, uint64_t vo, uint64_t fo, int *why) {
    *why = (df_match_origin(o, vo) >= 0 || df_match_origin(o, fo) >= 0) | df_bisect_hit(vb, vo) << 1 | df_bisect_hit(fb, fo) << 2;
    return *why != 0;
}
