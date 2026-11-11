// The game's zstd decoder on the host's libzstd. Most RomFS files of TotK 1.4.3 are zstd frames, many with a
// dictionary from Pack/ZsDic.pack.zs; the decoder is BSD zstd linked into main. The patch set "zstd"
// (recomp/patches-v143.txt) sends every call of ZSTD_decompressMultiFrame (main+0x3a1f0) and of ZSTD_decompress_usingDDict
// (main+0x3a1e0) to the two host functions below, on the patch page. Lifted, the decoder ran its literal and match
// copies as chains of small functions with the registers in Cpu; here the same frames decode in native code.
//
// - Results match the guest's: libzstd decodes the same format, and every result that is an error goes to the guest
//   decoder again (it writes its own error code; a frame libzstd rejects may still be one the game's version accepts).
// - A guest ZSTD_DDict (zstd's struct: dictBuffer, dictContent, dictSize first) becomes a host ZSTD_DDict made from a
//   copy of its content, cached by guest address, content address, size and a fingerprint of the content.
// - Inputs of at least AWAY_BYTES give the virtual core up while they decode (other guest threads run on it).
// TOTK_ZSTD_NATIVE=0 keeps the guest decoder (the patch set switch); TOTK_ZSTD_CHECK=1 decodes each call both ways,
// keeps the guest's result and logs every call where the two differ.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

#include "cfg.h"
#include "host.h"

#if __has_include("patches.h")
#include "patches.h"
#include "platform/platform.h"
#endif

#ifdef RT_PATCH_PAGE

#define GUEST_MULTIFRAME 0x700003a1f0ull   // 1.4.3 main: ZSTD_decompressMultiFrame (only 1.4.3 has the patch sites)
#define GUEST_USING_DDICT 0x700003a1e0ull  // ZSTD_decompress_usingDDict
#define HOST_MULTIFRAME (RT_PATCH_PAGE + 0xff0)
#define HOST_USING_DDICT (RT_PATCH_PAGE + 0xfe0)
#define AWAY_BYTES (64u << 10)
#define MAX_DICT (64u << 20)
#define NDICT 32

static void *gp(uint64_t a) { return (void *)(uintptr_t)a; }

typedef struct {
    uint64_t guest, content, size, print;
    ZSTD_DDict *dd;
} Dict;
static Dict dicts[NDICT];
static int ndicts;
static PlatMutex dict_lock = PLAT_MUTEX_INIT;

static struct {
    uint64_t calls, in, out, ns, guest_ns, fallbacks, away, inplace;
} S;

static int check_mode(void) {
    static int v = -1;
    if (v < 0) v = cfg_bool("TOTK_ZSTD_CHECK");
    return v;
}

// Size, both ends and a sample of the middle: what tells a new dictionary at a reused address from the old one.
static uint64_t fingerprint(const uint8_t *p, uint64_t n) {
    uint64_t h = n * 0x9E3779B97F4A7C15ull;
    uint64_t at[3] = {0, n / 2, n > 64 ? n - 64 : 0};
    for (int k = 0; k < 3; k++)
        for (uint64_t i = at[k]; i < at[k] + 64 && i < n; i += 8) {
            uint64_t v = 0;
            memcpy(&v, p + i, n - i < 8 ? n - i : 8);
            h = (h ^ v) * 0xC2B2AE3D27D4EB4Full;
            h ^= h >> 29;
        }
    return h;
}

// The host dictionary for a guest ZSTD_DDict or raw dictionary bytes (guest == 0); NULL when its fields look wrong.
static ZSTD_DDict *dict_for(uint64_t guest, uint64_t content, uint64_t size) {
    if (!content || !size || size > MAX_DICT || content >= RT_ADDRESS_LIMIT || content + size > RT_ADDRESS_LIMIT)
        return NULL;
    uint64_t print = fingerprint(gp(content), size);
    ZSTD_DDict *dd = NULL;
    plat_mutex_lock_shared(&dict_lock);
    for (int i = 0; i < ndicts && !dd; i++)
        if (dicts[i].guest == guest && dicts[i].content == content && dicts[i].size == size && dicts[i].print == print)
            dd = dicts[i].dd;
    plat_mutex_unlock_shared(&dict_lock);
    if (dd) return dd;
    // A copy: the guest may free or reuse its buffer. Entries are never freed (another thread may be decoding with
    // one); a full table sends further dictionaries to the guest decoder.
    ZSTD_DDict *made = ZSTD_createDDict(gp(content), (size_t)size);
    if (!made) return NULL;
    int slot = -1;
    plat_mutex_lock(&dict_lock);
    for (int i = 0; i < ndicts && !dd; i++)   // another thread may have made it meanwhile
        if (dicts[i].guest == guest && dicts[i].content == content && dicts[i].size == size && dicts[i].print == print)
            dd = dicts[i].dd;
    if (!dd && ndicts < NDICT) slot = ndicts, dicts[ndicts++] = (Dict){guest, content, size, print, made}, dd = made;
    plat_mutex_unlock(&dict_lock);
    if (dd != made) {
        ZSTD_freeDDict(made);
        if (!dd) rt_log("zstd: more than %d dictionaries: further ones decode in the guest", NDICT);
        return dd;
    }
    rt_log("zstd: host dictionary %d for %s %#llx: %llu bytes at %#llx, id %u", slot, guest ? "DDict" : "raw dict",
           (unsigned long long)(guest ? guest : content), (unsigned long long)size, (unsigned long long)content,
           ZSTD_getDictID_fromDDict(dd));
    return dd;
}

// Decodes on this thread's host context. Returns libzstd's result; *ok 0 when the inputs cannot be taken.
static size_t host_decode(uint64_t dst, uint64_t cap, uint64_t src, uint64_t size, uint64_t dict, uint64_t dict_size,
                          uint64_t ddict, int *ok) {
    static __thread ZSTD_DCtx *dctx;
    *ok = 0;
    if ((cap && !dst) || !src || src + size > RT_ADDRESS_LIMIT) return 0;   // dst: decode() checks a guest one
    ZSTD_DDict *dd = NULL;
    if (ddict) {
        if (ddict >= RT_ADDRESS_LIMIT - 24) return 0;
        uint64_t f[3];   // ZSTD_DDict: dictBuffer, dictContent, dictSize
        memcpy(f, gp(ddict), sizeof f);
        if (!(dd = dict_for(ddict, f[1], f[2]))) {
            static int logged;
            if (logged < 4) {
                uint64_t w[8];
                memcpy(w, gp(ddict), sizeof w);
                logged++;
                rt_log("zstd: DDict %#llx not taken: %llx %llx %llx %llx %llx %llx %llx %llx", (unsigned long long)ddict,
                       (unsigned long long)w[0], (unsigned long long)w[1], (unsigned long long)w[2], (unsigned long long)w[3],
                       (unsigned long long)w[4], (unsigned long long)w[5], (unsigned long long)w[6], (unsigned long long)w[7]);
            }
            return 0;
        }
    } else if (dict && dict_size) {
        if (!(dd = dict_for(0, dict, dict_size))) return 0;
    }
    if (!dctx && !(dctx = ZSTD_createDCtx())) return 0;
    *ok = 1;
    return dd ? ZSTD_decompress_usingDDict(dctx, gp(dst), (size_t)cap, gp(src), (size_t)size, dd)
              : ZSTD_decompressDCtx(dctx, gp(dst), (size_t)cap, gp(src), (size_t)size);
}

// The host call inside the thread's host depth, as hostfn.c counts it for asynchronous preemption.
static size_t tracked_decode(Cpu *c, uint64_t dst, uint64_t cap, uint64_t src, uint64_t size, uint64_t dict,
                             uint64_t dict_size, uint64_t ddict, int *ok) {
    int *d = kern_host_track && c->thread ? kern_host_depth(c->thread) : NULL;
    if (d) __atomic_store_n(d, *d + 1, __ATOMIC_RELAXED);
    int away = size >= AWAY_BYTES ? kern_core_away() : 0;
    if (away) __atomic_add_fetch(&S.away, 1, __ATOMIC_RELAXED);
    size_t r = host_decode(dst, cap, src, size, dict, dict_size, ddict, ok);
    kern_core_back(&away);
    if (d) {
        int n = *d - 1;
        __atomic_store_n(d, n, __ATOMIC_RELAXED);
        if (!n && kern_preempt_pending(c->thread)) kern_host_left(c->thread);
    }
    return r;
}

static void count(uint64_t in, uint64_t out, uint64_t ns) {
    uint64_t n = __atomic_add_fetch(&S.calls, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&S.in, in, __ATOMIC_RELAXED);
    __atomic_add_fetch(&S.out, out, __ATOMIC_RELAXED);
    __atomic_add_fetch(&S.ns, ns, __ATOMIC_RELAXED);
    if (n == 1 || !(n & 0x3fff))
        rt_log("zstd: %llu calls on the host, %.1f MiB in, %.1f MiB out, %.0f ms (check mode: the guest decoder %.0f ms); "
               "%llu to the guest decoder, %llu in place in the guest, %llu with the core given up", (unsigned long long)n,
               S.in / 1048576.0, S.out / 1048576.0, S.ns / 1e6, S.guest_ns / 1e6, (unsigned long long)S.fallbacks,
               (unsigned long long)S.inplace, (unsigned long long)S.away);
}

// x0 dctx, x1 dst, x2 capacity, x3 src, x4 size, then (dict, dict size, ddict) as the entry has them. guest: the
// guest function with the same arguments, n of them.
static void decode(Cpu *c, uint64_t dict, uint64_t dict_size, uint64_t ddict, uint64_t guest, int n) {
    uint64_t a[8];
    memcpy(a, c->x, sizeof a);
    uint64_t dst = a[1], cap = a[2], src = a[3], size = a[4];
    // In place (the input inside the output buffer): the result depends on how the decoder's writes overtake its
    // reads, which libzstd does not promise to do as the game's version does, and a host decode that failed half way
    // would leave the guest decoder a damaged input. Such calls stay in the guest.
    if (cap && src < dst + cap && dst < src + size) {
        __atomic_add_fetch(&S.inplace, 1, __ATOMIC_RELAXED);
        rt_call(c, guest, n, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
        return;
    }
    if (check_mode()) {
        uint8_t *mine = cap ? malloc((size_t)cap) : NULL;
        int ok = 0;
        uint64_t t0 = kern_now_ns();
        size_t r = cap && !mine ? 0 : tracked_decode(c, (uint64_t)(uintptr_t)mine, cap, src, size, dict, dict_size, ddict, &ok);
        uint64_t t1 = kern_now_ns();
        uint64_t g = rt_call(c, guest, n, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
        __atomic_add_fetch(&S.guest_ns, kern_now_ns() - t1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&S.ns, t1 - t0, __ATOMIC_RELAXED);
        static int logged;
        if (logged < 6 && (!ok || ZSTD_isError(r))) {
            logged++;
            rt_log("zstd check: host %s, guest %llu; dctx %#llx dst %#llx cap %llu src %#llx size %llu dict %#llx/%llu "
                   "ddict %#llx", !ok ? "inputs not taken" : ZSTD_getErrorName(r), (unsigned long long)g,
                   (unsigned long long)a[0], (unsigned long long)dst, (unsigned long long)cap, (unsigned long long)src,
                   (unsigned long long)size, (unsigned long long)dict, (unsigned long long)dict_size,
                   (unsigned long long)ddict);
        }
        if (ok && !ZSTD_isError(r) && (g != r || (r && memcmp(mine, gp(dst), r)))) {
            const uint8_t *p = gp(src);
            size_t fs = ZSTD_findFrameCompressedSize(p, (size_t)size);
            unsigned long long cs = ZSTD_getFrameContentSize(p, (size_t)size);
            ZSTD_DCtx *fresh = ZSTD_createDCtx();
            size_t r2 = ZSTD_decompressDCtx(fresh, mine, (size_t)cap, p, (size_t)size);
            ZSTD_freeDCtx(fresh);
            static int nlog;
            if (nlog++ < 8)
                rt_log("zstd check: MISMATCH %llu bytes from %#llx (cap %llu, first frame %lld bytes, content %lld, magic "
                       "%02x%02x%02x%02x, header %02x, dict %#llx/%llu, ddict %#llx, entry %s, lr %#llx): the host decoded "
                       "%llu bytes, the guest %lld%s; a fresh context without a dictionary: %s", (unsigned long long)size,
                       (unsigned long long)src, (unsigned long long)cap, ZSTD_isError(fs) ? -1ll : (long long)fs,
                       (long long)cs, p[3], p[2], p[1], p[0], p[4], (unsigned long long)dict,
                       (unsigned long long)dict_size, (unsigned long long)ddict, n == 8 ? "MultiFrame" : "usingDDict",
                       (unsigned long long)c->x[30], (unsigned long long)r, (long long)g, g == r ? " (other contents)" : "",
                       ZSTD_isError(r2) ? ZSTD_getErrorName(r2) : "decodes");
        }
        free(mine);
        count(size, ok && !ZSTD_isError(r) ? r : 0, 0);
        return;
    }
    uint64_t t0 = kern_now_ns();
    int ok = 0;
    size_t r = dst + cap > RT_ADDRESS_LIMIT ? 0 : tracked_decode(c, dst, cap, src, size, dict, dict_size, ddict, &ok);
    if (!ok || ZSTD_isError(r)) {   // the guest decoder decides (and writes its own error code)
        __atomic_add_fetch(&S.fallbacks, 1, __ATOMIC_RELAXED);
        static int logged;
        if (logged < 8) {
            logged++;
            rt_log("zstd: %llu bytes at %#llx go to the guest decoder: %s", (unsigned long long)size,
                   (unsigned long long)src, !ok ? "inputs not taken" : ZSTD_getErrorName(r));
        }
        rt_call(c, guest, n, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
        return;
    }
    c->x[0] = r;
    count(size, r, kern_now_ns() - t0);
}

static void host_multiframe(Cpu *c) { decode(c, c->x[5], c->x[6], c->x[7], GUEST_MULTIFRAME, 8); }
static void host_using_ddict(Cpu *c) { decode(c, 0, 0, c->x[5], GUEST_USING_DDICT, 6); }

__attribute__((constructor)) static void zstd_register(void) {
    rt_register_host(HOST_MULTIFRAME, host_multiframe);
    rt_register_host(HOST_USING_DDICT, host_using_ddict);
}


// ---- Bananza 1.0.0: the one-shot decoder (TOTK_ZSTD=1, default off)
// Candidate 5 of the probe, main+0x401fc4 (the probe found 5 call sites, 5141 calls and 1.08 s of thread time in one Canyon run), is
// the game's one-shot multi-frame decoder: size_t f(void *dst, uint32 dst_capacity, const void *src, uint32 src_size, void *dctx_memory,
// int flag). With flag 0 and an 8-byte aligned dctx_memory it initialises a zstd DCtx in dctx_memory, skips skippable frames
// (0x184d2a5x), decodes the zstd frames (magic 0xfd2fb528) into dst and returns the bytes written, or a zstd error code
// (ERROR(srcSize_wrong), ERROR(dstSize_tooSmall) ... as a size_t near 2^64) or 0xffffffff; flag nonzero hands the DCtx to
// main+0x148a010 (ZSTD_decompressDCtx with dictionary handling), an unaligned memory returns 0xffffffff.
// The host decodes the same frames with libzstd's ZSTD_decompressDCtx (the same multi-frame loop). What the host does not do is use
// dctx_memory: the game's DCtx is scratch it initialises on every call (nothing else of the game reads it between calls, as far as the
// lifted code shows). Only flag 0 with an aligned memory goes to the host, and every result that is an error, every in-place call
// (src inside dst) and every call the host cannot take stays with the guest decoder, which then writes its own result.
// Candidate 4 (main+0x3ce730, 14873 calls a run) is a streaming decoder with a DCtx the game keeps between calls: not replaced.
// TOTK_ZSTD_CHECK=1 decodes each call both ways, keeps the guest's result and logs every difference (sizes only).
#define GUEST_ONESHOT_DKB 0x7000401fc4ull

static int zstd_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = cfg_get("TOTK_ZSTD");
        v = e && !strcmp(e, "1");
    }
    return v;
}

// Returns 1 when the call was answered (c->x[0] set); 0 leaves it to the guest decoder (the caller passes it through).
static int native_oneshot(Cpu *c) {
    uint64_t a[8];
    memcpy(a, c->x, sizeof a);
    uint64_t dst = a[0], cap = (uint32_t)a[1], src = a[2], size = (uint32_t)a[3], ws = a[4], flag = a[5];
    if (flag || (ws & 7) || !ws || !dst || !src || !cap || !size || dst + cap > RT_ADDRESS_LIMIT) return 0;
    if (src < dst + cap && dst < src + size) {   // in place: left to the guest, as on TotK
        __atomic_add_fetch(&S.inplace, 1, __ATOMIC_RELAXED);
        return 0;
    }
    if (check_mode()) {
        uint8_t *mine = malloc((size_t)cap);
        int ok = 0;
        uint64_t t0 = kern_now_ns();
        size_t r = mine ? tracked_decode(c, (uint64_t)(uintptr_t)mine, cap, src, size, 0, 0, 0, &ok) : 0;
        uint64_t t1 = kern_now_ns();
        uint64_t g = rt_call(c, GUEST_ONESHOT_DKB, 6, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
        __atomic_add_fetch(&S.guest_ns, kern_now_ns() - t1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&S.ns, t1 - t0, __ATOMIC_RELAXED);
        static int nlog;
        if (ok && !ZSTD_isError(r) && (g != r || memcmp(mine, gp(dst), (size_t)r)) && nlog++ < 8)
            rt_log("zstd check: MISMATCH one-shot: %llu bytes in, capacity %llu: the host decoded %llu bytes, the guest returned %llu%s",
                   (unsigned long long)size, (unsigned long long)cap, (unsigned long long)r, (unsigned long long)g,
                   g == r ? " (other contents)" : "");
        else if ((!ok || ZSTD_isError(r)) && nlog++ < 8)
            rt_log("zstd check: one-shot, host %s, guest returned %llu (%llu bytes in, capacity %llu)",
                   !ok ? "inputs not taken" : ZSTD_getErrorName(r), (unsigned long long)g, (unsigned long long)size,
                   (unsigned long long)cap);
        free(mine);
        count(size, ok && !ZSTD_isError(r) ? r : 0, 0);
        return 1;   // the guest's result stands in c->x[0]
    }
    uint64_t t0 = kern_now_ns();
    int ok = 0;
    size_t r = tracked_decode(c, dst, cap, src, size, 0, 0, 0, &ok);
    if (!ok || ZSTD_isError(r)) {   // the guest decoder decides
        __atomic_add_fetch(&S.fallbacks, 1, __ATOMIC_RELAXED);
        return 0;
    }
    c->x[0] = r;
    count(size, r, kern_now_ns() - t0);
    return 1;
}

// ---- DKB_ZSTD_PROBE (Bananza 1.0.0): which guest functions are the decoder's entry points
//   DKB_ZSTD_PROBE=1 (patch set zstdprobe, recomp/patches-dkb100.txt; default off): finds out which guest functions of Bananza 1.0.0
//   are the zstd decoder's entry points, before any is replaced by libzstd (runtime/ov_zstd.c does that for TotK 1.4.3).
//  
//   Each direct call (bl) of the candidate functions, from the call graph of the lifted main, goes to a host function on the patch
//   page (+0xf00 + 0x10 * k for candidate k) that calls the original with the same eight argument registers, the same stack and
//   the same lr, and returns what it returned: the call is a pass-through. It counts, per call site and window of 60 frames
//   (TOTK_TRACE=perf): the calls, the time in the original, the return values (a size_t error code is a value near 2^64), and for
//   each of the eight argument registers how often it was a pointer to bytes that start with the zstd frame magic and the range of
//   the values that were small integers (a size or capacity). Sizes, counts and times only; no byte of the game's data is logged.
#define NCAND 7
#define MAX_SITES 24
#define SLOT0 0xf00u
#define SLOT_SIZE 0x10u

// Guest addresses of the candidates, in the order of tools' gen_probe_sites (slot k).
static const uint64_t cand[NCAND] = {0x700014ff00ull, 0x700014a540ull, 0x7000375f20ull, 0x7000376920ull,
                                     0x70003ce730ull, 0x7000401fc4ull, 0x7001489770ull};

typedef struct {
    uint64_t lr, calls, ns, errors, ok, rmin, rmax;
    uint64_t magic[8], small_n[8], small_lo[8], small_hi[8], ptr_n[8];
    int used, k;
} Site;

static Site sites[MAX_SITES];
static PlatMutex probe_lock = PLAT_MUTEX_INIT;

static Site *site_for(int k, uint64_t lr) {
    for (int i = 0; i < MAX_SITES; i++)
        if (sites[i].used && sites[i].k == k && sites[i].lr == lr) return &sites[i];
    for (int i = 0; i < MAX_SITES; i++)
        if (!sites[i].used) {
            sites[i] = (Site){.used = 1, .k = k, .lr = lr, .rmin = UINT64_MAX};
            memset(sites[i].small_lo, 0xff, sizeof sites[i].small_lo);
            return &sites[i];
        }
    return NULL;
}

static void probe(Cpu *c, int k) {
    uint64_t a[8], lr = c->x[30];
    memcpy(a, c->x, sizeof a);
    static int enabled = -1;   // the sites also route to the host for TOTK_ZSTD=1 alone: then a plain pass-through
    if (enabled < 0) {
        const char *e = cfg_get("DKB_ZSTD_PROBE");
        enabled = e && !strcmp(e, "1");
    }
    if (!enabled) {
        rt_call(c, cand[k], 8, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
        return;
    }
    int magic[8] = {0}, small[8] = {0}, ptr[8] = {0};
    for (int i = 0; i < 8; i++) {
        if (a[i] >= 0x10000 && a[i] < RT_ADDRESS_LIMIT - 4) {
            if (plat_mem_readable(a[i], 4)) {
                ptr[i] = 1;
                uint32_t m;
                memcpy(&m, (const void *)(uintptr_t)a[i], 4);
                magic[i] = m == 0xFD2FB528u;
            }
        } else if (a[i] && a[i] < (1ull << 32)) {
            small[i] = 1;
        }
    }
    uint64_t t0 = kern_now_ns();
    uint64_t r = rt_call(c, cand[k], 8, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    uint64_t dt = kern_now_ns() - t0;
    plat_mutex_lock(&probe_lock);
    Site *s = site_for(k, lr);
    if (s) {
        s->calls++, s->ns += dt;
        if (r > (uint64_t)-200) s->errors++;
        else {
            s->ok++;
            if (r < s->rmin) s->rmin = r;
            if (r > s->rmax) s->rmax = r;
        }
        for (int i = 0; i < 8; i++) {
            s->magic[i] += (uint64_t)magic[i], s->ptr_n[i] += (uint64_t)ptr[i];
            if (small[i]) {
                s->small_n[i]++;
                if (a[i] < s->small_lo[i]) s->small_lo[i] = a[i];
                if (a[i] > s->small_hi[i]) s->small_hi[i] = a[i];
            }
        }
    }
    plat_mutex_unlock(&probe_lock);
}

#define SLOT(K) static void host_probe_##K(Cpu *c) { probe(c, K); }
SLOT(0) SLOT(1) SLOT(2) SLOT(3) SLOT(4) SLOT(6)
static void host_probe_5(Cpu *c) {
    if (zstd_on() && native_oneshot(c)) return;
    probe(c, 5);
}

// Called every 60 frames by perf.c (TOTK_TRACE=perf): one line per call site that was called, then the window restarts.
void ov_zstd_probe_log(uint64_t frame) {
    static uint64_t last_calls;
    if (S.calls != last_calls) {
        last_calls = S.calls;
        rt_log("zstd frame %llu: %llu calls on the host, %.1f MiB in, %.1f MiB out, %.0f ms native%s; %llu left to the guest decoder "
               "(error, in place or not taken), %llu with the core given up", (unsigned long long)frame, (unsigned long long)S.calls,
               S.in / 1048576.0, S.out / 1048576.0, S.ns / 1e6, check_mode() ? "" : " (check mode off)", (unsigned long long)S.fallbacks,
               (unsigned long long)S.away);
        if (check_mode()) rt_log("zstd frame %llu: check mode: the guest decoder took %.0f ms for the same calls", (unsigned long long)frame,
                                 S.guest_ns / 1e6);
    }
    plat_mutex_lock(&probe_lock);
    for (int i = 0; i < MAX_SITES; i++) {
        Site *s = &sites[i];
        if (!s->used || !s->calls) continue;
        char args[400] = "";
        size_t o = 0;
        for (int j = 0; j < 8 && o < sizeof args - 80; j++) {
            if (s->magic[j]) o += (size_t)snprintf(args + o, sizeof args - o, " a%d:magic*%llu", j, (unsigned long long)s->magic[j]);
            else if (s->small_n[j] && s->small_n[j] >= s->calls / 2)
                o += (size_t)snprintf(args + o, sizeof args - o, " a%d:%llu..%llu", j, (unsigned long long)s->small_lo[j],
                                      (unsigned long long)s->small_hi[j]);
            else if (s->ptr_n[j] >= s->calls / 2) o += (size_t)snprintf(args + o, sizeof args - o, " a%d:ptr", j);
        }
        rt_log("zstd probe frame %llu: call site main+%#llx -> candidate %d (%#llx): %llu calls, %.2f ms in the original; "
               "returns: %llu errors, %llu ok [%llu..%llu]; args:%s", (unsigned long long)frame,
               (unsigned long long)(s->lr - 4 - 0x7000000000ull), s->k, (unsigned long long)cand[s->k],
               (unsigned long long)s->calls, s->ns / 1e6, (unsigned long long)s->errors, (unsigned long long)s->ok,
               (unsigned long long)(s->ok ? s->rmin : 0), (unsigned long long)s->rmax, args);
        s->calls = s->ns = s->errors = s->ok = 0;
        s->rmin = UINT64_MAX, s->rmax = 0;
        memset(s->magic, 0, sizeof s->magic), memset(s->small_n, 0, sizeof s->small_n), memset(s->ptr_n, 0, sizeof s->ptr_n);
        memset(s->small_lo, 0xff, sizeof s->small_lo), memset(s->small_hi, 0, sizeof s->small_hi);
    }
    plat_mutex_unlock(&probe_lock);
}

__attribute__((constructor)) static void zstd_probe_register(void) {
    static void (*const fn[NCAND])(Cpu *) = {host_probe_0, host_probe_1, host_probe_2, host_probe_3,
                                             host_probe_4, host_probe_5, host_probe_6};
    for (unsigned k = 0; k < NCAND; k++) rt_register_host(RT_PATCH_PAGE + SLOT0 + k * SLOT_SIZE, fn[k]);
}


#else
void ov_zstd_probe_log(uint64_t frame) { (void)frame; }
#endif
