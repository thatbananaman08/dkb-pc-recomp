// DKB_DELTA_SCAN=1 (Bananza 1.0.0 only, off by default): finds which guest variable carries the game's per-frame time step.
//
// The game's frame controller (main+0x1eace8) picks a present interval and main+0x1eaed0 stores it at window object
// +0x88 before calling nvnWindowSetPresentInterval. nvnWindowSetPresentInterval notes that object G (x20 of the caller). At
// every present this file copies G and the objects its pointer fields name (0x300 bytes each, at most 24) and keeps 600 frames
// of the copies beside the interval G holds, the present's display interval and G's measured frame ticks. Every 600 presents it
// compares each 32-bit word over the frames and logs the words that equal interval/60, interval/30, the interval itself, the
// display interval or the measured ticks in seconds in most of them. Only offsets, counts and ratios are logged, in game.log under
// build/ with the scenario logs. Plain reads of guest memory from the presenting thread: a word the game writes meanwhile
// can be torn for one frame, which the ratios tolerate.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"

#define FRAMES 600
#define REGIONS 25
#define RSIZE 0x300u
#define GSIZE 0x240u
#define RET_AFTER_SET 0x70001eaef0ull   // main+0x1eaeec bl nvnWindowSetPresentInterval wrapper, plus 4
#define TICK_FREQ 0x700231a4d0ull       // the counter frequency the controller divides by

static int enabled_flag = -1;
static uint64_t G;                                  // the game's window object
static uint64_t rbase[REGIONS], rsrc[REGIONS];      // region addresses; the G offset a pointer came from (G itself: 0)
static int nreg;
static uint32_t snap[FRAMES][REGIONS][RSIZE / 4];
static uint8_t ivs[FRAMES];
static double wall_ms[FRAMES], tick_s[FRAMES];
static int nframes;
static uint64_t last_display;
static PlatMutex lk = PLAT_MUTEX_INIT;

static int enabled(void) {
    if (enabled_flag < 0) {
        const char *e = cfg_get("DKB_DELTA_SCAN");
        enabled_flag = e && !strcmp(e, "1") && !rt_game_version;
        if (enabled_flag) rt_log("delta scan: on (DKB_DELTA_SCAN=1); a summary every %d presents", FRAMES);
    }
    return enabled_flag;
}

static int readable(uint64_t a, uint64_t n) { return a >= 0x10000 && a + n < RT_ADDRESS_LIMIT && plat_mem_readable(a, n); }

// The caller of nvnWindowSetPresentInterval at main+0x1eaeec has the game's window object in x20 (callee saved).
void delta_scan_note_window(const Cpu *c, uint64_t nvn_window) {
    if (!enabled() || c->x[30] != RET_AFTER_SET) return;
    uint64_t g = c->x[20];
    if (!readable(g, GSIZE)) return;
    uint64_t w, n;
    memcpy(&w, (const void *)(uintptr_t)(g + 0x150), 8);   // the object holds the C++ nvn::Window at +0x150 (main+0x1eaee4)...
    if (!readable(w, 0x78)) return;
    memcpy(&n, (const void *)(uintptr_t)(w + 0x70), 8);    // ...whose NVNwindow is at +0x70 (main+0x1eaf10)
    if (n != nvn_window) return;
    plat_mutex_lock(&lk);
    if (G != g) rt_log("delta scan: the game's window object found (+0x150 -> nvn::Window, +0x70 -> NVNwindow)");
    G = g;
    plat_mutex_unlock(&lk);
}

static void pick_regions(void) {
    nreg = 0;
    rbase[nreg] = G, rsrc[nreg++] = 0;
    for (uint64_t off = 0; off + 8 <= GSIZE && nreg < REGIONS; off += 8) {
        uint64_t v;
        memcpy(&v, (const void *)(uintptr_t)(G + off), 8);
        if (v < 0x100000 || (v & 7) || v >= RT_ADDRESS_LIMIT || (v >= 0x7000000000ull && v < 0x7010000000ull)) continue;
        if (v == G || !readable(v, RSIZE)) continue;
        int dup = 0;
        for (int i = 0; i < nreg; i++) dup |= rbase[i] == v;
        if (!dup) rbase[nreg] = v, rsrc[nreg++] = off;
    }
}

static float fl(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

static void analyse(void) {
    int cnt[5];
    int logged = 0;
    for (int r = 0; r < nreg; r++)
        for (int w = 0; w < (int)(RSIZE / 4); w++) {
            memset(cnt, 0, sizeof cnt);
            int distinct = 0;
            uint32_t first = snap[0][r][w];
            for (int t = 0; t < nframes; t++) {
                uint32_t b = snap[t][r][w];
                distinct |= b != first;
                float f = fl(b);
                int iv = ivs[t];
                if (iv >= 1 && iv <= 3 && fabsf(f - (float)iv / 60.0f) < 2e-4f) cnt[0]++;
                if (iv >= 1 && iv <= 3 && fabsf(f - (float)iv / 30.0f) < 2e-4f) cnt[1]++;
                if (iv >= 1 && iv <= 3 && b == (uint32_t)iv) cnt[2]++;
                if (wall_ms[t] > 4.0 && fabs((double)f * 1000.0 - wall_ms[t]) < 1.5) cnt[3]++;
                if (tick_s[t] > 0.004 && fabs((double)f - tick_s[t]) < 0.0015) cnt[4]++;
            }
            static const char *const what[5] = {"interval/60 s", "interval/30 s", "the interval as an int", "the display interval in s",
                                                "the measured frame ticks in s"};
            for (int k = 0; k < 5; k++)
                if (distinct && cnt[k] >= nframes * 7 / 10 && cnt[k] >= 8 && logged < 60) {
                    logged++;
                    uint64_t vt = 0;
                    memcpy(&vt, (const void *)(uintptr_t)rbase[r], 8);   // the object's vtable, as a main offset when it is in main
                    rt_log("delta scan: %s +%#x (word +%#x) equals %s in %d of %d frames; vtable %#llx", r ? "object at G" : "G",
                           r ? (unsigned)rsrc[r] : 0u, (unsigned)(w * 4), what[k], cnt[k], nframes,
                           (unsigned long long)(vt >= 0x7000000000ull ? vt - 0x7000000000ull : vt));
                }
        }
    int hist[4] = {0};
    for (int t = 0; t < nframes; t++) hist[ivs[t] > 3 ? 0 : ivs[t]]++;
    rt_log("delta scan: %d frames, intervals 1/2/3 %d/%d/%d (other %d), %d objects scanned, %d matches logged", nframes, hist[1], hist[2], hist[3],
           hist[0], nreg, logged);
}


// ---- second scan: the singletons the game keeps in its data and bss. A step could live in an object no pointer of G reaches, so the
// 8-byte values of main's data and bss that point into the heap name objects (0x400 bytes each are looked at), and every word of
// those objects and of the data itself whose float lies in 12.5 .. 60 ms is a candidate (at most MAXC). Their values are
// read at each present for FRAMES2 presents and compared like the words above, plus with the display interval and with 1/60 steps.
#define MAXC 20000
#define FRAMES2 600
#define DATA_LO 0x7002100000ull
#define DATA_HI 0x7002450000ull
static uint64_t cand[MAXC];
static int ncand;
static float *cval;   // [FRAMES2 * MAXC]
static int c_frames;
static uint8_t c_iv[FRAMES2];
static double c_wall[FRAMES2];

static int plausible(uint32_t b) {
    float f = fl(b);
    return f >= 0.0125f && f <= 0.06f;
}

static void add_words(uint64_t base, uint32_t bytes) {
    uint32_t buf[0x100];
    for (uint32_t o = 0; o < bytes && ncand < MAXC; o += sizeof buf) {
        uint32_t n = bytes - o < sizeof buf ? bytes - o : (uint32_t)sizeof buf;
        if (!readable(base + o, n)) break;
        memcpy(buf, (const void *)(uintptr_t)(base + o), n);
        for (uint32_t i = 0; i < n / 4 && ncand < MAXC; i++)
            if (plausible(buf[i])) cand[ncand++] = base + o + i * 4;
    }
}

static void build_candidates(void) {
    ncand = 0;
    if (!cval) cval = malloc(sizeof(float) * (size_t)FRAMES2 * MAXC);
    if (!cval) return;
    static uint64_t seen[4096];
    int nseen = 0;
    add_words(DATA_LO, (uint32_t)(DATA_HI - DATA_LO));
    for (uint64_t a = DATA_LO; a + 8 <= DATA_HI && ncand < MAXC; a += 8) {
        uint64_t v;
        memcpy(&v, (const void *)(uintptr_t)a, 8);
        if (v < 0x100000000ull || (v & 7) || v >= RT_ADDRESS_LIMIT || (v >= 0x7000000000ull && v < 0x7010000000ull)) continue;
        int dup = 0;
        for (int i = 0; i < nseen && !dup; i++) dup = seen[i] == v;
        if (dup) continue;
        if (nseen < 4096) seen[nseen++] = v;
        if (readable(v, 0x400)) add_words(v, 0x400);
    }
    rt_log("delta scan: %d candidate words (floats of 12.5 to 60 ms) in main's data and %d objects it points to", ncand, nseen);
}

static void sample_candidates(uint32_t iv, double wall) {
    static int warm;
    if (warm < 900) { warm++; return; }
    if (c_frames == 0) build_candidates();
    if (!cval || !ncand) return;
    for (int i = 0; i < ncand; i++) {
        uint32_t b = 0;
        if (readable(cand[i], 4)) memcpy(&b, (const void *)(uintptr_t)cand[i], 4);
        cval[(size_t)c_frames * MAXC + i] = fl(b);
    }
    c_iv[c_frames] = iv > 255 ? 255 : (uint8_t)iv;
    c_wall[c_frames] = wall;
    if (++c_frames < FRAMES2) return;
    int hist[4] = {0}, logged = 0;
    for (int t = 0; t < FRAMES2; t++) hist[c_iv[t] > 3 ? 0 : c_iv[t]]++;
    for (int i = 0; i < ncand && logged < 40; i++) {
        int m[3] = {0}, distinct = 0;
        float first = cval[i];
        for (int t = 0; t < FRAMES2; t++) {
            float f = cval[(size_t)t * MAXC + i];
            distinct |= f != first;
            int v = c_iv[t];
            if (v >= 1 && v <= 3 && fabsf(f - (float)v / 60.0f) < 2e-4f) m[0]++;
            if (c_wall[t] > 4.0 && fabs((double)f * 1000.0 - c_wall[t]) < 1.5) m[1]++;
            if (v >= 1 && v <= 3 && fabsf(f - (float)v / 30.0f) < 2e-4f) m[2]++;
        }
        static const char *const what[3] = {"interval/60 s", "the display interval in s", "interval/30 s"};
        for (int k = 0; k < 3; k++)
            if (distinct && m[k] >= FRAMES2 / 2) {
                logged++;
                rt_log("delta scan: word at %#llx equals %s in %d of %d frames", (unsigned long long)cand[i], what[k], m[k], FRAMES2);
            }
    }
    rt_log("delta scan: data/heap candidates: %d words over %d frames, intervals 1/2/3 %d/%d/%d, %d matches logged", ncand, FRAMES2, hist[1],
           hist[2], hist[3], logged);
    c_frames = 0;
}

void delta_scan_present(uint64_t display_ns) {
    if (!enabled() || !G) return;
    plat_mutex_lock(&lk);
    if (!readable(G, GSIZE)) {
        plat_mutex_unlock(&lk);
        return;
    }
    if (nframes == 0) pick_regions();
    uint32_t iv;
    uint64_t e8, freq = 0;
    memcpy(&iv, (const void *)(uintptr_t)(G + 0x88), 4);
    memcpy(&e8, (const void *)(uintptr_t)(G + 0xe8), 8);
    if (readable(TICK_FREQ, 8)) memcpy(&freq, (const void *)(uintptr_t)TICK_FREQ, 8);
    for (int r = 0; r < nreg; r++) {
        if (readable(rbase[r], RSIZE)) memcpy(snap[nframes][r], (const void *)(uintptr_t)rbase[r], RSIZE);
        else memset(snap[nframes][r], 0, RSIZE);
    }
    ivs[nframes] = iv > 255 ? 255 : (uint8_t)iv;
    wall_ms[nframes] = last_display ? (double)(display_ns - last_display) / 1e6 : 0.0;
    tick_s[nframes] = freq ? (double)e8 / (double)freq : 0.0;
    last_display = display_ns;
    sample_candidates(ivs[nframes], wall_ms[nframes]);
    if (++nframes == FRAMES) {
        analyse();
        nframes = 0;
    }
    plat_mutex_unlock(&lk);
}

// ---- DKB_DELTA_PROBE=1 (patch set deltaprobe, recomp/patches-dkb100.txt; default off): who asks the game's frame object for its
// measured frame rate. main+0x138bb30 is the framework's virtual getter (slot 0xa8): freq / [G+0xe8], the previous frame's ticks. It is
// only reached through the vtable, so its first instruction is replaced by `mov x17, x30; bl <page+0xfa0>`: the host function logs the
// guest caller (x17), the value the getter returns and G's interval, gives the getter's own first instruction back (adrp x8) and
// restores lr. The call of main+0x1eaf30 at main+0x1ead28 (the controller storing fps = 60 / interval at G+0xe0) goes to
// <page+0xfb0>, which logs the fps and calls the original. Per caller and 60 presents: calls and the range of the value. Counts and
// values only.
#if __has_include("patches.h")
#include "patches.h"
#endif
#ifdef RT_PATCH_PAGE
#define PSLOTS 16
typedef struct { uint64_t lr, calls; double lo, hi, sum; } PSite;
static PSite psites[PSLOTS];
static uint64_t ctrl_calls, ctrl_lo = ~0ull, ctrl_hi;
static PlatMutex plk = PLAT_MUTEX_INIT;

static void probe_getter(Cpu *c) {
    uint64_t g = c->x[0], lr = c->x[17], ticks = 0, freq = 0;
    if (readable(g + 0xe8, 8)) memcpy(&ticks, (const void *)(uintptr_t)(g + 0xe8), 8);
    if (readable(TICK_FREQ, 8)) memcpy(&freq, (const void *)(uintptr_t)TICK_FREQ, 8);
    double v = ticks ? (double)(float)freq / (double)(float)ticks : 0.0;
    plat_mutex_lock(&plk);
    PSite *s = NULL;
    for (int i = 0; i < PSLOTS && !s; i++)
        if (psites[i].lr == lr || !psites[i].lr) s = &psites[i];
    if (s) {
        if (!s->lr) *s = (PSite){.lr = lr, .lo = 1e30};
        s->calls++, s->sum += v;
        if (v < s->lo) s->lo = v;
        if (v > s->hi) s->hi = v;
    }
    plat_mutex_unlock(&plk);
    c->x[8] = 0x700231a000ull;   // the replaced first instruction: adrp x8, 0x700231a000
    c->x[30] = lr;               // the getter returns to its caller
}

static void probe_ctrl(Cpu *c) {
    uint64_t fps = (uint32_t)c->x[1];
    plat_mutex_lock(&plk);
    ctrl_calls++;
    if (fps < ctrl_lo) ctrl_lo = fps;
    if (fps > ctrl_hi) ctrl_hi = fps;
    plat_mutex_unlock(&plk);
    uint64_t g = c->x[0];
    rt_call(c, 0x70001eaf30ull, 2, c->x[0], c->x[1]);
    static int selftest;
    if (!selftest++) {   // proves the getter's stub works: the guest getter called once from the host shows up as a "getter" line
        rt_call(c, 0x700138bb30ull, 1, g);
        rt_log("delta probe: self-test called the getter once from the host");
    }
}


// Frame context handed to the framework's task objects: main+0x1e9f10 (before [G+0xf8]->slot 0x28) and main+0x1e9f28 (before [G+0x170]->slot
// 0x30) are `mov x1, sp` whose stack holds the context (0x1c0 bytes) the update passes down. The site replaces that word with `mov x1,
// sp; bl <page+0xfc0/0xfd0>`; lr is saved in the frame, so the bl costs nothing. The host function compares every context word with
// the interval G holds ([G+0x88], x19) over the window of 60 presents.
#define CTXW 112
typedef struct {
    uint64_t calls, valid[CTXW], m60[CTXW], mi[CTXW], mfps[CTXW], mint[CTXW], changed[CTXW];
    float lo[CTXW], hi[CTXW], last[CTXW];
} Ctx;
static Ctx ctxs[2];

static void ctx_probe(Cpu *c, int k) {
    uint64_t sp = c->x[1], g = c->x[19];
    uint32_t w[CTXW], iv = 0;
    if (!readable(sp, sizeof w) || !readable(g + 0x88, 4)) return;
    memcpy(w, (const void *)(uintptr_t)sp, sizeof w);
    memcpy(&iv, (const void *)(uintptr_t)(g + 0x88), 4);
    if (iv < 1 || iv > 3) return;
    plat_mutex_lock(&plk);
    Ctx *x = &ctxs[k];
    for (int i = 0; i < CTXW; i++) {
        float f = fl(w[i]);
        if (x->calls == 0) x->lo[i] = 1e30f, x->hi[i] = -1e30f, x->last[i] = f;
        if (f == f && f > 1e-4f && f < 1.0f) {
            x->valid[i]++;
            if (f < x->lo[i]) x->lo[i] = f;
            if (f > x->hi[i]) x->hi[i] = f;
            if (fabsf(f - (float)iv / 60.0f) < 2e-4f) x->m60[i]++;
            if (fabsf(f - 1.0f / (float)iv) < 2e-3f) x->mi[i]++;
        }
        if (f == f && f > 1.0f && f < 1000.0f && fabsf(f - 60.0f / (float)iv) < 0.01f) x->mfps[i]++;
        if (w[i] == iv) x->mint[i]++;
        if (f != x->last[i]) x->changed[i]++;
        x->last[i] = f;
    }
    x->calls++;
    plat_mutex_unlock(&plk);
}
static void ctx_probe0(Cpu *c) { ctx_probe(c, 0); }
static void ctx_probe1(Cpu *c) { ctx_probe(c, 1); }

static void ctx_log(uint64_t n) {
    for (int k = 0; k < 2; k++) {
        Ctx *x = &ctxs[k];
        if (!x->calls) continue;
        int shown = 0;
        for (int i = 0; i < CTXW && shown < 24; i++) {
            uint64_t half = x->calls / 2;
            if (x->m60[i] >= half || x->mi[i] >= half || x->mfps[i] >= half || x->mint[i] >= half ||
                (x->valid[i] >= half && x->changed[i] > x->calls / 4)) {
                shown++;
                rt_log("delta probe %llu: ctx%d +%#x of %llu calls: ==iv/60 %llu, ==1/iv %llu, ==60/iv %llu, ==iv(int) %llu, float in (0,1) %llu "
                       "[%.5f..%.5f], changed %llu", (unsigned long long)n, k, i * 4, (unsigned long long)x->calls, (unsigned long long)x->m60[i],
                       (unsigned long long)x->mi[i], (unsigned long long)x->mfps[i], (unsigned long long)x->mint[i], (unsigned long long)x->valid[i],
                       x->lo[i], x->hi[i], (unsigned long long)x->changed[i]);
            }
        }
        if (!shown) rt_log("delta probe %llu: ctx%d %llu calls, no word follows the interval or changes", (unsigned long long)n, k,
                           (unsigned long long)x->calls);
        memset(x, 0, sizeof *x);
    }
}


// Readers of the controller's fps (the integer 60 / interval the controller stores at +0x50, main+0x1eadf8). Each `ldr s, [xN, #0x50]` that
// converts it to a float is followed by `bl <page+0xe00+0x10*k>`; the host function counts the calls per 60 presents and the range of the
// integer, beside the interval G holds (the match count tells whether the value is the controller's).
#define NFPS 8
static const struct { uint64_t pc; int reg; } fps_site[NFPS] = {
    {0xf45f8, 8}, {0xf4930, 9}, {0x2da110, 9}, {0x34ec0c, 8}, {0x34ec4c, 8}, {0x692e80, 24}, {0x1510f0c, 19}, {0x1642080, 8}};
static uint64_t fps_calls[NFPS], fps_lo[NFPS], fps_hi[NFPS], fps_match[NFPS];

static void fps_probe(Cpu *c, int k) {
    uint64_t base = c->x[fps_site[k].reg], iv = 0;
    uint32_t v = 0, gi = 0;
    if (!readable(base + 0x50, 4)) return;
    memcpy(&v, (const void *)(uintptr_t)(base + 0x50), 4);
    if (G && readable(G + 0x88, 4)) memcpy(&gi, (const void *)(uintptr_t)(G + 0x88), 4);
    iv = gi;
    plat_mutex_lock(&plk);
    if (!fps_calls[k]) fps_lo[k] = ~0ull, fps_hi[k] = 0;
    fps_calls[k]++;
    if (v < fps_lo[k]) fps_lo[k] = v;
    if (v > fps_hi[k]) fps_hi[k] = v;
    if (iv && v == 60 / iv) fps_match[k]++;
    plat_mutex_unlock(&plk);
}
#define FPSSLOT(K) static void fps_probe##K(Cpu *c) { fps_probe(c, K); }
FPSSLOT(0) FPSSLOT(1) FPSSLOT(2) FPSSLOT(3) FPSSLOT(4) FPSSLOT(5) FPSSLOT(6) FPSSLOT(7)

static void fps_log(uint64_t n) {
    plat_mutex_lock(&plk);
    for (int k = 0; k < NFPS; k++) {
        if (!fps_calls[k]) continue;
        rt_log("delta probe %llu: fps reader main+%#llx: %llu calls, fps %llu..%llu, equals 60/interval in %llu", (unsigned long long)n,
               (unsigned long long)fps_site[k].pc, (unsigned long long)fps_calls[k], (unsigned long long)fps_lo[k],
               (unsigned long long)fps_hi[k], (unsigned long long)fps_match[k]);
        fps_calls[k] = 0, fps_match[k] = 0;
    }
    plat_mutex_unlock(&plk);
}


// The step the game really uses: main+0x2da128 stores it at [sp+8] before the 0x68-stride updater (main+0x2da5a0) is called. The site
// keeps the store and adds `bl <page+0xe80>`; the host function sums the steps per window of 60 presents and the log compares the sum with
// the wall time of the window: the game's speed.
static uint64_t st_calls, st_t0;
static double st_sum, st_lo = 1e30, st_hi;
static void step_probe(Cpu *c) {
    float f = 0;
    if (!readable(c->sp + 8, 4)) return;
    memcpy(&f, (const void *)(uintptr_t)(c->sp + 8), 4);
    plat_mutex_lock(&plk);
    if (!st_t0) st_t0 = kern_now_ns();
    st_calls++, st_sum += f;
    if (f < st_lo) st_lo = f;
    if (f > st_hi) st_hi = f;
    plat_mutex_unlock(&plk);
}
static void step_log(uint64_t n) {
    plat_mutex_lock(&plk);
    if (st_calls) {
        uint64_t now = kern_now_ns();
        double wall = (double)(now - st_t0) * 1e-9;
        rt_log("delta probe %llu: step passed to the updater: %llu calls, %.4f..%.4f s each, %.3f s stepped in %.3f s wall (speed %.3f)",
               (unsigned long long)n, (unsigned long long)st_calls, st_lo, st_hi, st_sum, wall, wall > 0 ? st_sum / wall : 0.0);
        st_t0 = now, st_calls = 0, st_sum = 0, st_lo = 1e30, st_hi = 0;
    }
    plat_mutex_unlock(&plk);
}

static void probe_log(uint64_t n) {
    fps_log(n);
    step_log(n);
    plat_mutex_lock(&plk);
    for (int i = 0; i < PSLOTS; i++) {
        PSite *s = &psites[i];
        if (!s->lr || !s->calls) continue;
        rt_log("delta probe %llu: getter main+%#llx: %llu calls, measured fps %.2f..%.2f (mean %.2f)", (unsigned long long)n,
               (unsigned long long)(s->lr - 0x7000000000ull), (unsigned long long)s->calls, s->lo, s->hi, s->sum / (double)s->calls);
        s->calls = 0, s->sum = 0, s->lo = 1e30, s->hi = 0;
    }
    if (ctrl_calls) rt_log("delta probe %llu: controller stored fps %llu..%llu, %llu times", (unsigned long long)n,
                           (unsigned long long)ctrl_lo, (unsigned long long)ctrl_hi, (unsigned long long)ctrl_calls);
    ctrl_calls = 0, ctrl_lo = ~0ull, ctrl_hi = 0;
    plat_mutex_unlock(&plk);
}

__attribute__((constructor)) static void delta_probe_register(void) {
    rt_register_host(RT_PATCH_PAGE + 0xfa0, probe_getter);
    rt_register_host(RT_PATCH_PAGE + 0xfb0, probe_ctrl);
    rt_register_host(RT_PATCH_PAGE + 0xfc0, ctx_probe0);
    rt_register_host(RT_PATCH_PAGE + 0xfd0, ctx_probe1);
    static void (*const fp[NFPS])(Cpu *) = {fps_probe0, fps_probe1, fps_probe2, fps_probe3, fps_probe4, fps_probe5, fps_probe6, fps_probe7};
    rt_register_host(RT_PATCH_PAGE + 0xe80, step_probe);
    for (int k = 0; k < NFPS; k++) rt_register_host(RT_PATCH_PAGE + 0xe00 + (unsigned)k * 0x10, fp[k]);
}
#else
static void probe_log(uint64_t n) { (void)n; }
#endif

void delta_probe_present(void) {
    static uint64_t n;
    if (++n % 60 == 0) probe_log(n);
}
