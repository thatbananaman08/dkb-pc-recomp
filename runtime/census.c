// DKB_CENSUS / DKB_DRAWSPLIT measurement counters: see census.h. Nothing here changes what the game or the replay does.
#include "census.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "platform/platform.h"

int census_flag = -1, split_flag = -1, split_sampling;

int census_read(void) {
    const char *e = cfg_get("DKB_CENSUS");
    census_flag = e && !strcmp(e, "1");
    if (census_flag) rt_log("census: on (DKB_CENSUS=1); lines every 60 presents");
    return census_flag;
}
int split_read(void) {
    const char *e = cfg_get("DKB_DRAWSPLIT");
    split_flag = e && !strcmp(e, "1");
    if (split_flag) rt_log("census: draw split on (DKB_DRAWSPLIT=1), every 8th draw timed with the cycle counter");
    return split_flag;
}

static uint64_t mix_inl(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    return x;
}

// ---------- R0
#define R0_THREADS 64
typedef struct {
    uint64_t tp;   // guest thread (TPIDR_EL0)
    uint64_t recs, ops, draws, span_ns, inl_calls, inl_bytes, max_ops;
    uint64_t begin_ns, cur_ops, first_ns, last_ns;
} R0T;
static R0T r0t[R0_THREADS];
static int r0n;
static _Thread_local R0T *r0me;
static PlatMutex r0_lock = PLAT_MUTEX_INIT;

static R0T *r0_self(uint64_t tp) {
    if (r0me && r0me->tp == tp) return r0me;
    plat_mutex_lock(&r0_lock);
    R0T *t = NULL;
    for (int i = 0; i < r0n; i++)
        if (r0t[i].tp == tp) t = &r0t[i];
    if (!t && r0n < R0_THREADS) t = &r0t[r0n++], t->tp = tp;
    plat_mutex_unlock(&r0_lock);
    r0me = t;
    return t;
}
void r0_begin(uint64_t tp) {
    R0T *t = r0_self(tp);
    if (!t) return;
    t->begin_ns = kern_now_ns();
    if (!t->first_ns) t->first_ns = t->begin_ns;
    t->cur_ops = 0;
}
void r0_op(int draw) {
    R0T *t = r0me;
    if (!t) return;
    t->ops++, t->cur_ops++;
    t->draws += draw != 0;
}
void r0_end(void) {
    R0T *t = r0me;
    if (!t || !t->begin_ns) return;
    uint64_t now = kern_now_ns();
    t->span_ns += now - t->begin_ns, t->recs++, t->last_ns = now, t->begin_ns = 0;
    if (t->cur_ops > t->max_ops) t->max_ops = t->cur_ops;
}
// Draws inside called recordings: CallCommands copies a recording the game made earlier. A handle first seen in this window is a recording made
// recently, one seen in an earlier window was made before (a copy address can be reused after the game frees a recording: approximate).
static struct { uint64_t key; uint64_t win; } inl_tab[1 << 16];
static uint64_t inl_win = 1, inl_calls_new, inl_calls_old, inl_draws_new, inl_draws_old, inl_ops_new, inl_ops_old, inl_n;
void r0_inline(uint64_t handle, const void *body, uint64_t bytes) {
    R0T *t = r0me;
    if (t) t->inl_calls++, t->inl_bytes += bytes;
    uint64_t draws = 0, ops = 0;
    const uint8_t *p = body, *end = p + bytes;
    while (p + sizeof(op_hdr) <= end) {
        op_hdr h;
        memcpy(&h, p, sizeof h);
        if (h.size < 8) break;
        ops++;
        draws += h.op == OP_DRAW || h.op == OP_DRAW_IDX || h.op == OP_DRAW_IDX_IND;
        p += h.size;
    }
    plat_mutex_lock(&r0_lock);
    uint32_t i = (uint32_t)mix_inl(handle) & ((1 << 16) - 1);
    int old = 0;
    for (int n = 0; n < (1 << 16); n++, i = (i + 1) & ((1 << 16) - 1)) {
        if (inl_tab[i].key == handle + 1) {
            old = inl_tab[i].win < inl_win;   // the first-seen window stays
            break;
        }
        if (!inl_tab[i].key) {
            inl_tab[i].key = handle + 1, inl_tab[i].win = inl_win, inl_n++;
            break;
        }
    }
    if (old) inl_calls_old++, inl_draws_old += draws, inl_ops_old += ops;
    else inl_calls_new++, inl_draws_new += draws, inl_ops_new += ops;
    plat_mutex_unlock(&r0_lock);
}

// ---------- replay census (one replay thread at a time)
typedef struct { uint64_t key; uint32_t last; } Ent;
#define NENT 65536
static Ent tab[NENT], tab2[NENT];
static uint32_t tab_n;
static uint32_t cf = 1;   // census frame
static uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ x >> 33;
}

// 0: not seen this frame nor the frame before (new), 1: seen the frame before, 2: seen this frame already; the entry is stamped
static int touch(uint64_t key, int *fresh) {
    uint64_t k = key + 1;
    uint32_t i = (uint32_t)mix(k) & (NENT - 1);
    for (int n = 0; n < NENT; n++, i = (i + 1) & (NENT - 1)) {
        if (tab[i].key == k) {
            uint32_t l = tab[i].last;
            tab[i].last = cf;
            *fresh = 0;
            return l == cf ? 2 : l + 1 == cf ? 1 : 0;
        }
        if (!tab[i].key) {
            tab[i].key = k, tab[i].last = cf;
            tab_n++;
            *fresh = 1;
            return 0;
        }
    }
    *fresh = 0;
    return 2;
}

typedef struct {
    uint64_t draws, nbind, ndesc;
    uint64_t tex_use[2], tex_dist[2], tex_new[2], tex_fresh[2], tex_chg[2];   // [0] sampled, [1] storage image
    uint64_t tid_dist, tid_new, sid_dist, sid_new;
    uint64_t buf_use[2], buf_dist[2], buf_new[2], buf_chg[2];                  // [0] uniform, [1] storage
    uint64_t desc[7];
    uint64_t ops[256];
} Fr;
static Fr fr, win;
static uint64_t win_frames;
static uint64_t last_h[8][2][132], last_b[2][8][20];

void census_op(uint32_t op) {
    if (op < 256) fr.ops[op]++;
}
void census_draw(unsigned nbind, unsigned ndesc) {
    fr.draws++, fr.nbind += nbind, fr.ndesc += ndesc;
}
void census_tex(int st, int slot, int img, uint64_t h) {
    int fresh, u = img != 0;
    if (st < 0 || st >= 8 || slot < 0 || slot >= 132) return;
    fr.tex_use[u]++;
    int s = touch(h | (uint64_t)u << 62, &fresh);
    if (s != 2) {
        fr.tex_dist[u]++;
        if (s == 0) fr.tex_new[u]++;
        if (fresh) fr.tex_fresh[u]++;
    }
    if (!u) {
        s = touch((h & 0xfffff) | 1ull << 61, &fresh);
        if (s != 2) fr.tid_dist++, fr.tid_new += s == 0;
        s = touch(((h >> 20) & 0xfff) | 2ull << 61, &fresh);
        if (s != 2) fr.sid_dist++, fr.sid_new += s == 0;
    }
    if (last_h[st][u][slot] != h + 1) fr.tex_chg[u]++, last_h[st][u][slot] = h + 1;
}
void census_buf(int ssbo, int st, int slot, uint64_t key, uint64_t range) {
    int fresh, b = ssbo != 0;
    (void)range;
    if (st < 0 || st >= 8 || slot < 0 || slot >= 20) return;
    fr.buf_use[b]++;
    int s = touch((mix(key) & 0x0fffffffffffffffull) | (uint64_t)(3 + b) << 60, &fresh);
    if (s != 2) {
        fr.buf_dist[b]++;
        if (s == 0) fr.buf_new[b]++;
    }
    if (last_b[b][st][slot] != key + 1) fr.buf_chg[b]++, last_b[b][st][slot] = key + 1;
}
void census_desc(int what) {
    if (what >= 0 && what < 7) fr.desc[what]++;
}

// ---------- draw split
static uint64_t sp_acc[SP_N], sp_n, sp_all;
void split_begin_draw(void) {
    sp_all++;
    split_sampling = (sp_all & 7) == 0;
    if (split_sampling) sp_n++;
}
void split_end_draw(void) { split_sampling = 0; }
void split_acc(int k, uint64_t cycles) {
    if (k >= 0 && k < SP_N) sp_acc[k] += cycles;
}

// ---------- logging
static uint64_t ent0_ns, ent0_tsc;
static void add_win(void) {
    uint64_t *a = (uint64_t *)&win, *b = (uint64_t *)&fr;
    for (size_t i = 0; i < sizeof(Fr) / 8; i++) a[i] += b[i];
    memset(&fr, 0, sizeof fr);
    win_frames++;
}

static void log_r0(uint64_t frame, double nf) {
    uint64_t tot_ops = 0, tot_draws = 0;
    plat_mutex_lock(&r0_lock);
    for (int i = 0; i < r0n; i++) {
        R0T *t = &r0t[i];
        if (!t->ops && !t->recs && !t->inl_calls) continue;
        rt_log("census r0 %llu: thread tp %#llx: %.1f recordings/frame, %.0f ops/frame, %.0f draws/frame, between Begin and End %.2f ms/frame "
               "(%.3f ms per recording, most ops in one recording %llu); first to last recording %.0f ms of the window; inlined CallCommands %.1f/frame (%.1f KiB)",
               (unsigned long long)frame, (unsigned long long)t->tp, t->recs / nf, t->ops / nf, t->draws / nf, t->span_ns / 1e6 / nf,
               t->recs ? t->span_ns / 1e6 / (double)t->recs : 0.0, (unsigned long long)t->max_ops,
               t->last_ns > t->first_ns ? (t->last_ns - t->first_ns) / 1e6 : 0.0, t->inl_calls / nf, t->inl_bytes / 1024.0 / nf);
        tot_ops += t->ops, tot_draws += t->draws;
        t->recs = t->ops = t->draws = t->span_ns = t->inl_calls = t->inl_bytes = t->max_ops = t->first_ns = t->last_ns = 0;
    }
    plat_mutex_unlock(&r0_lock);
    if (tot_ops) rt_log("census r0 %llu: all threads %.0f ops/frame, %.0f draws/frame recorded", (unsigned long long)frame, tot_ops / nf, tot_draws / nf);
    plat_mutex_lock(&r0_lock);
    rt_log("census r0 %llu: CallCommands copies per frame: %.0f of recordings first seen in this window (%.0f ops, %.0f draws), %.0f of recordings seen "
           "in an earlier window (%.0f ops, %.0f draws)", (unsigned long long)frame, inl_calls_new / nf, inl_ops_new / nf, inl_draws_new / nf,
           inl_calls_old / nf, inl_ops_old / nf, inl_draws_old / nf);
    inl_calls_new = inl_calls_old = inl_draws_new = inl_draws_old = inl_ops_new = inl_ops_old = 0;
    inl_win++;
    if (inl_n > 30000) memset(inl_tab, 0, sizeof inl_tab), inl_n = 0;   // handles are forgotten: the next window counts them all as new
    plat_mutex_unlock(&r0_lock);
}

void census_frame(uint64_t frame) {
    static int first = 1;
    int cen = census_on(), spl = split_on();
    if (!cen && !spl) return;
    if (first) {
        first = 0;
        ent0_ns = kern_now_ns(), ent0_tsc = split_tsc();
    }
    if (cen) {
        add_win();
        cf++;
        if (tab_n > NENT / 4) {   // forget what was not seen this or the previous frame ("never seen before" then means: not recently)
            memcpy(tab2, tab, sizeof tab);
            memset(tab, 0, sizeof tab);
            tab_n = 0;
            for (uint32_t j = 0; j < NENT; j++) {
                if (!tab2[j].key || tab2[j].last + 2 < cf) continue;
                uint32_t i = (uint32_t)mix(tab2[j].key) & (NENT - 1);
                while (tab[i].key) i = (i + 1) & (NENT - 1);
                tab[i] = tab2[j], tab_n++;
            }
        }
    }
    if (frame % 60) return;
    double nf = (double)(win_frames ? win_frames : 1);
    if (cen) {
        Fr *w = &win;
        double d = (double)(w->draws ? w->draws : 1);
        rt_log("census %llu: per frame: draws %.0f, bindings/draw %.1f, descriptors/draw %.1f; replay ops: bind tex %.0f, bind samp %.0f, bind img %.0f, "
               "bind ubo %.0f, bind ssbo %.0f, program %.0f, state %.0f, push const %.0f",
               (unsigned long long)frame, w->draws / nf, w->nbind / d, w->ndesc / d, w->ops[OP_BIND_TEX] / nf, w->ops[OP_BIND_SAMP] / nf, w->ops[OP_BIND_IMG] / nf,
               w->ops[OP_BIND_UBO] / nf, w->ops[OP_BIND_SSBO] / nf, w->ops[OP_BIND_PROGRAM] / nf, w->ops[OP_BIND_STATE] / nf, w->ops[OP_PUSH_CONST] / nf);
        rt_log("census %llu: textures per frame: %.0f sampled-handle lookups (%.1f/draw), %.0f distinct handles, %.0f not seen the previous frame "
               "(%.0f never seen before), slot changed against the previous lookup of the slot %.0f (%.0f%%); texture ids %.0f distinct (%.0f new), "
               "sampler ids %.0f distinct (%.0f new); storage images %.0f lookups, %.0f distinct",
               (unsigned long long)frame, w->tex_use[0] / nf, w->tex_use[0] / d, w->tex_dist[0] / nf, w->tex_new[0] / nf, w->tex_fresh[0] / nf,
               w->tex_chg[0] / nf, w->tex_use[0] ? 100.0 * (double)w->tex_chg[0] / (double)w->tex_use[0] : 0.0, w->tid_dist / nf, w->tid_new / nf,
               w->sid_dist / nf, w->sid_new / nf, w->tex_use[1] / nf, w->tex_dist[1] / nf);
        rt_log("census %llu: buffers per frame: uniform %.0f bindings (%.1f/draw), %.0f distinct ranges, %.0f new, %.0f (%.0f%%) differ from the slot previous "
               "range; storage %.0f bindings (%.1f/draw), %.0f distinct, %.0f new, %.0f (%.0f%%) differ",
               (unsigned long long)frame, w->buf_use[0] / nf, w->buf_use[0] / d, w->buf_dist[0] / nf, w->buf_new[0] / nf, w->buf_chg[0] / nf,
               w->buf_use[0] ? 100.0 * (double)w->buf_chg[0] / (double)w->buf_use[0] : 0.0, w->buf_use[1] / nf, w->buf_use[1] / d, w->buf_dist[1] / nf,
               w->buf_new[1] / nf, w->buf_chg[1] / nf, w->buf_use[1] ? 100.0 * (double)w->buf_chg[1] / (double)w->buf_use[1] : 0.0);
        rt_log("census %llu: descriptors per frame: pushes %.0f (skipped %.0f), sets allocated %.0f (reused %.0f), pools created %.2f, spare pools taken %.2f, "
               "whole bind skipped %.0f",
               (unsigned long long)frame, w->desc[0] / nf, w->desc[1] / nf, w->desc[2] / nf, w->desc[3] / nf, w->desc[4] / nf, w->desc[5] / nf, w->desc[6] / nf);
        memset(&win, 0, sizeof win);
        win_frames = 0;
        log_r0(frame, nf);
    }
    if (spl && sp_n) {
        uint64_t now = kern_now_ns(), tsc = split_tsc();
        double ns_per = (double)(now - ent0_ns) / (double)(tsc - ent0_tsc ? tsc - ent0_tsc : 1);
        static const char *const nm[SP_N] = {"prep (pass begin, shaders, pipeline lookup)", "bind pipeline + dynamic + fixed state", "bind_resources",
                                             "  signature check", "  uniform/storage/support buffers", "  texture lookups", "  push/set emit",
                                             "vertex buffers", "post (feedback, queries)", "whole draw op", "    emit: record buffers", "    emit: record textures", "    emit: scratch+push"};
        char line[1200];
        int o = snprintf(line, sizeof line, "census split %llu: %llu draws sampled of %llu; ns per draw:", (unsigned long long)frame,
                         (unsigned long long)sp_n, (unsigned long long)sp_all);
        for (int k = 0; k < SP_N && o < (int)sizeof line - 80; k++)
            o += snprintf(line + o, sizeof line - (size_t)o, " [%s %.0f]", nm[k], (double)sp_acc[k] * ns_per / (double)sp_n);
        rt_log("%s", line);
        memset(sp_acc, 0, sizeof sp_acc);
        sp_n = sp_all = 0;
    }
}
