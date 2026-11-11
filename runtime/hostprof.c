// TOTK_HOSTPROF: see hostprof.h.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "hostprof.h"

int g_hostprof;
_Thread_local uint64_t hp_wait_ns;
_Thread_local int hp_in_core;

#define NHOST 8192
#define NSVC 256
#define NOBJ 256

typedef struct { uint64_t calls, ns, max, wait; } Ent;
static Ent host[NHOST], svcs[NSVC], scopes[HPS_N], waits[HPW_N];             // this window
static uint64_t fr_host[NHOST], fr_svc[NSVC], fr_scope[HPS_N], fr_wait;       // this frame (ns)
typedef struct { uint64_t key, n, ns, max; uint32_t svc; } Obj;
static Obj objs[NOBJ];
static uint64_t sync_ns, sync_n;
static PlatMutex obj_lock = PLAT_MUTEX_INIT;

static const char *const wait_names[HPW_N] = {"scheduler lock", "virtual core retake", "frz_lock (mem_thaw)", "memory.c lock",
                                              "romfs.c lock", "fs_host.c locks"};
static const char *const scope_names[HPS_N] = {"romfs_open", "romfs_read", "romfs_read pread", "mem_thaw"};

void hostprof_init(void) {
    const char *e = cfg_get("TOTK_HOSTPROF");
    g_hostprof = e && !strcmp(e, "1");
    if (g_hostprof) rt_log("hostprof: timing host functions, SVCs, host lock waits and guest waits (TOTK_TRACE=perf prints them)");
}

static void upd(Ent *e, uint64_t ns, uint64_t wait) {
    __atomic_add_fetch(&e->calls, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&e->ns, ns, __ATOMIC_RELAXED);
    __atomic_add_fetch(&e->wait, wait, __ATOMIC_RELAXED);
    if (ns > __atomic_load_n(&e->max, __ATOMIC_RELAXED)) __atomic_store_n(&e->max, ns, __ATOMIC_RELAXED);
}

void hp_wait_add(int kind, uint64_t ns) {
    if (kind < 0 || kind >= HPW_N) return;
    upd(&waits[kind], ns, 0);
    if (kind == HPW_CORE || !hp_in_core) hp_wait_ns += ns;   // the scheduler lock inside a core retake is part of it
    __atomic_add_fetch(&fr_wait, ns, __ATOMIC_RELAXED);
}

void hp_host_call(int slot, uint64_t ns, uint64_t wait_ns) {
    if (slot < 0 || slot >= NHOST) return;
    upd(&host[slot], ns, wait_ns);
    __atomic_add_fetch(&fr_host[slot], ns, __ATOMIC_RELAXED);
}

void hp_scope_add(int id, uint64_t ns, uint64_t wait_ns) {
    if (id < 0 || id >= HPS_N) return;
    upd(&scopes[id], ns, wait_ns);
    __atomic_add_fetch(&fr_scope[id], ns, __ATOMIC_RELAXED);
}

void hp_svc(uint32_t imm, uint64_t ns, uint64_t wait_ns, int sync, uint64_t obj) {
    imm &= 0xff;
    upd(&svcs[imm], ns, sync ? ns : wait_ns);
    __atomic_add_fetch(&fr_svc[imm], ns, __ATOMIC_RELAXED);
    if (!sync) return;
    plat_mutex_lock(&obj_lock);
    sync_ns += ns, sync_n++;
    uint64_t key = obj & 0x00ffffffffffffffull;
    unsigned h = (unsigned)((key * 0x9E3779B97F4A7C15ull ^ imm) >> 56) % NOBJ;
    for (unsigned i = 0; i < NOBJ; i++) {
        Obj *o = &objs[(h + i) % NOBJ];
        if (!o->n || (o->key == key && o->svc == imm)) {
            o->key = key, o->svc = imm, o->n++, o->ns += ns;
            if (ns > o->max) o->max = ns;
            break;
        }
    }
    plat_mutex_unlock(&obj_lock);
}

static const char *svc_name(uint32_t n, char *buf, size_t sz) {
    switch (n) {
    case 0x01: return "svc SetHeapSize";
    case 0x0B: return "svc SleepThread";
    case 0x18: return "svc WaitSynchronization";
    case 0x1A: return "svc ArbitrateLock";
    case 0x1B: return "svc ArbitrateUnlock";
    case 0x1C: return "svc WaitProcessWideKeyAtomic";
    case 0x1D: return "svc SignalProcessWideKey";
    case 0x21: return "svc SendSyncRequest";
    case 0x2C: return "svc MapPhysicalMemory";
    case 0x2D: return "svc UnmapPhysicalMemory";
    case 0x34: return "svc WaitForAddress";
    case 0x35: return "svc SignalToAddress";
    }
    snprintf(buf, sz, "svc %#x", n);
    return buf;
}

typedef struct { char name[96]; Ent e; } Row;

static int row_cmp(const void *a, const void *b) {
    const Row *x = a, *y = b;
    return x->e.ns < y->e.ns ? 1 : x->e.ns > y->e.ns ? -1 : 0;
}

static size_t gather(Row *rows, size_t cap) {
    size_t n = 0;
    for (int i = 0; i < NHOST && n < cap; i++)
        if (host[i].calls) {
            const char *nm = hostfn_name(RT_HOSTFN_BASE + 16ull * (uint64_t)i);
            snprintf(rows[n].name, sizeof rows[n].name, "%s", nm ? nm : "host ?");
            rows[n++].e = host[i];
        }
    for (int i = 0; i < NSVC && n < cap; i++)
        if (svcs[i].calls) {
            char b[48];
            snprintf(rows[n].name, sizeof rows[n].name, "%s", svc_name((uint32_t)i, b, sizeof b));
            rows[n++].e = svcs[i];
        }
    for (int i = 0; i < HPS_N && n < cap; i++)
        if (scopes[i].calls) {
            snprintf(rows[n].name, sizeof rows[n].name, "%s", scope_names[i]);
            rows[n++].e = scopes[i];
        }
    return n;
}

void hostprof_log(uint64_t frame) {
    if (!g_hostprof) return;
    static Row rows[NHOST + NSVC + HPS_N];
    size_t n = gather(rows, sizeof rows / sizeof *rows);
    qsort(rows, n, sizeof *rows, row_cmp);
    for (size_t i = 0; i < n && i < 10; i++) {
        const Ent *e = &rows[i].e;
        rt_log("hostprof frame %llu: #%zu %s: %llu calls, %.1f ms total, max %.2f ms, waiting %.1f ms (%.0f%%)", (unsigned long long)frame,
               i + 1, rows[i].name, (unsigned long long)e->calls, e->ns / 1e6, e->max / 1e6, e->wait / 1e6,
               e->ns ? 100.0 * (double)e->wait / (double)e->ns : 0.0);
    }
    char lk[512] = "";
    size_t o = 0;
    for (int k = 0; k < HPW_N && o < sizeof lk - 100; k++)
        if (waits[k].calls)
            o += (size_t)snprintf(lk + o, sizeof lk - o, "%s%s %.1f ms in %llu waits (max %.2f)", o ? "; " : "", wait_names[k],
                                  waits[k].ns / 1e6, (unsigned long long)waits[k].calls, waits[k].max / 1e6);
    rt_log("hostprof frame %llu: host waits: %s", (unsigned long long)frame, o ? lk : "none");
    plat_mutex_lock(&obj_lock);
    Obj top[5];
    memset(top, 0, sizeof top);
    for (int i = 0; i < NOBJ; i++) {
        if (!objs[i].n) continue;
        for (int k = 0; k < 5; k++)
            if (objs[i].ns > top[k].ns) {
                memmove(&top[k + 1], &top[k], (size_t)(4 - k) * sizeof top[0]);
                top[k] = objs[i];
                break;
            }
    }
    char gs[640] = "";
    o = 0;
    for (int k = 0; k < 5 && top[k].n && o < sizeof gs - 120; k++) {
        char b[48];
        o += (size_t)snprintf(gs + o, sizeof gs - o, "%s%s on %#llx: %llu waits, %.1f ms (max %.2f)", o ? "; " : "",
                              svc_name(top[k].svc, b, sizeof b), (unsigned long long)top[k].key, (unsigned long long)top[k].n,
                              top[k].ns / 1e6, top[k].max / 1e6);
    }
    rt_log("hostprof frame %llu: guest waits: %llu waits, %.1f ms; top: %s", (unsigned long long)frame, (unsigned long long)sync_n,
           sync_ns / 1e6, o ? gs : "none");
    memset(objs, 0, sizeof objs);
    sync_ns = sync_n = 0;
    plat_mutex_unlock(&obj_lock);
    memset(host, 0, sizeof host), memset(svcs, 0, sizeof svcs), memset(scopes, 0, sizeof scopes), memset(waits, 0, sizeof waits);
}

// The three largest sums of this frame, over host functions, SVCs and scopes (all threads, so they can add up to more than the frame).
void hostprof_slow_summary(char *out, size_t n) {
    out[0] = 0;
    if (!g_hostprof || n < 80) return;
    struct { uint64_t ns; char name[64]; } top[3];
    memset(top, 0, sizeof top);
    for (int i = 0; i < NHOST + NSVC + HPS_N; i++) {
        uint64_t v;
        char name[64], b[64];
        if (i < NHOST) {
            v = fr_host[i];
            if (!v) continue;
            const char *nm = hostfn_name(RT_HOSTFN_BASE + 16ull * (uint64_t)i);
            snprintf(name, sizeof name, "%s", nm ? nm : "host ?");
        } else if (i < NHOST + NSVC) {
            v = fr_svc[i - NHOST];
            if (!v) continue;
            snprintf(name, sizeof name, "%s", svc_name((uint32_t)(i - NHOST), b, sizeof b));
        } else {
            v = fr_scope[i - NHOST - NSVC];
            if (!v) continue;
            snprintf(name, sizeof name, "%s", scope_names[i - NHOST - NSVC]);
        }
        for (int k = 0; k < 3; k++)
            if (v > top[k].ns) {
                memmove(&top[k + 1], &top[k], (size_t)(2 - k) * sizeof top[0]);
                top[k].ns = v;
                snprintf(top[k].name, sizeof top[k].name, "%s", name);
                break;
            }
    }
    size_t o = (size_t)snprintf(out, n, "; host time (all threads): ");
    for (int k = 0; k < 3 && top[k].ns && o < n - 100; k++)
        o += (size_t)snprintf(out + o, n - o, "%s%s %.1f ms", k ? ", " : "", top[k].name, top[k].ns / 1e6);
    snprintf(out + o, n - o, "; lock and core waits %.1f ms", __atomic_load_n(&fr_wait, __ATOMIC_RELAXED) / 1e6);
}

void hostprof_frame_reset(void) {
    if (!g_hostprof) return;
    memset(fr_host, 0, sizeof fr_host), memset(fr_svc, 0, sizeof fr_svc), memset(fr_scope, 0, sizeof fr_scope);
    __atomic_store_n(&fr_wait, 0, __ATOMIC_RELAXED);
}
