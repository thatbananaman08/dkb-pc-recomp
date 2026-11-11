// TOTK_HOSTPROF=1 (with TOTK_TRACE=perf): where the guest threads' host time goes. Off: every hook is one predicted branch.
// Per call timing of the host functions (hostfn.c) and SVCs (svc.c), of named scopes inside the file system path, the time spent
// waiting for host locks, for a virtual core and in the guest's own synchronisation SVCs (with the waited-on addresses), logged
// every 60 frames, and a compact summary on every slow frame (perf.c). Addresses and times only.
#pragma once
#include <stdint.h>

#include "host.h"
#include "platform/platform.h"

extern int g_hostprof;
extern _Thread_local uint64_t hp_wait_ns;   // this thread's wait time so far (locks, core retake); a call's share is the delta
extern _Thread_local int hp_in_core;        // inside kern_core_back: its scheduler-lock time is part of the core wait

enum { HPW_SCHED, HPW_CORE, HPW_FRZ, HPW_MEM, HPW_ROMFS, HPW_FS, HPW_N };
enum { HPS_ROMFS_OPEN, HPS_ROMFS_READ, HPS_ROMFS_PREAD, HPS_MEM_THAW, HPS_N };

typedef struct { uint64_t t0, w0; } HpT;

void hostprof_init(void);   // reads TOTK_HOSTPROF; before the guest runs
void hp_wait_add(int kind, uint64_t ns);
void hp_host_call(int slot, uint64_t ns, uint64_t wait_ns);
void hp_svc(uint32_t imm, uint64_t ns, uint64_t wait_ns, int sync, uint64_t obj);
void hp_scope_add(int id, uint64_t ns, uint64_t wait_ns);
void hostprof_log(uint64_t frame);                      // every 60 frames (perf.c)
void hostprof_slow_summary(char *out, size_t n);        // "; host time ..." of the frame just ended ("" when off)
void hostprof_frame_reset(void);                        // after the slow-frame line

static inline HpT hp_begin(void) {
    if (__builtin_expect(!g_hostprof, 1)) return (HpT){0, 0};
    return (HpT){kern_now_ns(), hp_wait_ns};
}
static inline void hp_end(int scope, HpT t) {
    if (__builtin_expect(!t.t0, 1)) return;
    hp_scope_add(scope, kern_now_ns() - t.t0, hp_wait_ns - t.w0);
}

static inline void hp_lock(PlatMutex *m, int kind) {
    if (__builtin_expect(!g_hostprof, 1)) {
        plat_mutex_lock(m);
        return;
    }
    uint64_t t = kern_now_ns();
    plat_mutex_lock(m);
    hp_wait_add(kind, kern_now_ns() - t);
}
static inline void hp_lock_shared(PlatMutex *m, int kind) {
    if (__builtin_expect(!g_hostprof, 1)) {
        plat_mutex_lock_shared(m);
        return;
    }
    uint64_t t = kern_now_ns();
    plat_mutex_lock_shared(m);
    hp_wait_add(kind, kern_now_ns() - t);
}
