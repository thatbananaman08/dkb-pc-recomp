// Counters for the multithreaded-replay measurement (TOTK_TRACE=perf): shared-state mutations the replay makes, counted per frame in
// vk_exec.c (mt_log). Plain increments: they are read approximately.
#ifndef MTSTAT_H
#define MTSTAT_H
#include <stdint.h>
enum { MT_MIRROR_CREATE, MT_MIRROR_FORGET, MT_POOL_INDEX, MT_USNAP_DROP, MT_GPU_WROTE, MT_NOTE_WRITTEN, MT_VIEW_CREATE, MT_PIPE_MISS, MT_PIPE_LOOKUP,
       MT_TEX_SLOT_NEW, MT_TEXINFO_FULL, MT_TEXINFO_CALL, MT_SCRATCH, MT_SCRATCH_CHUNK, MT_VOLATILE, MT_N };
extern _Thread_local uint64_t mt_cnt[MT_N];   // per thread: the replay workers count into their own
#define MTC(k) (mt_cnt[k]++)
// Time the replay spends in the costly first-use steps (vk_exec.c slow-submission line), per thread like mt_cnt.
enum { MTNS_MIRROR, MTNS_PIPE, MTNS_VIEW, MTNS_TEX, MTNS_WAIT, MTNS_N };
extern _Thread_local uint64_t mt_ns[MTNS_N];
#endif
