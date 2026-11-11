// DKB_CENSUS=1 and DKB_DRAWSPLIT=1 (both off by default; measurement only, no behaviour change): counters for the bindless and
// record-time-translation design (BINDLESS_DESIGN.md phase 0 and R0), logged every 60 presents as "census ..." lines.
//   DKB_CENSUS=1     per guest thread: recordings, ops and draws written into command buffers and the time between
//                    BeginRecording and EndRecording (R0); per frame: texture/sampler handles per draw and how they change,
//                    uniform and storage buffer bindings and how they change, descriptor pushes, sets and pools, replay
//                    op counts; the device descriptor-indexing limits once.
//   DKB_DRAWSPLIT=1  time per draw in the replay split into its parts, measured on every 8th draw with the cycle counter.
#pragma once
#include <stdint.h>

extern int census_flag, split_flag;   // -1: not read yet
int census_read(void);
int split_read(void);
static inline int census_on(void) { return census_flag > 0 || (census_flag < 0 && census_read()); }
static inline int split_on(void) { return split_flag > 0 || (split_flag < 0 && split_read()); }

void census_frame(uint64_t frame);   // at every present (perf.c); logs every 60th

// R0: command buffer recording (nvn_api_cmd.c). tpidr identifies the guest thread.
void r0_begin(uint64_t tpidr);
void r0_op(int draw);
void r0_end(void);
void r0_inline(uint64_t handle, const void *body, uint64_t bytes);   // a CallCommands copy; body = records between OP_BEGIN and OP_END

// replay (vk_exec_draw.c)
void census_op(uint32_t op);                                                   // every op that reaches vk_exec_op_draw
void census_draw(unsigned nbind, unsigned ndesc);                              // a graphics draw's bind_resources
void census_tex(int st, int slot, int img, uint64_t h);                        // one texture or image handle looked up
void census_buf(int ssbo, int st, int slot, uint64_t key, uint64_t range);     // one uniform or storage range bound
// what: 0 push, 1 push skipped, 2 set allocated, 3 set reused, 4 pool created, 5 spare pool taken, 6 bind skipped
void census_desc(int what);

// draw split: every 8th draw is timed; split_acc(k, cycles) adds to bucket k
extern int split_sampling;
void split_begin_draw(void);
void split_end_draw(void);
void split_acc(int k, uint64_t cycles);
static inline uint64_t split_tsc(void) {
#if defined(__x86_64__)
    return __builtin_ia32_rdtsc();
#else
    return 0;
#endif
}
#define CST() (split_sampling ? split_tsc() : 0)
#define CSA(k, t0) do { if (split_sampling) split_acc((k), split_tsc() - (t0)); } while (0)
enum { SP_PREP, SP_PIPE, SP_RES, SP_RES_SIG, SP_RES_BUF, SP_RES_TEX, SP_RES_EMIT, SP_VB, SP_POST, SP_TOTAL, SP_E_BUF, SP_E_TEX, SP_E_PUSH, SP_N };
