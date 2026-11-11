// Draws, binds and dispatches of the Vulkan op-stream replay: the strong
// vk_exec_op_draw behind vk_exec.c's weak hook. The bind ops fill the VkExec slots; a draw opens the pass, resolves the
// bound program to translated shaders (the translator's answers come from the bound state), asks vk_pipeline.c for the
// pipeline, pushes (or allocates) set 0 as VkPipeInfo.bindings says and records the dynamic state, vertex and index
// buffers and the draw. A dispatch ends the pass and does the same for the compute shader. UPDATE_UB is a buffer copy
// from transient memory between full barriers.
//
// Per command buffer the replay keeps transient memory (support buffers, copied uniform ranges, generated indices),
// descriptor pools, image views and the storage-buffer ranges the GPU may write; all of it is freed, and the ranges are
// reported with vk_gpu_written, from one completion callback once the submission's fence is done (the write watch
// does not see GPU writes). Nothing here locks: it runs on the submitting thread under vk_sync_submit's lock.
//
// Provenance: the op semantics (bind ops, program and slot conventions, draw and dispatch flow, vertex streams with a
// zeroed buffer for unbound ones) follow MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m:7409-7460,
// 9365-9488, 9489-9921, 10013-10079 @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou); the uniform-buffer spans, texture and
// sampler descriptor lookups and the dispatch flow follow LAS Recomp runtime/nvn/vulkan/exec.c:1851-1908, 2300-2516
// @ da35b83a (MIT, Copyright (c) 2026 Peiikai). The support buffer layout is Ryujinx's SupportBuffer.cs (MIT).
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bindless_table.h"
#include "census.h"
#include "vk_vertex_fetch.h"
#include "cfg.h"
#include "host.h"
#include "mtstat.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_geometry_control.h"
#include "nvn_indirect_reset.h"
#include "vk.h"
#include "lq_skip.h"
#include "draw_cache.h"
#include "draw_filter.h"
#include "vk_exec.h"
#include "platform/platform.h"

#define MAX_BINDINGS 128
// NVN 2 compute parameter layout of the program being dispatched (gfx_shader_c0_layout), NULL for NVN 1: read by
// vk_c0_fill, set by the dispatch ops around bind_resources like c0_dim.
static const Nvn2C0Layout *c0_lay;
// TOTK_REPLAY_SPLIT (the end of this file): rsp_worker is set on a replay worker thread; rsp_split_on once the switch is read.
static _Thread_local int rsp_worker;
static int rsp_split_on;
int vk_exec_split_ext(uint32_t bit);
static void rsp_note(int kind);
enum { RSPN_FB_EVENT, RSPN_ALIAS, RSPN_N };
static void rsp_begin(int work);
static void rsp_join(VkExec *ex);
static const uint32_t (*c0_push)[NVN2_ROOT_WORDS];
#define MAX_DESC 1024
#define CHUNK_BYTES (4u << 20)
#define ZERO_BYTES 65536u
#define SUPPORT_BYTES 1408u   // Ryujinx SupportBuffer (std140): alpha_test, is_bgra[8], viewport_inverse, viewport_size, ...
#define NVN_STAGE_VS 0
#define NVN_STAGE_FS 1
#define NVN_STAGE_GS 2
#define NVN_STAGE_CS 5
static _Thread_local uint8_t geometry_input;                              // normalized actual draw primitive; zero outside opt-in GS
#define ONESHOT_WAIT_MS 30000u   // prepare_draw: how long a one-shot draw waits for its pipeline before it is skipped

static void c0_lay_set(VkExec *ex, const GfxShader *cs) {
    c0_lay = gfx_shader_c0_layout(cs);
    c0_push = ex->push[1];
}

enum { W_PROGRAM, W_SHADER, W_PIPELINE, W_NO_VS, W_INDEX, W_UB, W_BIND, W_SSBO_ALIGN, W_SSBO_SLOT, W_RESTART, W_INDIRECT, W_FAULT_SKIP, W_FAULT_SKIP2, W_RESET, W_GEOMETRY, W_COUNT };

static void warn_once(int key, const char *fmt, ...) {
    static volatile int32_t seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

static void *grow(void *p, int *cap, size_t elem) {
    *cap = *cap ? *cap * 2 : 8;
    p = realloc(p, elem * (size_t)*cap);
    if (!p) rt_fatal("vk: out of memory");
    return p;
}

// ---- resources of one command buffer, freed by its completion callback

typedef struct { VkBuffer buf; VmaAllocation alloc; uint8_t *map; uint64_t size, used; } Chunk;
typedef struct {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    uint32_t aspect, level0, levels, layer0, layers, swz;
} ViewKey;
_Static_assert(sizeof(ViewKey) == 40, "ViewKey is hashed as bytes: no padding");
typedef struct { uint64_t addr, size; } Range;
typedef struct {
    Chunk *chunks;
    int nchunks, cchunks;
    VkDescriptorPool *pools;
    int npools, cpools;
    Range *written;
    int nwritten, cwritten;
    uint32_t *written_slot;              // note_written: open-addressed index of written, entry = index + 1, 0 empty
    uint32_t written_slots;              // a power of two
} CbRes;

static _Thread_local CbRes *cur;           // per thread: the replay workers (TOTK_REPLAY_SPLIT) own CbRes of their own
static _Thread_local uint64_t cur_fence;

// Descriptor pools and full-size transient chunks go back to these lists when their command buffer completes, instead
// of being destroyed and created again for the next one: each creation is a driver memory allocation (on Windows an
// allocation, a map and a paging-fence wait on the CPU), about ten of them per frame inside the replay.
#define SPARE_POOLS 64
#define SPARE_CHUNKS 16
static PlatMutex spare_lock = PLAT_MUTEX_INIT;
static VkDescriptorPool spare_pool[SPARE_POOLS];
static Chunk spare_chunk[SPARE_CHUNKS];
static int nspare_pool, nspare_chunk;

// State of the command buffer being recorded that the Vulkan command buffer itself holds.
#define RTC_MAX 32
static _Thread_local uint64_t rtc_hit, rtc_miss;
static _Thread_local struct {
    VkPipeline pipe[2];                  // bound per bind point: 0 graphics, 1 compute
    VkPipeState st;                      // the state vk_pipeline_set_dynamic last recorded
    int st_valid;
    VkBuffer vb[17];                     // vertex buffers last bound
    VkDeviceSize vo[17], vsz[17], vstr[17];
    int vcls[17];                        // their pool class and pool record, for the census
    void *vpool[17];
    NVNbufferRange vin[16];              // the ranges they were resolved from, under vk_mem_gen vgen
    uint64_t vgen;
    int vb_valid;
    uint8_t support[SUPPORT_BYTES];      // last support buffer and where it lives
    float sup_in[3];                     // the support_fill arguments it was filled from
    VkBuffer sup_buf;
    VkDeviceSize sup_off;
    int sup_valid;
    uint8_t c0[4][VK_C0_BYTES];          // last c0 bank per stage (vertex, fragment, compute, geometry) and where it lives
    VkBuffer c0_buf[4];
    VkDeviceSize c0_off[4];
    int c0_valid[4];
    struct { p_bind_buf ssbo[16], ubo[16]; uint32_t dim[3], binst; Nvn2C0Layout lay; uint8_t has_lay;
             uint32_t roots[NVN2_ROOT_TABLES][NVN2_ROOT_WORDS]; } c0_in[4];   // what vk_c0_fill read for c0[k]
    VkPipelineLayout ds_layout;          // graphics layout the sets below were last pushed or bound with
    VkDescriptorSet ds_bound[2];         // allocated set bound per set number (0: none or pushed since)
    VkDescriptorSet ds_last[2];          // last allocated and written set per set number, contents in DS
    VkDescriptorSetLayout ds_last_layout[2];
    int ds_pushed[2];                    // set number holds the push of DS.psig under ds_layout
    int res_valid;                       // the graphics sets bound hold what the inputs in RS give (bind_resources)
    VkPipelineLayout bl_layout;          // graphics layout the global texture table (set VKB_SET) was last bound with
    const VkPipeInfo *bl_info;           // pipeline whose slot blocks bl_pc holds (the push constants of the draws since)
    uint32_t bl_pc[2][GFX_BINDLESS_SLOTS_FS];   // last pushed slots, vertex then fragment block
    uint8_t rec_last[2][512];            // TOTK_BINDLESS_BUF: the last record written per stage, and where it is
    uint32_t rec_last_n[2];
    uint64_t rec_last_addr[2];
    struct RecTexCache { const VkPipeInfo *info; uint64_t serial, g3, h[RTC_MAX]; uint32_t g0, g1, g2, v[RTC_MAX]; } rtc[2];   // bindless_records: the last texture slots per stage
    struct MtLast { int valid; const VkPipeInfo *info; uint32_t bytes, nub, nt, g0, g1, g2; uint64_t mg, ug, serial, g3, addr; uint32_t nfbl; VkImage fbl[9]; uint8_t rec[512]; uint64_t ub[34][2], th[32]; } mtl[2];   // TOTK_REPLAY_MT: the last deferred record per stage
    DcIndex idx;                         // index buffer bound in this command buffer (TOTK_DRAW_FASTPATH)
    DcMemo lqm;                          // light_quad_match result for (program, vertex shader, fragment shader)
    DcMemo fmm;                          // lq_fold_missing result for (program, vertex shader)
} B;

void nvn2_origin_cache(int on);

// TOTK_DRAW_FASTPATH=0: the draw path recomputes per draw what it can remember (light_quad_match, nvn2_origin under its lock)
// and records every vkCmdBindIndexBuffer, as before.
static int fastpath_on(void) {
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("TOTK_DRAW_FASTPATH");
        nvn2_origin_cache(on);
    }
    return on;
}

// Contents of B.ds_last: the set's packed descriptors (VkDescSlot), which with its layout determine every write. A draw
// whose allocated set would hold the same reuses the set instead of allocating and writing one (a per-draw
// cost). psig: the same for the last push per set number; a push equal to it under the same layout is not recorded
// again (the descriptors stay bound across pipeline binds of that layout).
#define DS_SIG_BYTES (MAX_DESC * sizeof(VkDescSlot))
static _Thread_local struct { uint32_t n[2], pn[2]; uint8_t sig[2][DS_SIG_BYTES], psig[2][DS_SIG_BYTES]; } DS;

// TOTK_BIND_STATS counters (bind_stats).
static _Thread_local struct {
    int on;
    uint64_t draws, bind[2], desc[2], push, push_skip, set_new, set_reuse, ac_miss, same_prog, same_all, bind_skip;
} BS = {-1};

// The inputs of the last graphics bind_resources (bind_sig) and what its build counted for the traffic census, which a
// skipped build counts again. RS_BYTES bounds the bound-slot bytes a pipeline reads (RsList): per binding at most one
// c0 storage table, otherwise one bound range or handle per descriptor.
#define RS_BYTES (MAX_BINDINGS * sizeof(((VkExec *)0)->ssbo[0]) + MAX_DESC * sizeof(p_bind_buf))
typedef struct { int cls, use; uint64_t bytes; } RsCount;
static _Thread_local struct {
    int on;                              // TOTK_BIND_SKIP: 1 skip (default), 2 check, 0 off
    uint32_t hn;                         // words of h the last signature filled; 0: none
    uint64_t h[32];                      // generations, descriptor tables, feedback images, support inputs, base instance
    const VkPipeInfo *info;              // the pipeline whose RsList ranges body holds, as of bind op generation gen
    uint64_t gen;
    uint8_t body[RS_BYTES];
    int rec;                             // a graphics build is running: count_use records, scratch copies mark volatile
    int volatile_;                       // the build copied guest bytes into transient memory: a rebuild differs
    RsCount cnt[2 * VK_MCLS_N], pcnt[2 * VK_MCLS_N];
    int ncnt, pncnt, cnt_over;
    int cmds;                            // check mode: pushes, binds and set writes the build recorded
    uint32_t ckn;                        // check mode: the last build's packed descriptors (bytes in ck)
    uint8_t ck[DS_SIG_BYTES];
} RS = {-1};

// The bound-slot bytes of VkExec a pipeline's descriptors read, as ranges merged where they touch, built once per
// pipeline info: c0's storage table, the uniform and storage ranges of the bound slots and the texture and image
// handles. sup and c0: the support inputs and the base instance count too. p: storage buffers whose address the shader
// loads from a user uniform block; the address read from the block counts too. ok 0: never skipped (too many
// descriptors or such buffers).
#define RS_PTRS 4
typedef struct { uint32_t off, len; } RsRange;
typedef struct { int st; const GfxShaderRes *res; } RsPtr;
typedef struct { uint32_t n, bytes; uint8_t sup, c0, ok, np; RsPtr p[RS_PTRS]; RsRange r[]; } RsList;
static _Thread_local nvn_hmap rs_lists;                // VkPipeInfo pointer -> RsList, until shutdown
static _Thread_local const VkPipeInfo *rs_last;        // the last lookup
static _Thread_local const RsList *rs_last_l;

static _Thread_local uint32_t c0_dim[3], c0_binst;     // the dispatch size and base instance of the draw or dispatch being bound
static _Thread_local uint64_t bind_gen = 1;            // bumped by every bind op that writes ex->prog, ubo, ssbo, tex or img

static void cb_done(void *arg) {   // completion thread
    CbRes *r = arg;
    VkCtx *c = vk_ctx();
    for (int i = 0; i < r->nwritten; i++) vk_gpu_written(r->written[i].addr, r->written[i].size);
    for (int i = 0; i < r->npools; i++) {
        vkResetDescriptorPool(c->device, r->pools[i], 0);
        plat_mutex_lock(&spare_lock);
        int kept = nspare_pool < SPARE_POOLS;
        if (kept) spare_pool[nspare_pool++] = r->pools[i];
        plat_mutex_unlock(&spare_lock);
        if (!kept) vkDestroyDescriptorPool(c->device, r->pools[i], NULL);
    }
    for (int i = 0; i < r->nchunks; i++) {
        plat_mutex_lock(&spare_lock);
        int kept = r->chunks[i].size == CHUNK_BYTES && nspare_chunk < SPARE_CHUNKS;
        if (kept) spare_chunk[nspare_chunk++] = r->chunks[i];
        plat_mutex_unlock(&spare_lock);
        if (!kept) vmaDestroyBuffer(c->vma, r->chunks[i].buf, r->chunks[i].alloc);
    }
    free(r->written), free(r->written_slot), free(r->pools), free(r->chunks), free(r);
}

void vk_exec_draw_new_cb(void) { memset(&B, 0, sizeof B); }

static CbRes *cb_res(VkExec *ex) {
    if (cur && cur_fence == ex->fence) return cur;
    cur = calloc(1, sizeof *cur);
    if (!cur) rt_fatal("vk: out of memory");
    cur_fence = ex->fence;
    memset(&B, 0, sizeof B);
    vk_completion_add(ex->fence, cb_done, cur);
    return cur;
}

static uint8_t *scratch(VkExec *ex, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    CbRes *r = cb_res(ex);
    MTC(MT_SCRATCH);
    size = (size + 255) & ~255ull;
    Chunk *c = r->nchunks ? &r->chunks[r->nchunks - 1] : NULL;
    if (!c || c->used + size > c->size) {
        if (r->nchunks == r->cchunks) r->chunks = grow(r->chunks, &r->cchunks, sizeof *r->chunks);
        c = &r->chunks[r->nchunks];
        MTC(MT_SCRATCH_CHUNK);
        uint64_t n = size > CHUNK_BYTES ? size : CHUNK_BYTES;
        int reused = 0;
        if (n == CHUNK_BYTES) {
            plat_mutex_lock(&spare_lock);
            if (nspare_chunk) *c = spare_chunk[--nspare_chunk], reused = 1;
            plat_mutex_unlock(&spare_lock);
        }
        if (reused) {
            c->used = 0;
            r->nchunks++;
            *buf = c->buf;
            *off = 0;
            c->used = size;
            return c->map;
        }
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n,
                                 .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | vkb_buf_usage()};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        VkResult res = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &c->buf, &c->alloc, &info);
        if (res != VK_SUCCESS) rt_fatal("vk: transient buffer of %llu bytes failed (%d)", (unsigned long long)n, (int)res);
        c->map = info.pMappedData, c->size = n, c->used = 0;
        r->nchunks++;
    }
    *buf = c->buf;
    *off = c->used;
    uint8_t *p = c->map + c->used;
    c->used += size;
    return p;
}

static uint32_t written_hash(uint64_t addr, uint64_t size) {
    uint64_t h = (addr ^ size * 0xc2b2ae3d27d4eb4full) * 0x9e3779b97f4a7c15ull;
    return (uint32_t)(h >> 32);
}

// Each distinct range once per command buffer, in first-write order. A scene with thousands of written ranges and tens
// of thousands of draws binding them spent most of the replay searching this list, so the search goes through a table.
static void note_written(VkExec *ex, uint64_t addr, uint64_t size) {
    MTC(MT_NOTE_WRITTEN);
    CbRes *r = cb_res(ex);
    if ((uint32_t)r->nwritten * 2 >= r->written_slots) {
        uint32_t slots = r->written_slots ? r->written_slots * 2 : 256;
        uint32_t *table = calloc(slots, sizeof *table);
        if (!table) rt_fatal("vk: out of memory");
        for (int i = 0; i < r->nwritten; i++) {
            uint32_t at = written_hash(r->written[i].addr, r->written[i].size) & (slots - 1);
            while (table[at]) at = (at + 1) & (slots - 1);
            table[at] = (uint32_t)i + 1;
        }
        free(r->written_slot);
        r->written_slot = table, r->written_slots = slots;
    }
    uint32_t mask = r->written_slots - 1, at = written_hash(addr, size) & mask;
    for (; r->written_slot[at]; at = (at + 1) & mask) {
        const Range *w = &r->written[r->written_slot[at] - 1];
        if (w->addr == addr && w->size == size) return;
    }
    if (r->nwritten == r->cwritten) r->written = grow(r->written, &r->cwritten, sizeof *r->written);
    r->written[r->nwritten++] = (Range){addr, size};
    r->written_slot[at] = (uint32_t)r->nwritten;
    vk_mem_gpu_pending(addr, size);
}

static int reset_mapped(void *context, uint64_t addr, uint32_t size) {
    (void)context;
    VkMemSpan span;
    return vk_mem_resolve(addr, size, &span);
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

// Whether two nonnull targets share a byte (duplicates included). Their copies then need a barrier between them to
// keep the input order; disjoint targets can all be copied after one barrier.
static int reset_targets_overlap(const uint64_t *pointers, uint32_t count) {
    uint64_t *sorted = malloc((size_t)count * sizeof *sorted);
    if (!sorted) rt_fatal("vk: out of memory checking indirect-reset targets");
    uint32_t n = 0;
    for (uint32_t i = 0; i < count; i++)
        if (pointers[i]) sorted[n++] = pointers[i];
    qsort(sorted, n, sizeof *sorted, cmp_u64);
    int overlap = 0;
    for (uint32_t i = 1; i < n && !overlap; i++) overlap = sorted[i] - sorted[i - 1] < NVN_INDIRECT_RESET_RECORD_BYTES;
    free(sorted);
    return overlap;
}

// The pointer list may have been written earlier in this submission. Read its GPU-visible contents after submitting
// that prefix, then validate every target before recording a write. Null pointers have no record to reset.
int vk_exec_reset_indirect(VkExec *ex, uint64_t list_addr, uint32_t descriptor_bytes, int64_t first, int64_t count) {
    NvnIndirectResetList list;
    if (!nvn_indirect_reset_list(list_addr, descriptor_bytes, first, count, &list)) return 0;
    uint64_t *pointers = malloc(list.bytes);
    if (!pointers) rt_fatal("vk: out of memory reading an indirect-reset pointer list");
    if (!vk_exec_read_buffer(ex, list.begin, list.bytes, pointers) ||
        !nvn_indirect_reset_targets(&list, pointers, reset_mapped, NULL)) {
        free(pointers);
        return 0;
    }
    VkBuffer source;
    VkDeviceSize source_offset;
    static const uint32_t record[5] = {0, 1, 0, 0, 0};
    memcpy(scratch(ex, sizeof record, &source, &source_offset), record, sizeof record);
    CbRes *resources = cb_res(ex);
    Chunk *chunk = &resources->chunks[resources->nchunks - 1];
    if (vmaFlushAllocation(vk_ctx()->vma, chunk->alloc, source_offset, sizeof record) != VK_SUCCESS)
        rt_fatal("vk: flushing indirect-reset values failed");
    int ordered = reset_targets_overlap(pointers, list.count);
    vk_exec_barrier(ex);   // the copies follow every earlier access to the records
    for (uint32_t i = 0; i < list.count; i++) {
        uint64_t addr = pointers[i];
        if (!addr) continue;
        VkMemSpan span;
        if (!vk_mem_resolve(addr, sizeof record, &span))
            rt_fatal("vk: an indirect-reset mapping changed during replay");
        // Commit first-use imports to their mirror before choosing the destination. A later indirect bind must
        // read this same backing rather than create a mirror from the old contents.
        VK_MEM_USE(span);
        if (ordered) vk_exec_barrier(ex);   // preserves the input order of duplicate or overlapping target ranges
        VkBufferCopy copy = {source_offset, span.offset, sizeof record};
        vkCmdCopyBuffer(ex->cb, source, span.buffer, 1, &copy);
        ex->recorded++;
        vk_mem_gpu_wrote(&span, sizeof record);
        vk_usnap_drop(addr, sizeof record);
        vk_mem_count(span.cls, VK_MUSE_GPU_WRITE, sizeof record);
        note_written(ex, addr, sizeof record);
    }
    free(pointers);
    vk_exec_barrier(ex);   // subsequent shader accesses and indirect reads see every completed record
    return 1;
}

// ---- stand-ins for what a binding names but the game did not bind (zeros, a 1x1 image, the default sampler)

enum { DV_1D, DV_1D_ARRAY, DV_2D, DV_2D_ARRAY, DV_3D, DV_CUBE, DV_CUBE_ARRAY, DV_2D_MS, DV_2D_MS_ARRAY, DV_COUNT };

static struct {
    int ready;
    VkBuffer zbuf;
    VmaAllocation zalloc;
    VkBufferView zview;
    VkSampler samp, samp_shadow;
    struct { VkImage image; VmaAllocation alloc; VkImageView view, sview; } img[DV_COUNT];
} Z;

static const struct { VkImageType t; VkImageViewType vt; uint32_t layers; VkSampleCountFlagBits s; VkImageCreateFlags f; } dinfo[DV_COUNT] = {
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_4_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_4_BIT, 0}};

// A 1x1 stand-in image of view type dinfo[i] in `fmt`, moved to GENERAL by the upload batch (which reaches the queue before
// the submission that uses it). *sview is the view a storage descriptor takes (a cube becomes a 2D array).
static void make_dummy(int i, VkFormat fmt, VkImage *image, VmaAllocation *alloc, VkImageView *view, VkImageView *sview) {
    VkCtx *c = vk_ctx();
    int ms = dinfo[i].s != VK_SAMPLE_COUNT_1_BIT;
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = dinfo[i].f, .imageType = dinfo[i].t,
                            .format = fmt, .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = dinfo[i].layers,
                            .samples = dinfo[i].s, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | (ms ? 0 : VK_IMAGE_USAGE_STORAGE_BIT),
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ia = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ia, image, alloc, NULL) != VK_SUCCESS) rt_fatal("vk: stand-in image %d failed", i);
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *image,
                                .viewType = dinfo[i].vt, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, dinfo[i].layers}};
    if (vkCreateImageView(c->device, &vi, NULL, view) != VK_SUCCESS) rt_fatal("vk: stand-in view failed");
    *sview = *view;
    if (!ms && (dinfo[i].vt == VK_IMAGE_VIEW_TYPE_CUBE || dinfo[i].vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)) {
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        if (vkCreateImageView(c->device, &vi, NULL, sview) != VK_SUCCESS) rt_fatal("vk: stand-in view failed");
    }
    VkImageMemoryBarrier2 bar = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                 .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                 .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .image = *image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, dinfo[i].layers}};
    VkCommandBuffer cb = vk_upload_begin();
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &bar};
    vkCmdPipelineBarrier2(cb, &di);
    vk_upload_end();
}

// Storage image stand-ins in the format a shader declares (GfxShaderRes.format: Ryujinx TextureFormat order); Vulkan wants
// the view format to match the declaration. Created on first use.
static const VkFormat tex_format[40] = {
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT,
    VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT,
    VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM,
    VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM,
    VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT,
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT,
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT,
    VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
    VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_B10G11R11_UFLOAT_PACK32};
static struct { int used, di; VkFormat fmt; VkImage image; VmaAllocation alloc; VkImageView view, sview; } SD[48];

static VkImageView storage_dummy(int di, int code) {
    VkFormat fmt = code > 0 && code < 40 ? tex_format[code] : VK_FORMAT_R8G8B8A8_UNORM;
    if (fmt == VK_FORMAT_R8G8B8A8_UNORM || dinfo[di].s != VK_SAMPLE_COUNT_1_BIT) return Z.img[di].sview;
    for (int i = 0; i < 48; i++) {
        if (!SD[i].used) {
            SD[i].used = 1, SD[i].di = di, SD[i].fmt = fmt;
            make_dummy(di, fmt, &SD[i].image, &SD[i].alloc, &SD[i].view, &SD[i].sview);
            return SD[i].sview;
        }
        if (SD[i].di == di && SD[i].fmt == fmt) return SD[i].sview;
    }
    return Z.img[di].sview;
}

static void dummies_init(void) {
    if (Z.ready) return;
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = ZERO_BYTES,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | vkb_buf_usage()};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &Z.zbuf, &Z.zalloc, &info) != VK_SUCCESS) rt_fatal("vk: zero buffer failed");
    memset(info.pMappedData, 0, ZERO_BYTES);
    VkBufferViewCreateInfo bv = {.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = Z.zbuf,
                                 .format = VK_FORMAT_R32G32B32A32_SFLOAT, .range = ZERO_BYTES};
    if (vkCreateBufferView(c->device, &bv, NULL, &Z.zview) != VK_SUCCESS) rt_fatal("vk: zero buffer view failed");
    samp_par sp;
    memset(&sp, 0, sizeof sp);
    sp.lod_max = 1000.0f;
    Z.samp = vk_sampler_for(&sp);
    sp.cmp_mode = 1, sp.cmp_func = NVN_DEPTH_FUNC_ALWAYS;
    Z.samp_shadow = vk_sampler_for(&sp);

    for (int i = 0; i < DV_COUNT; i++)
        make_dummy(i, VK_FORMAT_R8G8B8A8_UNORM, &Z.img[i].image, &Z.img[i].alloc, &Z.img[i].view, &Z.img[i].sview);
    Z.ready = 1;
}

static void twins_release(VkCtx *c);

// Releases the stand-ins and the depth twins. The device must be idle; the game path leaves them to process exit like
// the pipelines.
void vk_exec_draw_shutdown(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    twins_release(c);
    for (int i = 0; Z.ready && i < DV_COUNT; i++) {
        if (Z.img[i].sview != Z.img[i].view) vkDestroyImageView(c->device, Z.img[i].sview, NULL);
        vkDestroyImageView(c->device, Z.img[i].view, NULL);
        vmaDestroyImage(c->vma, Z.img[i].image, Z.img[i].alloc);
    }
    for (int i = 0; i < 48 && SD[i].used; i++) {
        if (SD[i].sview != SD[i].view) vkDestroyImageView(c->device, SD[i].sview, NULL);
        vkDestroyImageView(c->device, SD[i].view, NULL);
        vmaDestroyImage(c->vma, SD[i].image, SD[i].alloc);
    }
    memset(SD, 0, sizeof SD);
    nvn_hm_clear(&rs_lists, free);
    rs_last = NULL, rs_last_l = NULL;
    for (int i = 0; i < nspare_pool; i++) vkDestroyDescriptorPool(c->device, spare_pool[i], NULL);
    for (int i = 0; i < nspare_chunk; i++) vmaDestroyBuffer(c->vma, spare_chunk[i].buf, spare_chunk[i].alloc);
    nspare_pool = nspare_chunk = 0;
    cur = NULL, cur_fence = 0;
    vk_usnap_shutdown();
    if (Z.ready) {
        vkDestroyBufferView(c->device, Z.zview, NULL);
        vmaDestroyBuffer(c->vma, Z.zbuf, Z.zalloc);
    }
    memset(&Z, 0, sizeof Z);
}

// ---- resource types

static VkImageViewType res_view_type(uint16_t t) {
    int arr = t & 0x100;
    switch (t & 0xff) {
    case 1: return arr ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
    case 4: return VK_IMAGE_VIEW_TYPE_3D;
    case 5: return arr ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
    default: return arr ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
}

static int dummy_index(VkImageViewType vt, int ms) {
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D: return DV_1D;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return DV_1D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_3D: return DV_3D;
    case VK_IMAGE_VIEW_TYPE_CUBE: return DV_CUBE;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return DV_CUBE_ARRAY;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? DV_2D_MS_ARRAY : DV_2D_ARRAY;
    default: return ms ? DV_2D_MS : DV_2D;
    }
}

// GFX_TEX_* of a bound texture slot (the translator's tex_type / img_type answers).
static int gfx_type_of(const VkTexImage *img, const view_t *v, int hv) {
    if (hv && v->set_target) return vk_pipeline_tex_type(v->target);
    int ms = img->samples > 1;
    switch (img->view_type) {
    case VK_IMAGE_VIEW_TYPE_1D: return GFX_TEX_1D;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return GFX_TEX_1D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? GFX_TEX_2D_MS_ARRAY : GFX_TEX_2D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_3D: return GFX_TEX_3D;
    case VK_IMAGE_VIEW_TYPE_CUBE: return GFX_TEX_CUBE;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return GFX_TEX_CUBE_ARRAY;
    default: return ms ? GFX_TEX_2D_MS : GFX_TEX_2D;
    }
}

static VkComponentSwizzle swz_of(uint8_t s) {
    switch (s) {
    case NVN_TEXTURE_SWIZZLE_ZERO: return VK_COMPONENT_SWIZZLE_ZERO;
    case NVN_TEXTURE_SWIZZLE_ONE: return VK_COMPONENT_SWIZZLE_ONE;
    case NVN_TEXTURE_SWIZZLE_R: return VK_COMPONENT_SWIZZLE_R;
    case NVN_TEXTURE_SWIZZLE_G: return VK_COMPONENT_SWIZZLE_G;
    case NVN_TEXTURE_SWIZZLE_B: return VK_COMPONENT_SWIZZLE_B;
    default: return VK_COMPONENT_SWIZZLE_A;
    }
}

// A view in STENCIL mode samples the stencil plane (TextureViewSetDepthStencilMode); otherwise depth wins. The
// translator declares every sampled image float and the shader takes the raw bits, as for integer formats (nxnative
// nvn_mtl.m).
VkImageAspectFlags vk_exec_view_aspect(const VkTexImage *img, const view_t *v, int hv) {
    if (hv && v->set_dsmode && v->dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL && (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT))
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    return img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT     ? VK_IMAGE_ASPECT_DEPTH_BIT
           : img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                       : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Image views of registered textures, kept for their image's lifetime: made per command buffer they were created again
// every frame (vkCreateImageView took about 8% of the fight replay on the 7900X). Used under the texture lock, as
// vk_exec_image_gone is (destroy_tex), which hands the views of a destroyed image to the deferred release with it.
typedef struct ViewNode { struct ViewNode *next; ViewKey k; VkImageView view; } ViewNode;
static nvn_hmap view_of;                 // ViewKey -> VkImageView
static nvn_hmap views_by_image;          // VkImage -> its first ViewNode

// An image view of a registered texture for a shader resource of type `rtype` (GfxShaderRes.type); VK_NULL_HANDLE when
// the image cannot serve it (the caller binds a stand-in). Texture lock held.
static VkImageView get_view(VkExec *ex, const VkTexImage *img, const view_t *v, int hv, uint16_t rtype, int storage) {
    (void)ex;
    VkImageViewType vt = res_view_type(rtype);
    int ms = (rtype & 0x200) != 0;
    if ((img->samples > 1) != ms || (storage && ms)) return VK_NULL_HANDLE;
    if (storage && !(img->usage & VK_IMAGE_USAGE_STORAGE_BIT)) return VK_NULL_HANDLE;
    int cube = vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    int ok = 0;
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D:
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: ok = img->type == VK_IMAGE_TYPE_1D; break;
    case VK_IMAGE_VIEW_TYPE_3D: ok = img->type == VK_IMAGE_TYPE_3D; break;
    case VK_IMAGE_VIEW_TYPE_CUBE:
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
        ok = img->type == VK_IMAGE_TYPE_2D && (img->view_type == VK_IMAGE_VIEW_TYPE_CUBE || img->view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY);
        break;
    default: ok = img->type == VK_IMAGE_TYPE_2D; break;
    }
    if (!ok) return VK_NULL_HANDLE;
    uint32_t l0 = hv && v->set_levels ? v->base_level : 0;
    if (l0 >= img->levels) return VK_NULL_HANDLE;
    uint32_t nl = hv && v->set_levels && v->nlevels ? v->nlevels : img->levels - l0;
    if (l0 + nl > img->levels) nl = img->levels - l0;
    if (storage) nl = 1;
    uint32_t y0 = 0, ny = 1;
    if (vt != VK_IMAGE_VIEW_TYPE_3D) {
        y0 = hv && v->set_layers ? v->min_layer : 0;
        if (y0 >= img->layers) return VK_NULL_HANDLE;
        uint32_t rest = hv && v->set_layers && v->nlayers ? v->nlayers : img->layers - y0;
        if (y0 + rest > img->layers) rest = img->layers - y0;
        if (vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_2D) ny = 1;
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE) ny = 6;
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) ny = rest / 6 * 6;
        else ny = rest;
        if (!ny || y0 + ny > img->layers) return VK_NULL_HANDLE;
    }
    if (storage && cube) vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    uint32_t aspect = vk_exec_view_aspect(img, v, hv);
    uint8_t sw[4] = {NVN_TEXTURE_SWIZZLE_R, NVN_TEXTURE_SWIZZLE_G, NVN_TEXTURE_SWIZZLE_B, NVN_TEXTURE_SWIZZLE_A};
    if (!storage) {   // MNG nvn_mtl.m tex_swizzle_sel: view selectors replace the texture's, then the format remap
        memcpy(sw, hv && v->set_swizzle ? v->swz : img->swz, 4);
        if (img->alpha_one || img->chan[0])
            for (int i = 0; i < 4; i++) {
                if (img->alpha_one && sw[i] == NVN_TEXTURE_SWIZZLE_A) sw[i] = NVN_TEXTURE_SWIZZLE_ONE;
                else if (img->chan[0] && sw[i] >= NVN_TEXTURE_SWIZZLE_R && sw[i] <= NVN_TEXTURE_SWIZZLE_A)
                    sw[i] = img->chan[sw[i] - NVN_TEXTURE_SWIZZLE_R];
            }
    }
    uint32_t swz = (uint32_t)sw[0] | (uint32_t)sw[1] << 8 | (uint32_t)sw[2] << 16 | (uint32_t)sw[3] << 24;
    ViewKey k = {img->image, img->format, vt, aspect, l0, nl, y0, ny, swz};
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    VkImageView found = (VkImageView)nvn_hm_get(&view_of, &k);
    if (found) return found;
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img->image, .viewType = vt,
                                .format = img->format,
                                .components = {swz_of(sw[0]), swz_of(sw[1]), swz_of(sw[2]), swz_of(sw[3])},
                                .subresourceRange = {aspect, l0, nl, y0, ny}};
    VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                      .usage = VK_IMAGE_USAGE_SAMPLED_BIT};
    if (img->block_views) ci.pNext = &use;   // the image's COLOR_ATTACHMENT usage is its block views' (vk_texture.c)
    VkImageView view;
    MTC(MT_VIEW_CREATE);
    if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
    ViewNode *n = malloc(sizeof *n);
    if (!n) rt_fatal("vk: out of memory");
    *n = (ViewNode){nvn_hm_get(&views_by_image, &img->image), k, view};
    nvn_hm_put(&views_by_image, &img->image, n);
    nvn_hm_put(&view_of, &k, (void *)view);
    return view;
}

// An attachment view of a registered texture, kept with its sampled views for the image's lifetime: made per command
// buffer, every render target's views were created again each frame (vkCreateImageView under vk_exec_pass_begin, 2%
// of the fight replay on the 7900X). Attachment keys carry a swizzle word no sampled key has (those hold four NVN
// selectors, each below 8). The texture lock guards the tables, as for get_view; vk_exec_pass_begin may already hold it.
VkImageView vk_exec_target_view(const VkTexImage *img, VkFormat format, VkImageViewType type, uint32_t aspect,
                                uint32_t level, uint32_t base, uint32_t layers, int blocks) {
    ViewKey k = {img->image, format, type, aspect, level, 1, base, layers, blocks ? 0xfffffffeu : 0xffffffffu};
    vk_texture_lock();
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    VkImageView view = (VkImageView)nvn_hm_get(&view_of, &k);
    if (!view) {
        VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img->image,
                                    .viewType = type, .format = format, .subresourceRange = {aspect, level, 1, base, layers}};
        VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                          .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
        if (blocks) ci.pNext = &use;   // not the BC image's sampling and transfer usages
        // A view in another format (an sRGB target over a UNORM texture) drops the image's storage usage, which that
        // format may lack: every colour texture that allows it is made with storage usage (vk_texture.c create_tex).
        else if (format != img->format && (img->usage & VK_IMAGE_USAGE_STORAGE_BIT))
            use.usage = img->usage & ~VK_IMAGE_USAGE_STORAGE_BIT, ci.pNext = &use;
        MTC(MT_VIEW_CREATE);
        if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS)
            rt_fatal("vk: render target view (format %d) could not be created", (int)format);
        ViewNode *n = malloc(sizeof *n);
        if (!n) rt_fatal("vk: out of memory");
        *n = (ViewNode){nvn_hm_get(&views_by_image, &img->image), k, view};
        nvn_hm_put(&views_by_image, &img->image, n);
        nvn_hm_put(&view_of, &k, (void *)view);
    }
    vk_texture_unlock();
    return view;
}

void vk_exec_image_gone(VkImage image) {
    vk_exec_bc_gone(image);
    if (!views_by_image.klen) return;
    for (ViewNode *n = nvn_hm_del(&views_by_image, &image), *next; n; n = next) {
        next = n->next;
        nvn_hm_del(&view_of, &n->k);
        vk_defer_view(n->view);
        free(n);
    }
}

// ---- program to shaders

// (program, stage, answers) -> shader, 4-way set-associative, most recently used first. A miss costs a SHA-256 of the
// whole shader code (gfx_shader.c req_init), slow on CPUs without SHA instructions: a field frame looks up ~3000 and
// the 2048-slot direct-mapped table this replaced missed 3% of them on conflicts.
#define SHADER_SETS 2048
#define SHADER_WAYS 4
typedef struct { uint64_t serial; int stage; int used; GfxShaderAnswers a; const GfxShader *s; } ShaderEnt;
static _Thread_local ShaderEnt shader_tab[SHADER_SETS][SHADER_WAYS];

static int find_stage_scan(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
        const uint8_t *c = p->sd[i].control;
        if (!c || !p->sd[i].data) continue;
        uint32_t st;
        memcpy(&st, c + 0x714, 4);
        if ((int)st != nvn_stage) continue;
        memcpy(len, c + 0x704, 4);
        *ctl = c, *code = (const uint8_t *)(uintptr_t)p->sd[i].data;
        return 1;
    }
    return 0;
}

// Which shader of a program serves each NVN stage, and its code length, per program serial (new at every
// nvnProgramSetShaders): the scan reads two words of every stage's control block, cold guest cache lines, and runs
// several times a draw. Direct-mapped; a collision only scans again.
#define STAGEM_N 512
static _Thread_local struct { uint64_t serial; int8_t idx[6]; uint32_t len[6]; } stagem[STAGEM_N];

static int find_stage(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    if (!p->serial || (unsigned)nvn_stage >= 6) return find_stage_scan(p, nvn_stage, ctl, code, len);
    typeof(stagem[0]) *m = &stagem[p->serial * 0x9E3779B97F4A7C15ull >> 55];
    if (m->serial != p->serial) {
        m->serial = p->serial;
        memset(m->idx, -1, sizeof m->idx);
        for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
            const uint8_t *c = p->sd[i].control;
            if (!c || !p->sd[i].data) continue;
            uint32_t st;
            memcpy(&st, c + 0x714, 4);
            if (st < 6 && m->idx[st] < 0) m->idx[st] = (int8_t)i, memcpy(&m->len[st], c + 0x704, 4);
        }
    }
    int i = m->idx[nvn_stage];
    if (i < 0) return 0;
    *ctl = p->sd[i].control, *code = (const uint8_t *)(uintptr_t)p->sd[i].data, *len = m->len[nvn_stage];
    return 1;
}

// Texture and image types of a stage's first 32 slots, each kept with the handle and the generation and slot token it
// was looked up under: a slot is looked up again, a hash probe, only when the handle changed or its slot may resolve
// otherwise (slot_still).
static _Thread_local struct { uint64_t context, tex[32], img[32], ttok[32], itok[32]; uint32_t tgen[32], igen[32]; uint8_t tt[32], it[32]; } AC[6];
static _Thread_local uint64_t descriptor_serial = 1;

// vk_texture_slot_token of the slot a texture or image handle names; an empty handle always resolves to nothing.
static uint64_t token_of(const VkExec *ex, uint64_t h) {
    return h ? vk_texture_slot_token(&ex->tex_pool, (int)(h & 0xfffff)) : 1;
}

static void descriptor_state(VkExec *ex);

// Whether what handle h resolved to under the global generation *egen and its slot token etok (both read before that
// lookup) still holds at global generation gen: nothing at all changed since, or not its slot (the token, read only when
// the global generation moved, about every 16 draws). A token hit moves *egen to gen.
static int slot_still(const VkExec *ex, uint32_t *egen, uint64_t etok, uint64_t h, uint32_t gen) {
    if (*egen == gen) return 1;
    uint64_t t = token_of(ex, h);
    if (!t || t != etok) return 0;
    *egen = gen;
    return 1;
}
static int shader_memo_on(void);

// The slots of (program, stage, GS input topology) whose types its translation reads (gfx_shader_slot_mask), learned from its first
// shader; ~0u while unknown. The types of other slots select the same shader, so answers_for leaves them zero instead
// of looking up all 64 handles. Direct-mapped by serial and stage; a collision only relearns.
#define SLOTM_N 4096
static _Thread_local struct { uint64_t serial; int st; uint32_t m; } slotm[SLOTM_N];

static uint32_t slotm_idx(uint64_t serial, int st) { return (uint32_t)((serial * 8 + (uint64_t)st) * 0x9E3779B97F4A7C15ull >> 52); }

static int slot_mask_stage(int st) { return st == NVN_STAGE_GS ? st + 8 * geometry_input : st; }

static uint32_t slot_mask_of(uint64_t serial, int st) {
    st = slot_mask_stage(st);                           // GS topology can change which texture queries survive translation
    const typeof(slotm[0]) *e = &slotm[slotm_idx(serial, st)];
    return serial && e->serial == serial && e->st == st ? e->m : ~0u;
}

static void slot_mask_learn(uint64_t serial, int st, const GfxShader *s) {
    if (!serial || slot_mask_of(serial, st) != ~0u) return;
    st = slot_mask_stage(st);
    slotm[slotm_idx(serial, st)] = (typeof(slotm[0])){serial, st, gfx_shader_slot_mask(s)};
}

static uint8_t type_of_handle(const VkExec *ex, uint64_t h) {   // texture lock held
    view_t v;
    int hv;
    const VkTexImage *img = h ? vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), &v, &hv) : NULL;
    return img ? (uint8_t)gfx_type_of(img, &v, hv) : 0;
}

// The answers of stage st for the slots in mask; the types of other slots stay zero.
static void answers_for(VkExec *ex, int st, const uint8_t *ctl, uint32_t mask, GfxShaderAnswers *a) {
    descriptor_state(ex);
    memset(a, 0, sizeof *a);
    if (st == NVN_STAGE_VS) vk_pipeline_answers(&ex->pipe, a, NULL);
    else if (st == NVN_STAGE_FS) vk_pipeline_answers(&ex->pipe, NULL, a);
    else if (st == NVN_STAGE_GS) a->geometry_input = geometry_input;
    else if (st == NVN_STAGE_CS)
        for (int k = 0; k < 3; k++) {
            uint32_t l;
            memcpy(&l, ctl + 0x748 + 4 * k, 4);
            a->local[k] = (uint16_t)(l ? l : 1);
        }
    typeof(AC[0]) *c = &AC[st];
    if (c->context != descriptor_serial) {
        memset(c, 0, sizeof *c);
        c->context = descriptor_serial;
    }
    // The generation and tokens are read before the lookups (one that creates an entry changes them) and unlocked: a
    // registration racing with them is ordered after this draw, and a slot stored under older ones is only looked up
    // again.
    uint32_t gen = vk_texture_slots_gen();
    int locked = 0, check = shader_memo_on() == 2;
    for (uint32_t m = mask; m; m &= m - 1) {
        int slot = __builtin_ctz(m);
        uint64_t h = ex->tex[st][slot], hi = ex->img[st][slot];
        int tmiss = c->tex[slot] != h || !c->ttok[slot] || !slot_still(ex, &c->tgen[slot], c->ttok[slot], h, gen);
        int imiss = c->img[slot] != hi || !c->itok[slot] || !slot_still(ex, &c->igen[slot], c->itok[slot], hi, gen);
        uint64_t tt = tmiss ? token_of(ex, h) : 0, it = imiss ? token_of(ex, hi) : 0;
        if ((tmiss || imiss || check) && !locked) vk_texture_lock(), locked = 1, BS.ac_miss++;
        if (tmiss) c->tt[slot] = type_of_handle(ex, h), c->tex[slot] = h, c->ttok[slot] = tt, c->tgen[slot] = gen;
        if (imiss) c->it[slot] = type_of_handle(ex, hi), c->img[slot] = hi, c->itok[slot] = it, c->igen[slot] = gen;
        if (check && (type_of_handle(ex, h) != c->tt[slot] || type_of_handle(ex, hi) != c->it[slot]))   // TOTK_SHADER_MEMO=check
            rt_fatal("vk_exec: kept texture types of stage %d slot %d differ from a lookup", st, slot);
        a->tex_type[slot] = c->tt[slot], a->img_type[slot] = c->it[slot];
    }
    if (locked) vk_texture_unlock();
}

// DKB_NATIVE_GS=1 runs lowered native passthrough geometry stages through the portable path (issue #45). Off by default.
// Eglantine takes the varyings from the hardware skip mask (Maxwell register 0x1240, translate_program.cpp,
// LowerGeometryPassthrough); the host does not have that mask for NVN 2 programs, so this infers it: what the vertex
// stage writes and the fragment stage reads. That is enough for a fragment stage that reads nothing, as the terrain's
// does; a stage whose real mask differs would forward the wrong set.
static int native_gs_on(void) {
    static int on = -1;
    if (on < 0) on = cfg_bool("TOTK_NATIVE_GS");
    return on;
}

static _Thread_local const GfxShader *native_gs_vs, *native_gs_fs;   // the draw's stages, set before its geometry lookup

static const uint8_t *native_gs_control(const uint8_t *ctl, const uint8_t *code, uint32_t len) {
    // nvn2_lower.cpp::nvn2_lower keeps these source controls for the process lifetime. Keep the packed copies
    // just as long, including after the first 256 contexts; shader_for borrows them even on shader-cache hits.
    static _Thread_local struct { const uint8_t *ctl; uint32_t mask; uint8_t *copy; } *made;
    static _Thread_local size_t count, capacity;
    GfxShaderIo io[32];
    const uint32_t fs_in = gfx_shader_inputs(native_gs_fs, io), vs_out = gfx_shader_outputs(native_gs_vs, io);
    const uint32_t mask = fs_in & vs_out;
    for (size_t i = 0; i < count; i++)
        if (made[i].ctl == ctl && made[i].mask == mask) return made[i].copy;
    uint32_t skip[8];
    skip[0] = 0x0fffffffu;
    for (unsigned i = 1; i < 8; i++) skip[i] = UINT32_MAX;
    for (unsigned l = 0; l < 32; l++)
        if (mask >> l & 1) skip[1 + l / 8] &= ~(15u << (l % 8 * 4));
    uint8_t *copy = malloc(0x880);
    if (!copy) return NULL;
    memcpy(copy, ctl, 0x880);
    if (!nvn_geometry_control_pack_lowered(copy, 0x880, code, len, skip)) {
        rt_log("native GS: stage %016llx does not fit the portable control", (unsigned long long)nvn2_origin(code));
        free(copy);
        return NULL;
    }
    if (count == capacity) {
        if (capacity > SIZE_MAX / 2) { free(copy); return NULL; }
        size_t next = capacity ? capacity * 2 : 256;
        if (next > SIZE_MAX / sizeof *made) { free(copy); return NULL; }
        void *grown = realloc(made, next * sizeof *made);
        if (!grown) { free(copy); return NULL; }
        made = grown, capacity = next;
    }
    made[count].ctl = ctl, made[count].mask = mask, made[count].copy = copy;
    count++;
    rt_log("native GS: stage %016llx forwards locations %#x (vertex writes %#x, fragment reads %#x)",
           (unsigned long long)nvn2_origin(code), mask, vs_out, fs_in);
    return copy;
}

// Every wait of the replay thread for a shader or a pipeline that lasts 2 ms or more is logged with its cause, the stage and
// the NVN 2 name of the stage (the first 200): "vk_exec: frame N: <what> waited X ms (<cause>) ...". The remaining causes are
// the one-shot and full-screen draw waits (prepare_draw) and the synchronous compute path (op_dispatch*), which cannot be
// skipped. TOTK_PIPE_ASYNC=1 makes a --shot run (scenario.py) use the same background builds as normal play.
static void wait_log(const VkExec *ex, const char *what, const char *cause, int st, uint64_t ns) {
    static int logs;
    if (ns < 2000000 || logs >= 200) return;
    const uint8_t *ctl, *code = NULL;
    uint32_t len;
    uint64_t origin = ex->prog && find_stage(ex->prog, st, &ctl, &code, &len) ? nvn2_origin(code) : 0;
    logs++;
    rt_log("vk_exec: frame %llu: %s waited %.1f ms (%s), stage %d origin %016llx",
           (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1, what, (double)ns / 1e6, cause, st,
           (unsigned long long)origin);
}

// *present: the program has this stage. Returns NULL when it has not or the translation failed. pending (graphics
// stages, vk_pipeline_async_on): a shader not translated yet is queued and NULL returned with *pending = 1. await: a
// shader queued earlier is waited for (gfx_shader_await) instead.
static const GfxShader *shader_for(VkExec *ex, int st, int *present, int *pending, int await) {
    const uint8_t *ctl, *code;
    uint32_t len;
    *present = 0;
    if (pending) *pending = 0;
    if (!ex->prog || !find_stage(ex->prog, st, &ctl, &code, &len)) return NULL;
    *present = 1;
    if (st == NVN_STAGE_GS && native_gs_on() && nvn2_origin(code)) {
        if (!native_gs_vs || !native_gs_fs) {   // its varyings follow from the other stages, which are still pending
            if (pending) *pending = 1;
            return NULL;
        }
        if (!(ctl = native_gs_control(ctl, code, len))) return NULL;
    }
    GfxShaderAnswers a;
    uint64_t serial = ex->prog->serial;
    answers_for(ex, st, ctl, slot_mask_of(serial, st), &a);
    ShaderEnt *set = NULL;
    if (serial) {
        uint64_t h = (serial * 0x9E3779B97F4A7C15ull ^ (uint64_t)st) ^ nvn_hm_hash(&a, sizeof a);   // a is zero-padded
        set = shader_tab[(h ^ h >> 29) & (SHADER_SETS - 1)];
        for (int w = 0; w < SHADER_WAYS && set[w].used; w++) {
            ShaderEnt *e = &set[w];
            if (e->serial != serial || e->stage != st || memcmp(&e->a, &a, sizeof a)) continue;
            if (w) {   // to the front
                ShaderEnt hit = *e;
                memmove(&set[1], &set[0], w * sizeof *set);
                set[0] = hit;
            }
            return set[0].s;
        }
    }
    int gs = st == NVN_STAGE_VS ? GFX_STAGE_VS : st == NVN_STAGE_FS ? GFX_STAGE_FS :
             st == NVN_STAGE_GS ? GFX_STAGE_GS : st == NVN_STAGE_CS ? GFX_STAGE_CS : -1;
    if (gs < 0) return NULL;
    const GfxShader *s = await ? gfx_shader_await(gs, ctl, code, len, &a)
                       : pending && vk_pipeline_async_on() ? gfx_shader_get_async(gs, ctl, code, len, &a, pending)
                                                           : gfx_shader_get(gs, ctl, code, len, NULL, &a);
    if (pending && *pending) return NULL;
    if (s) slot_mask_learn(serial, st, s);
    if (!s) warn_once(W_SHADER, "a stage %d shader of program %llu failed to translate: draws with it are skipped", st,
                      (unsigned long long)serial);
    else if (set) {   // in front, the least recently used way drops out
        memmove(&set[1], &set[0], (SHADER_WAYS - 1) * sizeof *set);
        set[0] = (ShaderEnt){.serial = serial, .stage = st, .used = 1, .a = a, .s = s};
    }
    return s;
}

// ---- descriptors

static int nvn_stage_of(int gfx_stage) {
    return gfx_stage == GFX_STAGE_VS ? NVN_STAGE_VS : gfx_stage == GFX_STAGE_FS ? NVN_STAGE_FS :
           gfx_stage == GFX_STAGE_GS ? NVN_STAGE_GS : gfx_stage == GFX_STAGE_CS ? NVN_STAGE_CS : -1;
}

static VkDescriptorBufferInfo zero_info(uint64_t max) {
    return (VkDescriptorBufferInfo){Z.zbuf, 0, max < ZERO_BYTES ? max : ZERO_BYTES};
}

// vk_mem_count, also kept per (class, use) while a graphics build runs (RS).
static void count_use(int cls, int use, uint64_t n) {
    vk_mem_count(cls, use, n);
    if (!RS.rec) return;
    for (int i = 0; i < RS.ncnt; i++)
        if (RS.cnt[i].cls == cls && RS.cnt[i].use == use) {
            RS.cnt[i].bytes += n;
            return;
        }
    if (RS.ncnt < (int)(sizeof RS.cnt / sizeof RS.cnt[0])) RS.cnt[RS.ncnt++] = (RsCount){cls, use, n};
    else RS.cnt_over = 1;
}

// A buffer descriptor for [addr, addr+size) of guest memory: its backing (a pool's VRAM mirror when it has one) when
// aligned, a copy in transient memory when the backing is host memory with an unaligned offset. 0 when neither works.
// written: the shader stores into it (a mirror's range goes back to guest memory at the end of the submission).
// direct_only: guarded aliases must share the real backing; a transient copy would lose writes and alias coherence.
static int bind_range(VkExec *ex, uint64_t addr, uint64_t size, uint64_t align, uint64_t max, int use, int written,
                      int direct_only, VkDescriptorBufferInfo *out) {
    VkMemSpan sp;
    if (!addr || !vk_mem_resolve(addr, 1, &sp)) return 0;
    uint64_t n = size && size < sp.avail ? size : sp.avail;
    if (n > max) n = max;
    if (!n) return 0;
    if (sp.offset % align == 0) {
        VK_MEM_USE(sp);
        *out = (VkDescriptorBufferInfo){sp.buffer, sp.offset, n};
        if (written) vk_mem_gpu_wrote(&sp, n), vk_usnap_drop(addr, n);
        count_use(sp.cls, use, n);
        VK_MEM_BOUND(sp);
        return 1;
    }
    if (sp.alloc || direct_only) return 0;
    VkBuffer b;
    VkDeviceSize o;
    RS.volatile_ = 1;
    memcpy(scratch(ex, n, &b, &o), (const void *)(uintptr_t)addr, n);
    vk_mem_count(sp.cls, VK_MUSE_CPU_READ, n);
    vk_mem_count(VK_MCLS_SCRATCH, use, n);
    *out = (VkDescriptorBufferInfo){b, o, n};
    return 1;
}

static void c0_table(uint8_t *at, const p_bind_buf b[16]) {
    for (int r = 0; r < 16; r++) {
        uint64_t a = b[r].addr;
        uint32_t n = (uint32_t)b[r].size;
        memcpy(at + 16 * r, &a, 8);
        memcpy(at + 16 * r + 8, &n, 4);
    }
}

// The driver constant bank c0 as GLSLC-compiled shaders read it. Storage-buffer tables {u64 address, u32 size} every
// 16 bytes at 0x110 (vertex), 0x510 (fragment), 0x310 (compute): Ryujinx's GlobalToStorage fallback loads each
// candidate base from here and compares the global pointer against it. Compute: dispatch size at 0x200, the uniform
// buffer table at 0x210 + 16 r. Graphics: the largest point size (Maxwell's limit, GLSLC clamps gl_PointSize to it) at
// 0x610, the base instance at 0x644. Other stages: zeros. Unbound slots read zero, since the replay keeps them zero.
// Layout: nxnative dm/gfx/nvn/nvn_mtl.m (MIT).
void vk_c0_fill(uint8_t bank[VK_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                const uint32_t dispatch[3], uint32_t base_instance) {
    if (stage == NVN_STAGE_CS && c0_lay) {
        if (!nvn2_c0_pack(bank, VK_C0_BYTES, c0_lay, c0_push, ssbo, ubo))
            rt_fatal("vk: invalid NVN 2 compute parameter-block layout");
        return;
    }
    memset(bank, 0, VK_C0_BYTES);
    int base =stage == NVN_STAGE_VS ? 0x110 : stage == NVN_STAGE_FS ? 0x510 : stage == NVN_STAGE_CS ? 0x310 :
              stage == NVN_STAGE_GS ? 0x410 : -1;   // geometry (DKB_NATIVE_GS): the table ssbo_slot expects
    if (base < 0) return;
    c0_table(bank + base, ssbo);
    if (stage == NVN_STAGE_CS) {
        memcpy(bank + 0x200, dispatch, 12);
        c0_table(bank + 0x210, ubo);
        return;
    }
    const float max_point = 2047.0f;
    memcpy(bank + 0x610, &max_point, 4);
    memcpy(bank + 0x644, &base_instance, 4);
}

static int c0_on(void) {   // TOTK_FIX_C0=0: c0 reads zeros instead of the driver constants
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("TOTK_FIX_C0");
    }
    return on;
}

// c0 from replay scratch; a bank equal to the stage's previous one reuses its range, so it costs no copy.
static VkDescriptorBufferInfo c0_info(VkExec *ex, int st) {
    if (st != NVN_STAGE_VS && st != NVN_STAGE_FS && st != NVN_STAGE_CS && st != NVN_STAGE_GS)
        rt_fatal("vk_exec: synthetic c0 is unavailable for stage %d", st);
    int k = st == NVN_STAGE_VS ? 0 : st == NVN_STAGE_FS ? 1 : st == NVN_STAGE_GS ? 3 : 2;
    // vk_c0_fill reads buffer bindings, compute root words and dispatch size, or the graphics base instance:
    // equal inputs give the same bank, so neither the fill nor the 2 KiB compare is needed.
    typeof(B.c0_in[0]) in;
    memset(&in, 0, sizeof in);
    memcpy(in.ssbo, ex->ssbo[st], sizeof in.ssbo);
    if (st == NVN_STAGE_CS) memcpy(in.ubo, ex->ubo[st], sizeof in.ubo), memcpy(in.dim, c0_dim, sizeof in.dim);
    else in.binst = c0_binst;
    if (st == NVN_STAGE_CS && c0_lay) {
        in.lay = *c0_lay, in.has_lay = 1;
        for (unsigned i = 0; i < NVN2_ROOT_TABLES; i++)
            memcpy(in.roots[i], ex->push[1][i], 4u * c0_lay->roots[i]);
    }
    if (B.c0_valid[k] && !memcmp(&in, &B.c0_in[k], sizeof in))
        return (VkDescriptorBufferInfo){B.c0_buf[k], B.c0_off[k], VK_C0_BYTES};
    B.c0_in[k] = in;
    uint8_t tmp[VK_C0_BYTES];
    vk_c0_fill(tmp, st, ex->ssbo[st], ex->ubo[st], c0_dim, c0_binst);
    if (!B.c0_valid[k] || memcmp(tmp, B.c0[k], VK_C0_BYTES)) {
        memcpy(scratch(ex, VK_C0_BYTES, &B.c0_buf[k], &B.c0_off[k]), tmp, VK_C0_BYTES);
        memcpy(B.c0[k], tmp, VK_C0_BYTES);
        B.c0_valid[k] = 1;
    }
    return (VkDescriptorBufferInfo){B.c0_buf[k], B.c0_off[k], VK_C0_BYTES};
}

// The buffer range limits, read once: the device's properties are a cold cache line on every binding of every draw.
static struct { uint64_t ubo_max, ubo_align, sbo_max, sbo_align; } LIM;

static void limits_init(void) {
    const VkPhysicalDeviceLimits *l = &vk_ctx()->props.limits;
    LIM.ubo_align = l->minUniformBufferOffsetAlignment, LIM.sbo_max = l->maxStorageBufferRange;
    LIM.sbo_align = l->minStorageBufferOffsetAlignment;
    __atomic_store_n(&LIM.ubo_max, l->maxUniformBufferRange, __ATOMIC_RELEASE);
}

static VkDescriptorBufferInfo ubo_info(VkExec *ex, int st, int slot) {
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    uint64_t max = LIM.ubo_max, align = LIM.ubo_align;
    VkDescriptorBufferInfo bi;
    if (slot == 0 && c0_on()) return c0_info(ex, st);
    if (slot == 1 && ex->prog) {   // compiler constants stored after the shader code
        const uint8_t *ctl, *code;
        uint32_t len, c1;
        if (find_stage(ex->prog, st, &ctl, &code, &len)) {
            memcpy(&c1, ctl + 0x700, 4);
            if (c1 < len && bind_range(ex, (uint64_t)(uintptr_t)code + c1, len - c1, align, max, VK_MUSE_UBO, 0, 0, &bi))
                return bi;
            if (c1 < len && len - c1 <= max) {   // NVN 2 stages: the lowered code (nvn2_lower.cpp) is host memory
                VkBuffer b;
                VkDeviceSize o;
                RS.volatile_ = 1;
                memcpy(scratch(ex, len - c1, &b, &o), code + c1, len - c1);
                return (VkDescriptorBufferInfo){b, o, len - c1};
            }
        }
    } else if (slot == 17 && st != NVN_STAGE_GS) {
        // Switch 2 full-screen blits read a parameter block the console supplies from bank 24, which the lowering moves
        // to bank 17 (never bound by games). For a blit over the whole render area (Eglantine, ir_opt/position_pass.cpp
        // BlitConstantsPass): texel size at 0x0/0x4 and 0x10/0x14, size at 0x8/0xc, UV scale 1 at 0x18/0x1c, UV clamp
        // rectangle 0,0..1,1 at 0x30-0x3c, zero dither bias, integer source rectangle 0,0..W-1,H-1 at 0x50-0x5c.
        // Bank 24 is NVN 2's graphics root table 0: the words the game pushed (OP_PUSH_CONST) replace the guess
        // (Bananza pushes 1/w, 1/h, w, h). No Bananza compute stage reads bank 24; one that did would get zeros plus
        // the compute pushes, never the graphics guess.
        float w = ex->pass_w ? (float)ex->pass_w : 1.0f, h = ex->pass_h ? (float)ex->pass_h : 1.0f;
        union { float f[64]; uint32_t u[64]; } blk = {{1 / w, 1 / h, w, h, 1 / w, 1 / h, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1}};
        blk.u[22] = (uint32_t)(w - 1), blk.u[23] = (uint32_t)(h - 1);
        int k = st == NVN_STAGE_CS;
        if (k) memset(&blk, 0, sizeof blk);
        for (int i = 0; i < 64; i++)
            if (ex->push_set[k][0] >> i & 1) blk.u[i] = ex->push[k][0][i];
        VkBuffer b;
        VkDeviceSize o;
        RS.volatile_ = 1;
        memcpy(scratch(ex, sizeof blk, &b, &o), &blk, sizeof blk);
        return (VkDescriptorBufferInfo){b, o, sizeof blk};
    } else if (slot >= 3 && slot - 3 < 18) {
        const p_bind_buf *b = &ex->ubo[st][slot - 3];
        if (b->size && vk_usnap_find(b->addr, b->size < max ? b->size : max, align, &bi)) {   // as of the submit call
            count_use(VK_MCLS_SCRATCH, VK_MUSE_UBO, bi.range);
            return bi;
        }
        if (bind_range(ex, b->addr, b->size, align, max, VK_MUSE_UBO, 0, 0, &bi)) return bi;
    }
    return zero_info(max);
}

// The NVN slot of a storage binding. Ryujinx numbers a shader's storage buffers in the order it meets them
// (GfxShaderRes.slot); the NVN slot is the entry of the driver constant buffer c0 that holds the buffer's address
// (cbuf_slot 0, handle = word offset): 16-byte entries at byte 0x110 (vertex), 0x510 (fragment) and 0x310 (compute).
// A compute address at c0 0x210..0x30f is uniform block (offset - 0x210) / 16 read as storage: *ubo is set. -1: the
// address is not in c0 (the shader loads it from a user uniform block: ubo_ptr) or the stage has no known table.
// Mapping: MacNativeGames games/totk/tools/totk_xlate/Service.cs:206-222 @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou).
static int ssbo_slot(const GfxShaderRes *r, int st, int *ubo) {
    *ubo = 0;
    if (r->cbuf_slot != 0) return -1;
    // Eglantine shader_recompiler/frontend/maxwell/translate_program.cpp::AddNVNStorageBuffers uses a distinct
    // ordinary geometry table at c0+0x410. This only maps declared
    // bindings; it does not synthesize that driver bank or establish the native NVN2 geometry ABI.
    int off = r->handle * 4, base = st == NVN_STAGE_VS ? 0x110 : st == NVN_STAGE_FS ? 0x510 :
                                  st == NVN_STAGE_GS ? 0x410 : st == NVN_STAGE_CS ? 0x310 : -1;
    if (st == NVN_STAGE_CS && off >= 0x210 && off < 0x310 && (off - 0x210) % 16 == 0) {
        *ubo = 1;
        return (off - 0x210) / 16;
    }
    return base >= 0 && off >= base && off < base + 0x100 && (off - base) % 16 == 0 ? (off - base) / 16 : -1;
}

// 1 when the shader loads the storage buffer's address from a user uniform block (cbuf_slot 3.., NVN uniform slot
// cbuf_slot - 3, byte offset handle * 4). The Depths terrain reads its vertices through one, about 500 draws a frame;
// with a zero buffer they collapse and the ground is missing. MacNativeGames resolves these the same way (NVN_RB_SBUFU,
// games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m @ b75be8cf).
static int ubo_ptr_res(const GfxShaderRes *r) {
    return r->cbuf_slot >= 3 && r->cbuf_slot - 3 < 18;
}

// The inline uniform updates of the submission being replayed, newest last (a ring of UPD_N). op_update_ub writes them
// into the buffer's GPU copy; guest memory gets them only from the write-back when the submission ends. The terrain
// rewrites its pointer block this way before every draw, all at one address, so ubo_ptr reads the latest update that
// covers the pointer, else memory. Payloads stay in the recorded command buffer for the whole replay.
#define UPD_N 4096
static struct {
    uint64_t fence;                      // the submission the entries belong to
    uint32_t n;                          // updates noted in it
    struct { uint64_t lo, hi; const uint8_t *data; } u[UPD_N];
} UPD;

static _Thread_local uint32_t upd_cap = ~0u;   // a split replay worker sees the updates made before its chunk (UPD.n then)

static void upd_note(const VkExec *ex, const p_update_ub *u) {
    if (UPD.fence != ex->fence) UPD.fence = ex->fence, UPD.n = 0;
    UPD.u[UPD.n++ % UPD_N] = (typeof(UPD.u[0])){u->dst, u->dst + u->size, u->data};
}

// 1 when an update of this submission supplied [a, a+n).
static int upd_read(const VkExec *ex, uint64_t a, void *out, size_t n) {
    if (UPD.fence != ex->fence) return 0;
    uint32_t un = UPD.n < upd_cap ? UPD.n : upd_cap;
    for (uint32_t i = un, end = un > UPD_N ? un - UPD_N : 0; i-- > end;) {
        const typeof(UPD.u[0]) *e = &UPD.u[i % UPD_N];
        if (a >= e->hi || a + n <= e->lo) continue;
        if (a < e->lo || a + n > e->hi) return 0;   // partly covered: memory
        memcpy(out, e->data + (a - e->lo), n);
        return 1;
    }
    return 0;
}

// That address as the bound block holds it at this point of the replay; 0 when the block is unbound or too short. The
// block has no size field.
static uint64_t ubo_ptr(const VkExec *ex, int st, const GfxShaderRes *r) {
    const p_bind_buf *u = &ex->ubo[st][r->cbuf_slot - 3];
    uint64_t off = (uint64_t)r->handle * 4, a = 0;
    VkMemSpan sp;
    if (!u->addr || u->size < off + 8 || upd_read(ex, u->addr + off, &a, 8)) return a;
    if (vk_mem_resolve(u->addr + off, 8, &sp)) memcpy(&a, (const void *)(uintptr_t)(u->addr + off), 8);
    return a;
}

// Guarded NVN 2 candidates use the same bytes that the dispatch binds as uniform banks. Bank 0 includes pushed
// roots and packed descriptor tables; bank 1 is the program's bounded compiler-constant data. No scalar is guessed
// to be a pointer here: bind_range determines whether a candidate has backing, and the shader checks the true range.
static p_bind_buf guarded_storage_binding(const VkExec *ex, int st, const GfxShaderRes *r) {
    uint64_t off = (uint64_t)r->handle * 4;
    p_bind_buf b = {0};
    if (st != NVN_STAGE_CS || !c0_lay) return b;
    if (r->cbuf_slot == 0) {
        if (!c0_on() || off > VK_C0_BYTES - 8) return b;
        uint8_t bank[VK_C0_BYTES];
        vk_c0_fill(bank, st, ex->ssbo[st], ex->ubo[st], c0_dim, c0_binst);
        memcpy(&b.addr, bank + off, 8);
        size_t ub = nvn2_c0_part(c0_lay, 1), end = nvn2_c0_part(c0_lay, 3);
        if (off >= ub && off < end && (off - ub) % 16 == 0) {
            uint32_t bytes;
            memcpy(&bytes, bank + off + 8, 4);
            b.size = bytes;   // preserve a typed table's declared extent, not the remainder of its pool
        }
    } else if (r->cbuf_slot == 1 && ex->prog) {
        const uint8_t *ctl, *code;
        uint32_t len, c1;
        if (!find_stage(ex->prog, st, &ctl, &code, &len)) return b;
        memcpy(&c1, ctl + 0x700, 4);
        if (c1 > len || off > len - c1 || len - c1 - off < 8) return b;
        memcpy(&b.addr, code + c1 + off, 8);
    } else if (r->cbuf_slot == 17) {
        // ubo_info binds the synthetic compute push bank here, before its user-UBO mapping. Unwritten words are zero.
        if (off > sizeof ex->push[1][0] - 8) return b;
        uint32_t words[2] = {0};
        for (unsigned i = 0; i < 2; i++) {
            unsigned word = r->handle + i;
            if (ex->push_set[1][0] >> word & 1) words[i] = ex->push[1][0][word];
        }
        memcpy(&b.addr, words, 8);
    } else if (ubo_ptr_res(r)) {
        b.addr = ubo_ptr(ex, st, r);
    }
    return b;
}

static VkDescriptorBufferInfo ssbo_info(VkExec *ex, int st, const GfxShaderRes *r, int written) {
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    uint64_t max = LIM.sbo_max, align = LIM.sbo_align;
    VkDescriptorBufferInfo bi;
    int ubo = 0, slot = -1;
    p_bind_buf up = {0};
    const p_bind_buf *b = NULL;
    int guarded = !!(r->flags & GFX_RES_SSBO_GUARDED);
    if (!guarded) slot = ssbo_slot(r, st, &ubo);
    if (guarded) up = guarded_storage_binding(ex, st, r), b = &up;
    else if (slot >= 0 && slot < 16) b = ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot];
    else if (ubo_ptr_res(r)) up.addr = ubo_ptr(ex, st, r), b = &up;   // size 0: up to the end of the backing
    if (b) {
        // vk_caps.c: shaders translated for sbuf_align address the buffer from its address rounded down to it.
        uint64_t lead = vk_caps.sbuf_align ? b->addr & (vk_caps.sbuf_align - 1) : 0;
        if (bind_range(ex, b->addr - lead, b->size ? b->size + lead : 0, align, max, ubo ? VK_MUSE_UBO : VK_MUSE_SSBO,
                       written, guarded, &bi)) {
            if (written && bi.buffer != VK_NULL_HANDLE) {
                note_written(ex, b->addr - lead, bi.range);
                const uint8_t *ctl, *code;
                uint32_t len;
                if (!rsp_worker && vk_sync_noting() && ex->prog && find_stage(ex->prog, st, &ctl, &code, &len))
                    vk_sync_note_write(b->addr - lead, bi.range, nvn2_origin(code));
            }
            return bi;
        }
        if (b->addr && !guarded) warn_once(W_SSBO_ALIGN, "storage buffer %#llx is not backed or not aligned: a zero buffer is bound",
                               (unsigned long long)b->addr);
    } else {
        warn_once(W_SSBO_SLOT, "a storage buffer whose address is at c%d+%#x (stage %d) is not replayed: a zero buffer is bound",
                  r->cbuf_slot, r->handle * 4, st);
    }
    bi = zero_info(max);
    // Vulkan requires a nonzero descriptor range. One byte is zero elements of the shader's uint runtime array;
    // the guarded operation branches away without touching it. Unguarded users retain the existing 64 KiB dummy.
    if (guarded) bi.range = 1;
    return bi;
}

// ---- feedback loops (maxwell.md 1.9, vk.h VkCtx.feedback)
// TotK's G-buffer and terrain shaders sample targets of the pass that renders them, at their own pixel, after earlier
// draws of the pass wrote them (albedo, normals, R32F linear depth). Maxwell reads see those writes; in Vulkan the read
// is undefined without a barrier, and on the RX 9070 it showed as dark screen-aligned blocks on the island ground
// and hid Tulin's sage spirit. Before a draw whose fragment shader samples an attachment of
// the open pass that a draw or clear of the pass wrote since the pass began or the last barrier:
// - VK_FEEDBACK_LAYOUT: the image is learned (VkTexImage.fb_learned; it has the feedback usage since its creation), the
//   pass restarts once with it in ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT, and from then on its passes attach it in that
//   layout, its descriptors carry that layout, the draws that sample it use a pipeline with the feedback flag
//   (VkPipeState.feedback) and a by-region barrier precedes such a draw.
// - VK_FEEDBACK_RESTART: the pass ends and restarts behind the full barrier (vk_exec_pass_begin).
static int in_fbl(const VkExec *ex, VkImage img) {
    for (int i = 0; i < ex->nfbl; i++)
        if (ex->fbl_img[i] == img) return 1;
    return 0;
}

// The image and the levels [l0, l1) and layers [a0, a1) a texture's view covers.
typedef struct { VkImage image; uint32_t l0, l1, a0, a1; } SubRange;

static SubRange sub_of(const VkTexImage *img, const view_t *v, int hv) {
    SubRange r = {img->image, 0, img->levels, 0, img->layers};
    if (hv && v->set_levels) r.l0 = v->base_level, r.l1 = v->base_level + (v->nlevels ? v->nlevels : img->levels);
    if (hv && v->set_layers) r.a0 = v->min_layer, r.a1 = v->min_layer + (v->nlayers ? v->nlayers : img->layers);
    return r;
}

// Attachment slots of the open pass (bit 8 depth) whose subresource r covers.
static uint32_t attached_as(const VkExec *ex, const SubRange *r) {
    uint32_t m = 0;
    if (!r->image) return 0;
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if ((i < 8 && i >= ex->pass_ncolor) || t->image != r->image) continue;
        if (t->level < r->l0 || t->level >= r->l1) continue;
        if (!t->is3d && (t->base_layer + t->layers <= r->a0 || t->base_layer >= r->a1)) continue;
        m |= 1u << i;
    }
    return m;
}

// TOTK_TEX_MEMO (tex_info, feedback_scan): 0 off, 1 on, 2 (=check) computes every hit again and stops on a difference.
static int td_mode(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_value("TOTK_TEX_MEMO");
        on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
    }
    return on;
}

// The subresource a texture handle samples, per handle, kept while its slot resolves the same (slot_still): a draw's
// texture bindings change the whole-scan memo below, but the handles they bind were mostly looked up before, in this
// pass or an earlier one. Direct-mapped.
#define FM_N 8192
static _Thread_local struct { uint64_t context, h, tok; uint32_t gen; SubRange r; } FM[FM_N];

static SubRange sampled_sub(const VkExec *ex, uint64_t h) {   // texture lock held
    view_t v;
    int hv;
    const VkTexImage *img = vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), &v, &hv);
    return img ? sub_of(img, &v, hv) : (SubRange){0};
}

// Attachments of the open pass the fragment shader fs samples; kept while the shader, the texture bindings and the
// pass stay the same.
static uint32_t feedback_scan(VkExec *ex, const GfxShader *fs) {
    descriptor_state(ex);
    if (ex->fb_fs == fs && ex->fb_serial == ex->tex_serial && ex->fb_pass == ex->pass_serial) return ex->fb_sampled;
    const GfxShaderRes *res;
    int n = gfx_shader_resources(fs, GFX_RES_TEX, &res);
    int st = nvn_stage_of(gfx_shader_stage(fs));
    if (st < 0) return 0;
    uint32_t m = 0, gen = vk_texture_slots_gen();   // before any lookup
    int memo = td_mode(), locked = 0;
    for (int r = 0; r < n; r++)
        for (int k = 0; k < (res[r].count ? res[r].count : 1); k++) {
            int slot = res[r].slot + k;
            uint64_t h = slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
            if (!h) continue;
            typeof(FM[0]) *f = &FM[(h * 0x9E3779B97F4A7C15ull) >> 51];
            if (memo && f->context == descriptor_serial && f->h == h && slot_still(ex, &f->gen, f->tok, h, gen)) {
                if (memo == 2) {
                    if (!locked) vk_texture_lock(), locked = 1;
                    SubRange c = sampled_sub(ex, h);
                    if (memcmp(&c, &f->r, sizeof c))
                        rt_fatal("vk_exec: feedback memo of handle %#llx differs from a lookup", (unsigned long long)h);
                }
                m |= attached_as(ex, &f->r);
                continue;
            }
            uint64_t tok = token_of(ex, h);   // before the lookup
            if (!locked) vk_texture_lock(), locked = 1;
            SubRange sr = sampled_sub(ex, h);
            m |= attached_as(ex, &sr);
            if (tok) *f = (typeof(*f)){.context = descriptor_serial, .h = h, .tok = tok, .gen = gen, .r = sr};
            else f->h = 0;
        }
    if (locked) vk_texture_unlock();
    ex->fb_fs = fs, ex->fb_serial = ex->tex_serial, ex->fb_pass = ex->pass_serial, ex->fb_sampled = m;
    return m;
}

static int geometry_feedback_safe(VkExec *ex, const GfxShader *vs, const GfxShader *gs) {
    // The existing in-pass feedback dependency is fragment-only. Pre-raster reads of an attachment need a different
    // synchronization contract; reject them rather than reuse that by-region fragment barrier.
    if (gs && (feedback_scan(ex, vs) || feedback_scan(ex, gs))) {
        warn_once(W_GEOMETRY, "experimental geometry draw samples an attachment before rasterization: skipped");
        return 0;
    }
    return 1;
}

// Before the pipeline lookup of a draw with fragment shader fs: learns and restarts as the mode says. Returns the
// sampled attachments in the feedback layout (the pipeline's feedback flag and feedback_barrier).
static void feedback_barrier(VkExec *ex);
static uint32_t feedback_prepare(VkExec *ex, const GfxShader *fs) {
    int mode = vk_ctx()->feedback;
    if (mode == VK_FEEDBACK_OFF || !fs || !ex->in_pass) return 0;
    uint32_t m = feedback_scan(ex, fs);
    uint32_t d = m & ex->fb_dirty;
    if (d) vk_exec_frame.fb_draws++;
    if (d & ~(mode == VK_FEEDBACK_LAYOUT ? ex->fbl_mask : 0)) {
        if (mode == VK_FEEDBACK_LAYOUT)
            for (uint32_t i = 0; i < 9; i++) {
                const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
                if ((d >> i & 1) && t->img && t->img->fb_usage) ((VkTexImage *)t->img)->fb_learned = 1;
            }
        if (rsp_worker) {   // a replay worker cannot end the pass: the target is learned for the next pass, this draw gets a barrier
            rsp_note(RSPN_FB_EVENT);
            feedback_barrier(ex);
        } else {
            vk_exec_pass_end(ex);
            vk_exec_pass_begin(ex);   // learned targets now in the feedback layout; the full barrier cleared fb_dirty
            vk_exec_frame.fb_restarts++;
        }
    }
    return mode == VK_FEEDBACK_LAYOUT ? m & ex->fbl_mask : 0;
}

// By-region barrier inside the pass: earlier attachment writes become visible to this draw's fragment shader reads.
static void feedback_barrier(VkExec *ex) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
                           .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded++;
    ex->fb_dirty = 0;
    vk_exec_frame.fb_barriers++;
}

// Marks the attachments the draw with state s writes: colour targets with a nonzero effective write mask, depth with
// depth writes or the stencil test on.
static void feedback_written(VkExec *ex, const VkPipeState *s) {
    if (vk_ctx()->feedback == VK_FEEDBACK_OFF) return;
    if (ex->depth.image && !(ex->fb_dirty & 0x100) && ((s->ds.dt && s->ds.dw) || s->ds.st)) ex->fb_dirty |= 0x100;
    for (uint32_t i = 0; i < ex->pass_ncolor; i++)
        if (ex->color[i].image && !(ex->fb_dirty >> i & 1) && vk_pipeline_target_wmask(s, (int)i)) ex->fb_dirty |= 1u << i;
}

static void tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, VkImage *image);

// Sampled-texture descriptors per (handle, resource type, separate), kept while nothing they derive from changed: the
// slot's resolution and texture entry (slot_still) and the samplers (vk_sampler_gen); the layout is set again on every
// hit from the images in the feedback layout (in_fbl of the image kept with it). The first binding of a slot in a submission (vk_texture_epoch) still makes the slot lookup
// for its storage check; a re-upload writes the same image, so the descriptor stays. Draws bind the same textures
// thousands of times a frame (15,000 lookups a frame in the monsters fight); a hit skips the slot lookup, the view key
// with its hash probe and the sampler lookup. Storage images, which note their binding per submission, are
// not kept. Direct-mapped; texture lock held. TOTK_TEX_MEMO=0 turns it off; =check computes every hit again and stops
// on a difference.
#define TD_N 8192
static _Thread_local struct {
    uint64_t context, h, tok;
    uint32_t gen, epoch, sgen;
    uint16_t type;
    uint8_t separate, valid;
    VkImage image;
    VkDescriptorImageInfo out;
} TD[TD_N];   // per thread: the replay workers (TOTK_REPLAY_MT) keep their own

// context: the descriptor serial the memo entries are valid for (descriptor_serial on the walk thread, the serial the
// draw was queued under on a replay worker).
static void tex_info_core(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, uint64_t context) {
    MTC(MT_TEXINFO_CALL);
    int td_on = td_mode();
    VkImage image;
    if (!h || storage || !td_on) {
        vk_texture_lock();   // recursive: held by most callers; the replay workers (TOTK_REPLAY_SPLIT) take it only here and for a miss
        tex_info_full(ex, r, h, storage, separate, out, &image);
        vk_texture_unlock();
        return;
    }
    uint32_t gen = vk_texture_slots_gen(), ep = vk_texture_epoch(), sg = vk_sampler_gen();   // before any lookup
    typeof(TD[0]) *m = &TD[(h * 0x9E3779B97F4A7C15ull ^ (uint64_t)r->type << 7 ^ (uint64_t)separate) >> 51];
    if (m->valid && m->context == context && m->h == h && m->type == r->type && m->separate == separate && m->sgen == sg &&
        slot_still(ex, &m->gen, m->tok, h, gen)) {
        if (m->epoch != ep) {   // the slot's first binding in this submission: only its storage check is due
            vk_texture_lock();
            vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), NULL, NULL);
            vk_texture_unlock();
            m->epoch = ep;
        }
        *out = m->out;
        out->imageLayout = m->image && ex->nfbl && in_fbl(ex, m->image) ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                                          : VK_IMAGE_LAYOUT_GENERAL;
        if (td_on == 2) {
            VkDescriptorImageInfo full;
            vk_texture_lock();
            tex_info_full(ex, r, h, storage, separate, &full, &image);
            vk_texture_unlock();
            if (full.sampler != out->sampler || full.imageView != out->imageView || full.imageLayout != out->imageLayout)
                rt_fatal("vk_exec: texture descriptor memo of handle %#llx type %#x differs from a lookup",
                         (unsigned long long)h, r->type);
        }
        return;
    }
    uint64_t tok = token_of(ex, h);   // before the lookup
    MTC(MT_TEXINFO_FULL);
    vk_texture_lock();
    tex_info_full(ex, r, h, storage, separate, out, &image);
    vk_texture_unlock();
    *m = (typeof(*m)){.context = context, .h = h, .tok = tok, .gen = gen, .epoch = ep, .sgen = sg, .type = r->type,
                      .separate = (uint8_t)separate, .valid = 1, .image = image, .out = *out};
}

static void tex_info_lookup(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                            VkDescriptorImageInfo *out) {
    descriptor_state(ex);
    tex_info_core(ex, r, h, storage, separate, out, descriptor_serial);
}

static void tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, VkImage *image) {
    int ms = (r->type & 0x200) != 0;
    *image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler samp = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    if (h) {
        view_t v;
        int hv;
        const VkTexImage *img = vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), &v, &hv);
        if (img) view = get_view(ex, img, &v, hv, r->type, storage), *image = img->image;
        if (img && storage) vk_texture_storage_bound_locked(img);
        if (img && !storage && ex->nfbl && in_fbl(ex, img->image)) layout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
        if (!storage && !separate) {
            // A depth texture sampled without a compare reads one texel on the Switch, whatever the sampler's filter:
            // blending neighbouring depths puts reconstructed positions in mid-air at object edges (shadow masks).
            int sid = (int)(h >> 20 & 0xfff);
            int point = img && (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && !(r->type & 0x400) &&
                        vk_exec_view_aspect(img, &v, hv) != VK_IMAGE_ASPECT_STENCIL_BIT;
            samp = point ? vk_sampler_point_by_id_locked(&ex->samp_pool, sid) : vk_sampler_by_id_locked(&ex->samp_pool, sid);
        }
    }
    if (!view) {
        int di = dummy_index(res_view_type(r->type), ms);
        view = storage ? storage_dummy(di, r->format) : Z.img[di].view;
    }
    if (!samp) samp = r->type & 0x400 ? Z.samp_shadow : Z.samp;
    *out = (VkDescriptorImageInfo){separate || storage ? VK_NULL_HANDLE : samp, view, layout};
}

// ---- depth twins: colour images under a compare sampler in a compute shader
// TotK's compute passes sample an R32_SFLOAT texture through a shadow sampler. Maxwell compares any single-channel
// float or unorm texel; Vulkan only formats with SAMPLED_IMAGE_DEPTH_COMPARISON (VUID-vkCmdDispatch-None-06479), which
// AMD's drivers do not give R32_SFLOAT, so the result there is undefined. Such an image gets a depth twin of the same
// size and bits (R32_SFLOAT -> D32_SFLOAT, R16_UNORM -> D16_UNORM), copied through a buffer right before each dispatch
// that samples it (a dispatch is outside any render pass), and the dispatch samples the twin. Draws keep the colour
// image: a copy cannot go inside their pass. Design: Savage Wind Recomp runtime/nvn/vulkan/exec.c depth_twin_update
// (https://github.com/Revivedx/SavageWindRecomp, MIT); the code is ours. TOTK_DEPTH_TWIN=0: off.
#define TWINS 8
typedef struct { VkImage src, img; VmaAllocation mem; VkImageView view[2]; uint32_t w, h, layers; VkFormat fmt; } Twin;
static Twin twins[TWINS];
static unsigned twin_next;
static int twin_dispatch;                // op_dispatch*: the resources being bound are a dispatch's
static VkBuffer twin_buf;
static VmaAllocation twin_buf_mem;
static VkDeviceSize twin_buf_cap;

static void twin_free(void *arg) {   // completion thread, after the last submission that may sample it
    Twin *t = arg;
    VkCtx *c = vk_ctx();
    for (int k = 0; k < 2; k++)
        if (t->view[k]) vkb_view_gone(t->view[k]), vkDestroyImageView(c->device, t->view[k], NULL);
    vmaDestroyImage(c->vma, t->img, t->mem);
    free(t);
}

typedef struct { VkBuffer buf; VmaAllocation mem; } TwinBuf;
static void twin_buf_free(void *arg) {   // completion thread, after the last submission that may copy through it
    TwinBuf *b = arg;
    vmaDestroyBuffer(vk_ctx()->vma, b->buf, b->mem);
    free(b);
}

static VkFormat twin_format(const VkTexImage *img) {
    static int on = -1;
    if (on < 0) on = rt_knob("DEPTH_TWIN", "1");
    if (!on || (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || img->samples > 1 || img->depth > 1) return VK_FORMAT_UNDEFINED;
    VkFormat df = img->format == VK_FORMAT_R32_SFLOAT ? VK_FORMAT_D32_SFLOAT
                : img->format == VK_FORMAT_R16_UNORM  ? VK_FORMAT_D16_UNORM : VK_FORMAT_UNDEFINED;
    if (!df) return df;
    static int compares[2] = {-1, -1};   // per source format: the driver compares it as it is
    int i = df == VK_FORMAT_D16_UNORM;
    if (compares[i] < 0) {
        VkFormatProperties3 p3 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
        VkFormatProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &p3};
        vkGetPhysicalDeviceFormatProperties2(vk_ctx()->phys, img->format, &p2);
        compares[i] = (p3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT) != 0;
        rt_log("vk_exec: %s under a compare sampler: %s", i ? "R16_UNORM" : "R32_SFLOAT",
               compares[i] ? "compared by the driver" : "sampled through a depth twin in compute passes");
    }
    return compares[i] ? VK_FORMAT_UNDEFINED : df;
}

static void twins_release(VkCtx *c) {
    for (int i = 0; i < TWINS; i++) {
        for (int k = 0; k < 2; k++)
            if (twins[i].view[k]) vkDestroyImageView(c->device, twins[i].view[k], NULL);
        if (twins[i].img) vmaDestroyImage(c->vma, twins[i].img, twins[i].mem);
    }
    memset(twins, 0, sizeof twins);
    if (twin_buf) vmaDestroyBuffer(c->vma, twin_buf, twin_buf_mem);
    twin_buf = VK_NULL_HANDLE, twin_buf_mem = NULL, twin_buf_cap = 0;
}

// The twin of img holding img's current texels, the copies recorded into ex->cb; NULL when img needs none or it cannot
// be made.
static Twin *twin_update(VkExec *ex, const VkTexImage *img) {
    VkFormat df = twin_format(img);
    if (!df) return NULL;
    VkCtx *c = vk_ctx();
    uint32_t bpp = df == VK_FORMAT_D32_SFLOAT ? 4 : 2, layers = img->layers ? img->layers : 1;
    VkDeviceSize bytes = (VkDeviceSize)img->width * img->height * layers * bpp;
    if (bytes > (64u << 20)) return NULL;
    if (bytes > twin_buf_cap) {   // one copy buffer, grown; the old one goes after the submission being recorded
        VkDeviceSize want = twin_buf_cap ? twin_buf_cap * 2 : 1u << 16;
        while (want < bytes) want *= 2;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = want,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkBuffer nb;
        VmaAllocation nm;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &nb, &nm, NULL) != VK_SUCCESS) return NULL;
        if (twin_buf) {
            TwinBuf *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (depth twin)");
            *old = (TwinBuf){twin_buf, twin_buf_mem};
            vk_completion_add(vk_sync_issued() + 1, twin_buf_free, old);
        }
        twin_buf = nb, twin_buf_mem = nm, twin_buf_cap = want;
    }
    Twin *t = NULL;
    for (int i = 0; i < TWINS && !t; i++)
        if (twins[i].img && twins[i].src == img->image && twins[i].w == img->width && twins[i].h == img->height &&
            twins[i].layers == layers && twins[i].fmt == df)
            t = &twins[i];
    if (!t) {
        t = &twins[twin_next++ % TWINS];
        if (t->img) {   // the submission being recorded may still sample the replaced twin
            Twin *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (depth twin)");
            *old = *t;
            vk_completion_add(vk_sync_issued() + 1, twin_free, old);
        }
        *t = (Twin){.src = img->image, .w = img->width, .h = img->height, .layers = layers, .fmt = df};
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = df,
                                .extent = {img->width, img->height, 1}, .mipLevels = 1, .arrayLayers = layers,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &t->img, &t->mem, NULL) != VK_SUCCESS) {
            *t = (Twin){0};
            return NULL;
        }
        for (int k = 0; k < 2; k++) {
            VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->img,
                                        .viewType = k ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, .format = df,
                                        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, k ? layers : 1}};
            if (vkCreateImageView(c->device, &vi, NULL, &t->view[k]) != VK_SUCCESS) t->view[k] = VK_NULL_HANDLE;
        }
        static int nlog;
        if (nlog++ < 8)
            rt_log("vk_exec: depth twin %ux%ux%u %s for a compute pass's compare sampler", img->width, img->height,
                   layers, df == VK_FORMAT_D32_SFLOAT ? "D32_SFLOAT" : "D16_UNORM");
    }
    // The source's earlier writes and the twin's earlier readers before the copies; the twin's old texels are dropped.
    VkMemoryBarrier2 m0 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = t->img, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, layers}};
    VkDependencyInfo d0 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m0,
                           .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(ex->cb, &d0);
    VkBufferImageCopy src = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers},
                             .imageExtent = {img->width, img->height, 1}};
    vkCmdCopyImageToBuffer(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, twin_buf, 1, &src);
    VkMemoryBarrier2 m1 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
    VkDependencyInfo d1 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m1};
    vkCmdPipelineBarrier2(ex->cb, &d1);
    VkBufferImageCopy dst = {.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, layers},
                             .imageExtent = {img->width, img->height, 1}};
    vkCmdCopyBufferToImage(ex->cb, twin_buf, t->img, VK_IMAGE_LAYOUT_GENERAL, 1, &dst);
    // The twin's texels before the dispatch; the source's next writers and the buffer's next copy after these copies.
    VkMemoryBarrier2 m2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo d2 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m2};
    vkCmdPipelineBarrier2(ex->cb, &d2);
    ex->recorded += 5;
    return t;
}

// The image descriptor of slot h; in a dispatch, a compare-sampled colour image the driver cannot compare is swapped
// for its depth twin.
static void tex_info(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate, VkDescriptorImageInfo *out) {
    tex_info_lookup(ex, r, h, storage, separate, out);
    if (!twin_dispatch || storage || separate || !h || !(r->type & 0x400)) return;
    VkImageViewType vt = res_view_type(r->type);
    if (vt != VK_IMAGE_VIEW_TYPE_2D && vt != VK_IMAGE_VIEW_TYPE_2D_ARRAY) return;
    const VkTexImage *img = vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), NULL, NULL);
    Twin *t = img ? twin_update(ex, img) : NULL;
    VkImageView v = t ? t->view[vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY] : VK_NULL_HANDLE;
    if (v) out->imageView = v, out->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
}

static VkDescriptorSet alloc_set(VkExec *ex, VkDescriptorSetLayout layout) {
    CbRes *r = cb_res(ex);
    VkCtx *c = vk_ctx();
    if (census_on()) census_desc(2);
    for (int attempt = 0; attempt < 2; attempt++) {
        if (r->npools) {
            VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                              .descriptorPool = r->pools[r->npools - 1], .descriptorSetCount = 1,
                                              .pSetLayouts = &layout};
            VkDescriptorSet set;
            VkResult res = vkAllocateDescriptorSets(c->device, &ai, &set);
            if (res == VK_SUCCESS) return set;
            if (res != VK_ERROR_OUT_OF_POOL_MEMORY && res != VK_ERROR_FRAGMENTED_POOL) rt_fatal("vk: descriptor set allocation failed (%d)", (int)res);
        }
        static const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256}};
        VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 256,
                                         .poolSizeCount = sizeof sizes / sizeof sizes[0], .pPoolSizes = sizes};
        if (r->npools == r->cpools) r->pools = grow(r->pools, &r->cpools, sizeof *r->pools);
        VkDescriptorPool pool = VK_NULL_HANDLE;
        plat_mutex_lock(&spare_lock);
        if (nspare_pool) pool = spare_pool[--nspare_pool];
        plat_mutex_unlock(&spare_lock);
        if (census_on()) census_desc(pool ? 5 : 4);
        if (!pool && vkCreateDescriptorPool(c->device, &pi, NULL, &pool) != VK_SUCCESS) rt_fatal("vk: descriptor pool failed");
        r->pools[r->npools++] = pool;
    }
    rt_fatal("vk: descriptor set allocation failed twice");
    return VK_NULL_HANDLE;
}

static void support_in(const VkExec *ex, float in[3]) {   // what support_fill reads: viewport size, point size
    in[0] = ex->in_pass && ex->has_vp ? ex->vp[0][2] : (float)ex->pass_w;
    in[1] = ex->in_pass && ex->has_vp ? ex->vp[0][3] : (float)ex->pass_h;
    in[2] = ex->point_size > 0 ? ex->point_size : 1.0f;
}

static void support_fill(uint8_t *b, float w, float h, float point_size) {
    memset(b, 0, SUPPORT_BYTES);
    float inv[4] = {w > 0 ? 2.0f / w : 0, h > 0 ? 2.0f / h : 0, 0, 0}, size[4] = {w, h, 0, 0};
    memcpy(b + 144, inv, 16);
    memcpy(b + 160, size, 16);
    const float one = 1.0f;
    memcpy(b + 1392, &point_size, 4);   // point_size (translator patch 0008): vertex shaders that write none output it
    for (int i = 0; i < 73; i++) memcpy(b + 192 + 16 * i, &one, 4);   // render scale 1: the host renders at guest size
}

// TOTK_BIND_STATS=1: bindings and descriptors per draw of each stage (Task 2.2's choice of the pushed stage), and how
// often a draw repeats the previous one, logged every 20000 draws: pushes recorded and skipped, allocated sets written
// and reused, stage texture-type recomputations (answers_for), draws with the previous draw's program, bound state
// and textures (shaders and pipeline would be the same) and those that also have its buffers and viewport (all
// descriptors would be the same). BS is declared beside DS.
static int bind_stats_on(void) {
    if (BS.on < 0) {
        const char *e = cfg_get("TOTK_BIND_STATS");
        BS.on = e && *e && strcmp(e, "0");
    }
    return BS.on;
}

static void bind_stats(const VkPipeInfo *info) {
    if (!bind_stats_on() || info->bind_point != VK_PIPELINE_BIND_POINT_GRAPHICS) return;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        int fs = info->n_sets > 1 ? b->set : !(b->stages & VK_SHADER_STAGE_VERTEX_BIT);
        BS.bind[fs]++, BS.desc[fs] += b->count ? b->count : 1;
    }
    if (++BS.draws % 20000 == 0) {
        double d = (double)BS.draws;
        rt_log("vk_exec: per draw over %llu draws: vertex %.2f bindings %.2f descriptors, fragment %.2f bindings %.2f descriptors",
               (unsigned long long)BS.draws, BS.bind[0] / d, BS.desc[0] / d, BS.bind[1] / d, BS.desc[1] / d);
        rt_log("vk_exec: per draw: pushes %.3f skipped %.3f, sets written %.3f reused %.3f, texture types %.3f; "
               "same program and state as the draw before %.3f, also same buffers %.3f; descriptor builds skipped %.3f",
               BS.push / d, BS.push_skip / d, BS.set_new / d, BS.set_reuse / d, BS.ac_miss / d, BS.same_prog / d,
               BS.same_all / d, BS.bind_skip / d);
    }
}

// The bound state a draw reads, compared with the previous draw's (TOTK_BIND_STATS only).
static void repeat_stats(const VkExec *ex, const VkPipeInfo *info) {
    static struct {
        const prog_t *prog;
        VkPipeState pipe;
        uint64_t tex[2][128], img[2][32];
        const VkPipeInfo *info;
        p_bind_buf ubo[2][18], ssbo[2][16];
        float vp[6], point_size;
    } last, cur;
    if (!bind_stats_on()) return;
    memset(&cur, 0, sizeof cur);
    cur.prog = ex->prog, cur.pipe = ex->pipe, cur.info = info;
    const int st[2] = {NVN_STAGE_VS, NVN_STAGE_FS};
    for (int k = 0; k < 2; k++) {
        memcpy(cur.tex[k], ex->tex[st[k]], sizeof cur.tex[k]), memcpy(cur.img[k], ex->img[st[k]], sizeof cur.img[k]);
        memcpy(cur.ubo[k], ex->ubo[st[k]], sizeof cur.ubo[k]), memcpy(cur.ssbo[k], ex->ssbo[st[k]], sizeof cur.ssbo[k]);
    }
    memcpy(cur.vp, ex->vp[0], sizeof cur.vp), cur.point_size = ex->point_size;
    size_t prog_part = offsetof(typeof(cur), info);
    if (!memcmp(&cur, &last, prog_part)) {
        BS.same_prog++;
        if (!memcmp((const char *)&cur + prog_part, (const char *)&last + prog_part, sizeof cur - prog_part)) BS.same_all++;
    }
    last = cur;
}

static int push_skip_on(void) {   // TOTK_PUSH_SKIP=0: every graphics draw pushes its set again
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("TOTK_PUSH_SKIP");
    }
    return on;
}

// A graphics draw whose descriptor inputs equal the previous graphics draw's in this command buffer skips building its
// sets: every write would be the same, the push would be skipped and the allocated set reused without a bind (about a
// quarter of field and monsters2 draws, TOTK_BIND_STATS). The inputs, compared and stored in place, are what
// bind_resources and its helpers read beyond the pipeline info's fixed bindings and Z's stand-ins:
// - the info (bindings, layouts), the program and its serial (the compiler-constant range of UBO slot 1);
// - vk_texture_slots_gen, vk_texture_epoch (a slot's image, view and once-per-submit storage check), vk_sampler_gen;
// - vk_mem_gen (buffer and offset of a span, mirror creation and per-submission sync);
// - the images in the feedback layout (the sampled layout);
// - per binding: the support inputs, c0's storage table and base instance, the bound uniform and storage ranges, the
//   texture and image handles (RsList; compared only when a bind op ran since the last draw, bind_gen);
// - storage buffer addresses read from user uniform blocks (RsList.p, ubo_ptr): compared on every draw, since the
//   block's contents change without a bind op.
// The side effects of a build repeat as no-ops in the same command buffer and submission: c0 and support buffers kept
// in B, note_written's ranges, vk_mem_gpu_wrote's write-backs, the per-submission mirror sync and dirty census note,
// the views cached per submission. The census counts are added again. A build that copied guest bytes into transient
// memory (an unaligned host range) is never skipped after: the copy may differ. B.res_valid says the sets bound hold
// the last build; vk_exec_draw_new_cb's memset clears it, and every build clears it until it ends.
// TOTK_BIND_SKIP=0 builds every draw; =check builds every draw and stops when a draw with equal inputs gets other
// writes, census counts or descriptor commands.
static int bind_skip_on(void) {
    if (RS.on < 0) {
        const char *e = cfg_value("TOTK_BIND_SKIP");
        RS.on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
        if (RS.on != 1) rt_log("vk_exec: descriptor build skip %s", RS.on ? "check" : "off");
    }
    return RS.on;
}

static void rs_add(RsRange *r, uint32_t *n, const VkExec *ex, const void *at, size_t len) {
    uint32_t off = (uint32_t)((const uint8_t *)at - (const uint8_t *)ex);
    if (*n && r[*n - 1].off + r[*n - 1].len == off) r[*n - 1].len += (uint32_t)len;
    else r[(*n)++] = (RsRange){off, (uint32_t)len};
}

static const RsList *rs_list(const VkExec *ex, const VkPipeInfo *info) {
    if (info == rs_last) return rs_last_l;
    if (!rs_lists.klen) nvn_hm_init(&rs_lists, sizeof info);
    RsList *l = nvn_hm_get(&rs_lists, &info);
    if (!l) {
        static _Thread_local RsRange tmp[MAX_DESC];
        RsPtr ptr[RS_PTRS];
        uint32_t n = 0, nd = 0, np = 0;
        int sup = 0, c0 = 0, ok = 1;
        for (uint32_t i = 0; i < info->n_bindings + info->n_bb && ok; i++) {
            const VkPipeBinding *b = i < info->n_bindings ? &info->bindings[i] : &info->bb[i - info->n_bindings];
            const GfxShaderRes *r = b->res;
            uint32_t cnt = b->count ? b->count : 1;
            if ((nd += cnt) > MAX_DESC) { ok = 0; break; }   // the build stops on it
            int st = nvn_stage_of(b->stage);
            if (st < 0) rt_fatal("vk_exec: unsupported descriptor stage %u", b->stage);
            if (b->kind == VK_PIPE_SUPPORT) {
                sup = 1;
            } else if (b->kind == GFX_RES_UBO) {
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    if (slot == 0 && c0_on()) c0 = 1, rs_add(tmp, &n, ex, ex->ssbo[st], sizeof ex->ssbo[st]);
                    else if (slot >= 3 && slot - 3 < 18) rs_add(tmp, &n, ex, &ex->ubo[st][slot - 3], sizeof(p_bind_buf));
                }   // slot 1: the program (bind_sig); others: the zero buffer
            } else if (b->kind == GFX_RES_SSBO) {   // every descriptor of the binding is the same range
                int ubo, slot = ssbo_slot(r, st, &ubo);
                if (slot >= 0 && slot < 16) {
                    rs_add(tmp, &n, ex, ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot], sizeof(p_bind_buf));
                } else if (ubo_ptr_res(r)) {
                    if (np == RS_PTRS) { ok = 0; break; }
                    rs_add(tmp, &n, ex, &ex->ubo[st][r->cbuf_slot - 3], sizeof(p_bind_buf));
                    ptr[np++] = (RsPtr){st, r};
                }
            } else if ((b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) && b->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER &&
                       b->type != VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                int img = b->kind == GFX_RES_IMG, cap = img ? 32 : 128;
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    if (slot >= 0 && slot < cap) rs_add(tmp, &n, ex, img ? &ex->img[st][slot] : &ex->tex[st][slot], 8);
                }
            }
        }
        for (uint32_t i = 0; i < info->n_bl && ok; i++) {   // textures read from the global table: their handles feed the pushed slots
            int st = nvn_stage_of(info->bl[i].stage), slot = info->bl[i].res->slot;
            if (st >= 0 && slot >= 0 && slot < 128) rs_add(tmp, &n, ex, &ex->tex[st][slot], 8);
        }
        uint64_t bytes = 0;
        for (uint32_t i = 0; i < n; i++) bytes += tmp[i].len;
        if (bytes > RS_BYTES) ok = 0;
        l = malloc(sizeof *l + sizeof *tmp * n);
        if (!l) rt_fatal("vk: out of memory");
        l->n = n, l->bytes = (uint32_t)bytes, l->sup = (uint8_t)sup, l->c0 = (uint8_t)c0, l->ok = (uint8_t)ok;
        l->np = (uint8_t)np;
        memcpy(l->p, ptr, sizeof *ptr * np);
        memcpy(l->r, tmp, sizeof *tmp * n);
        nvn_hm_put(&rs_lists, &info, l);
    }
    rs_last = info, rs_last_l = l;
    return l;
}

// The signature of a graphics draw's descriptor inputs into RS; 1 when it equals the stored one. The bound-slot bytes
// are compared only when a bind op ran since they were stored or the pipeline changed: otherwise they are the same.
// Check mode compares them anyway and stops when the bind op generation said they could not differ but they do.
static int bind_sig(const VkExec *ex, const VkPipeInfo *info) {
    const RsList *l = rs_list(ex, info);
    if (!l->ok) return RS.hn = 0, RS.info = NULL, 0;
    uint64_t *h = RS.h, diff = 0;
    uint32_t n = 0;
#define PUT(v) do { uint64_t v_ = (uint64_t)(v); diff |= h[n] ^ v_; h[n++] = v_; } while (0)
    PUT((uintptr_t)info);
    PUT((uintptr_t)ex->prog);
    PUT(ex->prog->serial);
    PUT(ex->tex_pool.addr);
    PUT(ex->tex_pool.count);
    PUT(ex->samp_pool.addr);
    PUT(ex->samp_pool.count);
    PUT((uint64_t)vk_texture_slots_gen() << 32 | vk_texture_epoch());
    PUT((uint64_t)vk_sampler_gen() << 32 | (uint32_t)ex->nfbl);
    PUT(vk_mem_gen() + vk_usnap_gen());   // both only grow
    for (int i = 0; i < ex->nfbl && i < 9; i++) PUT((uintptr_t)ex->fbl_img[i]);
    if (l->sup) {
        float in[3];
        support_in(ex, in);
        uint64_t a;
        uint32_t ps;
        memcpy(&a, in, 8), memcpy(&ps, &in[2], 4);
        PUT(a);
        PUT(ps);
    }
    if (l->c0) PUT(c0_binst);
    for (int i = 0; i < l->np; i++) PUT(ubo_ptr(ex, l->p[i].st, l->p[i].res));
#undef PUT
    diff |= n != RS.hn;
    RS.hn = n;
    int fast = RS.info == info && RS.gen == bind_gen;
    if (fast && RS.on != 2) return !diff;
    const uint8_t *e = (const uint8_t *)ex;
    uint8_t *d = RS.body;
    int bd = 0;
    for (uint32_t i = 0; i < l->n; i++) {
        const RsRange *r = &l->r[i];
        if (!bd && memcmp(d, e + r->off, r->len)) bd = 1;
        if (bd) memcpy(d, e + r->off, r->len);
        d += r->len;
    }
    if (fast && bd) rt_fatal("vk_exec: descriptor build skip check: bound slots changed without a bind op");
    RS.info = info, RS.gen = bind_gen;
    return !diff && !bd;
}

// Writes allocated set s from its packed descriptors d: through the layout's update template, or with the write
// structs w[i0..i1) when it has none.
static void write_set(const VkPipeInfo *info, uint32_t s, VkDescriptorSet set, const VkDescSlot *d, VkWriteDescriptorSet *w,
                      uint32_t i0, uint32_t i1) {
    if (i1 == i0) return;
    if (info->tpl[s]) {
        vk_pipeline_update_tpl_fn()(vk_ctx()->device, set, info->tpl[s], d);
        return;
    }
    for (uint32_t k = i0; k < i1; k++) w[k].dstSet = set;
    vkUpdateDescriptorSets(vk_ctx()->device, i1 - i0, w + i0, 0, NULL);
}

// Writes the pipeline's sets from the bound state: set 0, and set 1 with independent layouts. Each is pushed into the
// command buffer, or allocated, written and bound; a graphics set whose contents equal the last one allocated for that
// set number in this command buffer is reused, and bound again only when another layout or a push came between. A
// graphics push equal to the last one at its set number, with no other layout or bind between, is not recorded.
// The descriptors go into one packed array d, a VkDescSlot per descriptor in binding order, which a set's update
// template reads directly (VkPipeInfo.tpl); write structs are filled only when a set has no template.
// TOTK_BINDLESS_TEX=1: the textures of a bindless pipeline. Each is resolved as before (tex_info, with its memo: the handle is read on the
// CPU), entered in the global table (vkb_slot) and its slot pushed to the stage's block of push constants; the table is bound once per
// layout. A block is pushed again only when the pipeline or a slot changed since the last draw. Texture lock not held.
// Resolved buffer addresses per (resource, stage), kept while the binding's guest address and size and the memory-map and uniform-
// snapshot generations are unchanged (vk_mem_gen, vk_usnap_gen: they move whenever a binding may resolve to another span). Only
// results that did not copy into transient memory (RS.volatile_) are kept. Direct-mapped. TOTK_BUF_CACHE=0 turns it off.
#define BC_N 512
static _Thread_local struct { const GfxShaderRes *res; int st; uint64_t addr, size, mgen, ugen, dev; } BC[BC_N];   // per thread
static _Thread_local uint64_t bc_hit, bc_miss;
static int bc_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("TOTK_BUF_CACHE"); on = !(e && !strcmp(e, "0")); }
    return on;
}
static VkDeviceAddress record_buffer_slow(VkExec *ex, const VkPipeBinding *b);
// The address of the buffer a record entry names, as the descriptor path would bind it.
static VkDeviceAddress record_buffer(VkExec *ex, const VkPipeBinding *b) {
    const GfxShaderRes *r = b->res;
    int st = nvn_stage_of(b->stage);
    const p_bind_buf *g = NULL;
    if (bc_on() && b->kind != VK_PIPE_SUPPORT && st >= 0) {
        if (b->kind == GFX_RES_UBO) {
            if (r->slot >= 3 && r->slot - 3 < 18) g = &ex->ubo[st][r->slot - 3];
        } else if (!(r->flags & GFX_RES_SSBO_GUARDED) && !(r->flags & 1)) {
            int ubo = 0, slot = ssbo_slot(r, st, &ubo);
            if (slot >= 0 && slot < 16 && !ubo) g = &ex->ssbo[st][slot];
        }
    }
    if (!g) return record_buffer_slow(ex, b);
    uint64_t mg = vk_mem_gen(), ug = vk_usnap_gen();
    typeof(BC[0]) *e = &BC[(((uintptr_t)r >> 4) * 0x9E3779B97F4A7C15ull ^ (uint64_t)st) >> 55 & (BC_N - 1)];
    if (e->res == r && e->st == st && e->addr == g->addr && e->size == g->size && e->mgen == mg && e->ugen == ug) {
        bc_hit++;
        return e->dev;
    }
    bc_miss++;
    if (((bc_hit + bc_miss) & 0xfffff) == 0) rt_log("buffer cache: %llu hits, %llu misses; texture slots: %llu hits, %llu misses", (unsigned long long)bc_hit, (unsigned long long)bc_miss, (unsigned long long)rtc_hit, (unsigned long long)rtc_miss);
    int vol = RS.volatile_;
    RS.volatile_ = 0;
    VkDeviceAddress a = record_buffer_slow(ex, b);
    int copied = RS.volatile_;
    RS.volatile_ |= vol;
    if (!copied && rsp_split_on) {   // split replay: a snapshot address depends on the draw's number (drops), so it is not kept
        uint64_t ub, us;
        vk_usnap_span(&ub, &us);
        if (us && a >= ub && a < ub + us) copied = 1;
    }
    if (!copied) *e = (typeof(*e)){r, st, g->addr, g->size, mg, ug, a};
    return a;
}

static VkDeviceAddress record_buffer_slow(VkExec *ex, const VkPipeBinding *b) {
    int st = nvn_stage_of(b->stage);
    VkDescriptorBufferInfo bi;
    if (b->kind == VK_PIPE_SUPPORT) {
        float in[3];
        support_in(ex, in);
        if (!B.sup_valid || memcmp(in, B.sup_in, sizeof in)) {   // as in bind_resources: a new buffer only when the contents changed
            uint8_t tmp[SUPPORT_BYTES];
            support_fill(tmp, in[0], in[1], in[2]);
            memcpy(B.sup_in, in, sizeof in);
            if (!B.sup_valid || memcmp(tmp, B.support, SUPPORT_BYTES)) {
                VkBuffer sb;
                VkDeviceSize so;
                memcpy(scratch(ex, SUPPORT_BYTES, &sb, &so), tmp, SUPPORT_BYTES);
                memcpy(B.support, tmp, SUPPORT_BYTES);
                B.sup_buf = sb, B.sup_off = so, B.sup_valid = 1;
            }
        }
        bi = (VkDescriptorBufferInfo){B.sup_buf, B.sup_off, SUPPORT_BYTES};
    } else if (b->kind == GFX_RES_UBO) {
        bi = ubo_info(ex, st, b->res->slot);
    } else {
        bi = ssbo_info(ex, st, b->res, b->res->flags & 1);
    }
    return bi.buffer ? vkb_buffer_address(bi.buffer) + bi.offset : 0;
}


// TOTK_DRAW_REUSE=1 (probe): how often the inputs of a whole record (raw buffer bindings, texture handles, the program, the info)
// repeat, within the same submission and across submissions, and why a record cannot be reused: records hold addresses of
// transient and snapshot memory that change every submission, so only same-submission repeats could skip the work.
enum { DR_SAME, DR_OTHER, DR_COLD, DR_UNC_C0, DR_UNC_SLOT1, DR_UNC_BLIT, DR_UNC_SUPPORT, DR_UNC_SSBO, DR_N };
static uint64_t dr_cnt[DR_N];
static int dr_report;
static struct { uint64_t key, fence; } DR[1 << 21];
static uint64_t dr_mix(uint64_t h, uint64_t w) { h = (h ^ w) * 0x9E3779B97F4A7C15ull; return h ^ (h >> 31); }
// The record as built, with the addresses in transient chunks and the snapshot arena made relative to their region (chunk index in the
// submission, offset): what a deterministic placement would make equal across submissions. TOTK_DRAW_REUSE=1.
static void draw_reuse_probe(VkExec *ex, const VkPipeInfo *info, int blk, const uint8_t *rec, uint32_t nb, uint32_t bytes) {
    CbRes *r = cb_res(ex);
    uint64_t base[64], ubase = 0, usize = 0;
    int n = r->nchunks < 64 ? r->nchunks : 64;
    for (int i = 0; i < n; i++) base[i] = vkb_buffer_address(r->chunks[i].buf);
    vk_usnap_span(&ubase, &usize);
    uint64_t k = dr_mix(0x77, (uintptr_t)info) ^ (uint64_t)blk;
    int cls[3] = {0, 0, 0};
    for (uint32_t i = 0; i < nb; i++) {
        uint64_t a;
        memcpy(&a, rec + 8u * i, 8);
        uint64_t rel = a, tag = 0;
        if (usize && a >= ubase && a < ubase + usize) rel = a - ubase, tag = 2, cls[2]++;
        else
            for (int c = 0; c < n; c++)
                if (a >= base[c] && a < base[c] + r->chunks[c].size) { rel = a - base[c], tag = 1 + 4 * (uint64_t)c, cls[1]++; break; }
        if (!tag) cls[0]++;
        k = dr_mix(dr_mix(k, rel), tag);
    }
    for (uint32_t i = 8u * nb; i < bytes; i += 8) {
        uint64_t w;
        memcpy(&w, rec + i, 8);
        k = dr_mix(k, w);
    }
    k |= 1;
    static uint64_t calls;
    if (!(++calls & 0xfffff)) dr_report = 1;
    typeof(DR[0]) *e = &DR[(k >> 20) & ((1 << 21) - 1)];
    if (e->key == k) dr_cnt[e->fence == ex->fence ? DR_SAME : DR_OTHER]++;
    else dr_cnt[DR_COLD]++;
    dr_cnt[DR_UNC_C0] += cls[1] > 0, dr_cnt[DR_UNC_SLOT1] += cls[2] > 0;
    e->key = k, e->fence = ex->fence;
    if (dr_report)
        dr_report = 0, rt_log("draw reuse probe (resolved, relative): same-submission %llu, earlier-submission %llu, cold %llu; records with scratch addresses %llu, with snapshot addresses %llu",
               (unsigned long long)dr_cnt[DR_SAME], (unsigned long long)dr_cnt[DR_OTHER], (unsigned long long)dr_cnt[DR_COLD], (unsigned long long)dr_cnt[DR_UNC_C0], (unsigned long long)dr_cnt[DR_UNC_SLOT1]);
}

// ---- TOTK_REPLAY_MT=<workers>: deferred fill of the draw records (MT_DESIGN.md, version (i)). Needs TOTK_BINDLESS_TEX=1 and
// TOTK_BINDLESS_BUF=1. The walk thread reserves each draw's record slot, writes everything that mutates shared state in
// order (c0, support, compiler constants, blit block, every storage buffer), pushes the slot's address at once and queues
// the draw: a worker fills the unwritten uniform-buffer entries (slots 3-20) and the texture slots later. The walk joins
// before any part is submitted (vk_exec_mt_join). A uniform range the worker cannot resolve without changing state (a
// mirror to create or sync, an unaligned copy) is flagged and resolved by the walk thread at the join. B, RS and BS stay
// walk-thread only: the worker code never touches them; the texture memo (TD) and the buffer cache (BC) are per thread.
#define MT_DRAWS 256
#define MT_BUF 34   // up to 18 uniform ranges (slots 3-20) and 16 unwritten storage buffers per stage
#define MT_TEX 32
typedef struct {
    uint8_t *dst;                        // the record slot, in mapped transient memory
    const VkPipeInfo *info;
    uint64_t mg, ug, serial, g3;
    uint32_t g0, g1, g2, seq, blk, nub, nt;
    uint64_t flags;                      // bit k: buffer entry k left to the walk thread
    dpool_ref tex_pool, samp_pool;
    uint32_t nfbl;                       // images in the feedback layout at the draw, for the descriptor layout of its textures
    VkImage fbl[9];
    uint64_t ub[MT_BUF][2];              // address, size of the bound ranges, in info->bb order
    uint64_t th[MT_TEX];                 // texture handles, in info->bl order
} MtDraw;
typedef struct MtChunk { struct MtChunk *qnext, *snext; int n, nflag; MtDraw d[MT_DRAWS]; } MtChunk;

static struct {
    int state;                           // 0 not read, 1 on, -1 off
    int workers, active;                 // active: this submission defers
    PlatMutex m;
    PlatCond work_cv, idle_cv;
    MtChunk *qh, *qt, *freel, *cur, *sh, *st;
    int pending, stop;
    uint32_t seq, want_n, drained;       // want: addresses whose pool a worker found not ready (mirror to create or sync)
    uint64_t want[64];
    uint64_t n_fbl, n_nothing, n_defer, n_same, n_flag, n_inelig, n_chunks, join_ns, n_join;
} MT;

static _Thread_local VkExec wex;         // a worker's view of the walk's state: the descriptor pools of the draw
static _Thread_local struct { const VkPipeInfo *info; uint64_t serial, g3; uint32_t g0, g1, g2, n; uint64_t h[MT_TEX]; uint32_t v[MT_TEX]; } wrtc[2];

// 1: a uniform buffer of slots 3-20, 2: an unwritten storage buffer of c0's table (what record_buffer resolves from the binding
// alone, the rest stays on the walk thread); 0: neither.
static int mt_buf_kind(const VkPipeBinding *b) {
    int st = nvn_stage_of(b->stage);
    if (st < 0 || b->kind == VK_PIPE_SUPPORT) return 0;
    const GfxShaderRes *r = b->res;
    if (b->kind == GFX_RES_UBO) return r->slot >= 3 && r->slot - 3 < 18 ? 1 : 0;
    if (b->kind == GFX_RES_SSBO && !(r->flags & GFX_RES_SSBO_GUARDED) && !(r->flags & 1)) {
        int ubo = 0, slot = ssbo_slot(r, st, &ubo);
        return slot >= 0 && slot < 16 && !ubo ? 2 : 0;
    }
    return 0;
}

static const p_bind_buf *mt_buf_binding(const VkExec *ex, const VkPipeBinding *b, int kind) {
    int st = nvn_stage_of(b->stage), ubo = 0;
    return kind == 1 ? &ex->ubo[st][b->res->slot - 3] : &ex->ssbo[st][ssbo_slot(b->res, st, &ubo)];
}

// ubo_info (kind 1) or ssbo_info (kind 2) as of draw sequence d->seq, without changing shared state; 0: the walk thread must
// resolve it.
static int mt_ubo(const VkPipeBinding *b, int kind, const uint64_t g[2], const MtDraw *d, VkDeviceAddress *out) {
    uint64_t max = kind == 1 ? LIM.ubo_max : LIM.sbo_max, align = kind == 1 ? LIM.ubo_align : LIM.sbo_align, addr = g[0], size = g[1];
    const GfxShaderRes *r = b->res;
    int st = nvn_stage_of(b->stage);
    typeof(BC[0]) *e = &BC[(((uintptr_t)r >> 4) * 0x9E3779B97F4A7C15ull ^ (uint64_t)st) >> 55 & (BC_N - 1)];
    if (e->res == r && e->st == st && e->addr == addr && e->size == size && e->mgen == d->mg && e->ugen == d->ug) {
        *out = e->dev;
        return 1;
    }
    VkDescriptorBufferInfo bi;
    VkDeviceAddress a;
    if (kind == 1 && size && vk_usnap_find_asof(addr, size < max ? size : max, align, &bi, d->seq)) {
        a = vkb_buffer_address(bi.buffer) + bi.offset;
    } else {
        uint64_t lead = kind == 2 && vk_caps.sbuf_align ? addr & (vk_caps.sbuf_align - 1) : 0;
        uint64_t ra = addr - lead, rs = kind == 2 && size ? size + lead : size;
        VkMemSpan sp;
        if (!ra || !vk_mem_resolve(ra, 1, &sp)) {
            a = vkb_buffer_address(Z.zbuf);
        } else {
            uint64_t n = rs && rs < sp.avail ? rs : sp.avail;
            if (n > max) n = max;
            if (!n) a = vkb_buffer_address(Z.zbuf);
            else if (sp.offset % align == 0) {
                if (!vk_mem_use_ready(&sp)) {
                    MT.want[__atomic_fetch_add(&MT.want_n, 1, __ATOMIC_RELAXED) & 63] = ra;
                    return 0;
                }
                a = vkb_buffer_address(sp.buffer) + sp.offset;
            } else if (sp.alloc) a = vkb_buffer_address(Z.zbuf);
            else return 0;   // an unaligned range is copied into transient memory
        }
    }
    *e = (typeof(*e)){r, st, addr, size, d->mg, d->ug, a};
    *out = a;
    return 1;
}

static void mt_do(MtDraw *d) {
    const VkPipeInfo *info = d->info;
    int blk = (int)d->blk;
    uint32_t nb = info->rec_nb[blk], nt = blk ? info->bl_fs : info->bl_vs, k = 0;
    for (uint32_t i = 0; i < info->n_bb; i++) {
        const VkPipeBinding *b = &info->bb[i];
        int kind = (b->stage == GFX_STAGE_FS) == blk ? mt_buf_kind(b) : 0;
        if (!kind) continue;
        VkDeviceAddress a;
        if (mt_ubo(b, kind, d->ub[k], d, &a)) memcpy(d->dst + 8u * b->binding, &a, 8);
        else d->flags |= 1ull << k;
        k++;
    }
    wex.tex_pool = d->tex_pool, wex.samp_pool = d->samp_pool, wex.nfbl = (int)d->nfbl;
    memcpy(wex.fbl_img, d->fbl, sizeof d->fbl);
    typeof(wrtc[0]) *rc = &wrtc[blk];
    int same = rc->info == info && rc->serial == d->serial && rc->g0 == d->g0 && rc->g1 == d->g1 && rc->g2 == d->g2 && rc->g3 == d->g3 &&
               rc->n == d->nt && !d->nfbl;
    for (uint32_t i = 0; same && i < d->nt; i++) same = rc->h[i] == d->th[i];
    if (same) {
        memcpy(d->dst + 8u * nb, rc->v, 4u * nt);
        return;
    }
    uint32_t v[MT_TEX] = {0};
    vk_texture_lock();
    k = 0;
    for (uint32_t i = 0; i < info->n_bl; i++) {
        const VkPipeBinding *b = &info->bl[i];
        if ((b->stage == GFX_STAGE_FS) != blk) continue;
        const GfxShaderRes *r = b->res;
        VkDescriptorImageInfo ii;
        tex_info_core(&wex, r, d->th[k], 0, r->separate, &ii, d->serial);
        int cls = bt_class_of_type(r->type);
        uint32_t s = 0;
        if (cls >= 0) {
            int di = dummy_index(res_view_type(r->type), (r->type & 0x200) != 0);
            s = vkb_slot((uint32_t)cls, ii.sampler, ii.imageView, ii.imageLayout, r->type & 0x400 ? Z.samp_shadow : Z.samp, Z.img[di].view);
        }
        memcpy(d->dst + 8u * nb + 4u * b->binding, &s, 4);
        if (b->binding < MT_TEX) v[b->binding] = s;
        k++;
    }
    vk_texture_unlock();
    if (nt <= MT_TEX && d->nt <= MT_TEX && !d->nfbl) {
        memcpy(rc->h, d->th, 8u * d->nt);
        memcpy(rc->v, v, 4u * nt);
        rc->info = info, rc->serial = d->serial, rc->g0 = d->g0, rc->g1 = d->g1, rc->g2 = d->g2, rc->g3 = d->g3, rc->n = d->nt;
    } else rc->info = NULL;
}

static void mt_run(MtChunk *c) {
    c->nflag = 0;
    for (int i = 0; i < c->n; i++) {
        mt_do(&c->d[i]);
        c->nflag += c->d[i].flags != 0;
    }
}

static void mt_worker(void *arg) {
    (void)arg;
    plat_mutex_lock(&MT.m);
    for (;;) {
        while (!MT.qh && !MT.stop) plat_cond_wait(&MT.work_cv, &MT.m);
        if (MT.stop) break;
        MtChunk *c = MT.qh;
        MT.qh = c->qnext;
        if (!MT.qh) MT.qt = NULL;
        plat_mutex_unlock(&MT.m);
        mt_run(c);
        plat_mutex_lock(&MT.m);
        if (!--MT.pending) plat_cond_broadcast(&MT.idle_cv);
    }
    plat_mutex_unlock(&MT.m);
}

static int mt_on(void) {
    if (MT.state) return MT.state > 0;
    const char *e = getenv("TOTK_REPLAY_MT");
    int n = e ? atoi(e) : 0;
    MT.state = -1;
    if (n <= 0) return 0;
    if (!vkb_buf_mode()) {
        rt_log("TOTK_REPLAY_MT=%d needs TOTK_BINDLESS_TEX=1 and TOTK_BINDLESS_BUF=1: off", n);
        return 0;
    }
    if (n > 16) n = 16;
    td_mode(), bc_on();
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    plat_mutex_init(&MT.m), plat_cond_init(&MT.work_cv), plat_cond_init(&MT.idle_cv);
    MT.workers = n, MT.state = 1;
    for (int i = 0; i < n; i++) plat_thread_spawn(mt_worker, NULL, "replay-mt", 0);
    rt_log("TOTK_REPLAY_MT: %d replay worker threads", n);
    return 1;
}

static void mt_enqueue(MtChunk *c) {
    c->qnext = NULL, c->snext = NULL;
    if (MT.st) MT.st->snext = c; else MT.sh = c;
    MT.st = c;
    MT.n_chunks++;
    plat_mutex_lock(&MT.m);
    if (MT.qt) MT.qt->qnext = c; else MT.qh = c;
    MT.qt = c;
    MT.pending++;
    plat_cond_signal(&MT.work_cv);
    plat_mutex_unlock(&MT.m);
}

// Start of a submission's replay (record_impl): defer only the ops walk, not synchronous work recording.
void vk_exec_mt_begin(int work) {
    rsp_begin(work);
    MT.active = !work && !vk_exec_split_active && mt_on();
    MT.seq = 0;
}

static void mt_fallback(VkExec *ex, MtDraw *d) {
    const VkPipeInfo *info = d->info;
    uint32_t k = 0;
    for (uint32_t i = 0; i < info->n_bb; i++) {
        const VkPipeBinding *b = &info->bb[i];
        int kind = (b->stage == GFX_STAGE_FS) == (int)d->blk ? mt_buf_kind(b) : 0;
        if (!kind) continue;
        if (d->flags >> k & 1) {
            VkDescriptorBufferInfo bi;
            uint64_t max = kind == 1 ? LIM.ubo_max : LIM.sbo_max, align = kind == 1 ? LIM.ubo_align : LIM.sbo_align;
            uint64_t addr = d->ub[k][0], size = d->ub[k][1];
            uint64_t lead = kind == 2 && vk_caps.sbuf_align ? addr & (vk_caps.sbuf_align - 1) : 0;
            if (!(kind == 1 && size && vk_usnap_find_asof(addr, size < max ? size : max, align, &bi, d->seq)) &&
                !bind_range(ex, addr - lead, kind == 2 && size ? size + lead : size, align, max, kind == 1 ? VK_MUSE_UBO : VK_MUSE_SSBO, 0, 0, &bi))
                bi = zero_info(max);
            VkDeviceAddress a = bi.buffer ? vkb_buffer_address(bi.buffer) + bi.offset : 0;
            memcpy(d->dst + 8u * b->binding, &a, 8);
            MT.n_flag++;
        }
        k++;
    }
}

// Waits for every queued draw (helping with the queue), resolves the flagged ones in order and recycles the chunks. Before
// every submission of a part and every read of GPU results.
void vk_exec_mt_join(VkExec *ex) {
    if (vk_exec_split_active) rsp_join(ex);
    if (MT.state <= 0 || (!MT.cur && !MT.sh)) return;
    uint64_t t0 = plat_now_ns();
    if (MT.cur) {
        MtChunk *c = MT.cur;
        MT.cur = NULL;
        if (c->n) mt_enqueue(c); else c->snext = MT.freel, MT.freel = c;
    }
    for (;;) {
        plat_mutex_lock(&MT.m);
        MtChunk *c = MT.qh;
        if (c) {
            MT.qh = c->qnext;
            if (!MT.qh) MT.qt = NULL;
        }
        plat_mutex_unlock(&MT.m);
        if (!c) break;
        mt_run(c);
        plat_mutex_lock(&MT.m);
        if (!--MT.pending) plat_cond_broadcast(&MT.idle_cv);
        plat_mutex_unlock(&MT.m);
    }
    plat_mutex_lock(&MT.m);
    while (MT.pending) plat_cond_wait(&MT.idle_cv, &MT.m);
    plat_mutex_unlock(&MT.m);
    for (MtChunk *c = MT.sh, *next; c; c = next) {
        next = c->snext;
        if (c->nflag)
            for (int i = 0; i < c->n; i++)
                if (c->d[i].flags) mt_fallback(ex, &c->d[i]);
        c->snext = MT.freel, MT.freel = c;
    }
    MT.sh = MT.st = NULL;
    MT.join_ns += plat_now_ns() - t0, MT.n_join++;
    if (MT.n_join % 600 == 0)
        rt_log("mt: per join %.0f us; draws deferred %llu, same record %llu, flagged entries %llu, serial %llu (feedback loop %llu, nothing to defer %llu), chunks %llu",
               MT.join_ns / 1e3 / (double)MT.n_join, (unsigned long long)MT.n_defer, (unsigned long long)MT.n_same,
               (unsigned long long)MT.n_flag, (unsigned long long)MT.n_inelig, (unsigned long long)MT.n_fbl, (unsigned long long)MT.n_nothing, (unsigned long long)MT.n_chunks);
}

// The walk thread makes the pools the workers could not use ready (what the serial path does at each first use in a
// submission), so later draws on them are filled by workers instead of flagged.
static void mt_drain(void) {
    uint32_t n = __atomic_load_n(&MT.want_n, __ATOMIC_RELAXED);
    if (n - MT.drained > 64) MT.drained = n - 64;
    for (; MT.drained != n; MT.drained++) {
        uint64_t a = MT.want[MT.drained & 63];
        VkMemSpan sp;
        if (a && vk_mem_resolve(a, 1, &sp)) VK_MEM_USE(sp);
    }
}

// One stage's record by deferred fill. 0: not eligible (the caller builds it as before). Walk thread.
static int mt_record(VkExec *ex, const VkPipeInfo *info, int blk, uint32_t nb, uint32_t nt, uint32_t bytes) {
    (void)nb;
    if (ex->nfbl > 9) return MT.n_fbl++, 0;
    if (MT.want_n != MT.drained) mt_drain();
    if (nt > MT_TEX) return MT.n_inelig++, 0;
    uint64_t ub[MT_BUF][2], th[MT_TEX];
    uint32_t nub = 0, ntex = 0;
    for (uint32_t i = 0; i < info->n_bb; i++) {
        const VkPipeBinding *b = &info->bb[i];
        if ((b->stage == GFX_STAGE_FS) != blk) continue;
        int kind = mt_buf_kind(b);
        if (kind) {
            if (nub >= MT_BUF) return 0;
            const p_bind_buf *g = mt_buf_binding(ex, b, kind);
            ub[nub][0] = g->addr, ub[nub][1] = g->size, nub++;
        }
    }
    for (uint32_t i = 0; i < info->n_bl; i++) {
        const VkPipeBinding *b = &info->bl[i];
        if ((b->stage == GFX_STAGE_FS) != blk) continue;
        int st = nvn_stage_of(b->stage), slot = b->res->slot;
        if (ntex >= MT_TEX) return 0;
        th[ntex++] = st >= 0 && slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
    }
    if (!nub && !ntex) return MT.n_nothing++, 0;
    descriptor_state(ex);
    uint8_t rec[512];
    memset(rec, 0, bytes);
    uint64_t ct_e = CST();
    for (uint32_t i = 0; i < info->n_bb; i++) {
        const VkPipeBinding *b = &info->bb[i];
        if ((b->stage == GFX_STAGE_FS) != blk || mt_buf_kind(b)) continue;
        VkDeviceAddress a = record_buffer(ex, b);
        memcpy(rec + 8u * b->binding, &a, 8);
    }
    CSA(SP_E_BUF, ct_e);
    ct_e = CST();
    uint64_t mg = vk_mem_gen(), ug = vk_usnap_gen(), g3 = vkb_gen();
    uint32_t g0 = vk_texture_slots_gen(), g1 = vk_texture_epoch(), g2 = vk_sampler_gen();
    typeof(B.mtl[0]) *l = &B.mtl[blk];
    int fresh = B.bl_info != info;
    int same = l->valid && l->info == info && l->bytes == bytes && l->nub == nub && l->nt == ntex && l->mg == mg && l->ug == ug &&
               l->serial == descriptor_serial && l->g0 == g0 && l->g1 == g1 && l->g2 == g2 && l->g3 == g3 && l->nfbl == (uint32_t)ex->nfbl && !memcmp(l->fbl, ex->fbl_img, sizeof(VkImage) * (size_t)ex->nfbl) && !memcmp(l->rec, rec, bytes) &&
               !memcmp(l->ub, ub, sizeof ub[0] * nub) && !memcmp(l->th, th, sizeof th[0] * ntex);
    if (!same) {
        VkBuffer sb;
        VkDeviceSize so;
        uint8_t *p = scratch(ex, bytes, &sb, &so);
        memcpy(p, rec, bytes);
        l->addr = vkb_buffer_address(sb) + so;
        if (!MT.cur) {
            MT.cur = MT.freel;
            if (MT.cur) MT.freel = MT.cur->snext;
            else if (!(MT.cur = malloc(sizeof *MT.cur))) rt_fatal("vk: out of memory");
            MT.cur->n = 0;
        }
        MtDraw *d = &MT.cur->d[MT.cur->n++];
        d->dst = p, d->info = info, d->mg = mg, d->ug = ug, d->serial = descriptor_serial, d->g3 = g3;
        d->g0 = g0, d->g1 = g1, d->g2 = g2, d->seq = ++MT.seq, d->blk = (uint32_t)blk, d->nub = nub, d->nt = ntex, d->flags = 0;
        d->tex_pool = ex->tex_pool, d->samp_pool = ex->samp_pool;
        d->nfbl = (uint32_t)ex->nfbl;
        memcpy(d->fbl, ex->fbl_img, sizeof(VkImage) * (size_t)ex->nfbl);
        memcpy(d->ub, ub, sizeof ub[0] * nub);
        memcpy(d->th, th, sizeof th[0] * ntex);
        vk_usnap_seq(MT.seq);
        MT.n_defer++;
        if (MT.cur->n == MT_DRAWS) {
            MtChunk *c = MT.cur;
            MT.cur = NULL;
            mt_enqueue(c);
        }
        l->valid = 1, l->info = info, l->bytes = bytes, l->nub = nub, l->nt = ntex, l->mg = mg, l->ug = ug, l->serial = descriptor_serial;
        l->g0 = g0, l->g1 = g1, l->g2 = g2, l->g3 = g3;
        l->nfbl = (uint32_t)ex->nfbl;
        memcpy(l->fbl, ex->fbl_img, sizeof(VkImage) * (size_t)ex->nfbl);
        memcpy(l->rec, rec, bytes), memcpy(l->ub, ub, sizeof ub[0] * nub), memcpy(l->th, th, sizeof th[0] * ntex);
        fresh = 1;
    } else {
        MT.n_same++;
    }
    B.rec_last_n[blk] = 0;   // the serial path's last record is stale now
    CSA(SP_E_TEX, ct_e);
    ct_e = CST();
    if (fresh || B.bl_info != info) {
        vkCmdPushConstants(ex->cb, info->layout, blk ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT, blk ? GFX_BINDLESS_REC_FS : GFX_BINDLESS_REC_VS, 8, &l->addr);
        ex->recorded++;
    }
    CSA(SP_E_PUSH, ct_e);
    return 1;
}

// TOTK_BINDLESS_BUF: the record of each stage (buffer addresses, then texture slots) in the transient arena, its address pushed when it
// differs from the stage's last record. Texture lock held by the caller for the texture lookups.
static void bindless_records(VkExec *ex, const VkPipeInfo *info) {
    for (int blk = 0; blk < 2; blk++) {
        uint32_t nb = info->rec_nb[blk], nt = blk ? info->bl_fs : info->bl_vs;
        if (!nb) continue;
        uint8_t rec[512];
        uint64_t ct_e = CST();
        uint32_t bytes = (nb * 8 + nt * 4 + 7) & ~7u;
        if (MT.active && mt_record(ex, info, blk, nb, nt, bytes)) continue;
        B.mtl[blk].valid = 0;
        memset(rec, 0, bytes);
        for (uint32_t i = 0; i < info->n_bb; i++) {
            const VkPipeBinding *b = &info->bb[i];
            if ((b->stage == GFX_STAGE_FS) != blk) continue;
            VkDeviceAddress a = record_buffer(ex, b);
            memcpy(rec + 8u * b->binding, &a, 8);
        }
        CSA(SP_E_BUF, ct_e);
        ct_e = CST();
        {
            // The slots of the textures only change when a handle, a texture/sampler generation, the table or the feedback-loop
            // set changed: the last resolution of this stage is reused then (tex_info and vkb_slot are skipped).
            struct RecTexCache *rc = &B.rtc[blk];
            uint64_t hs[RTC_MAX];
            uint32_t n = 0, g0 = vk_texture_slots_gen(), g1 = vk_texture_epoch(), g2 = vk_sampler_gen();
            uint64_t g3 = vkb_gen();
            int same = rc->info == info && !ex->nfbl && rc->serial == descriptor_serial && rc->g0 == g0 && rc->g1 == g1 && rc->g2 == g2 &&
                       rc->g3 == g3 && nt <= RTC_MAX;
            for (uint32_t i = 0; i < info->n_bl; i++) {
                const VkPipeBinding *b = &info->bl[i];
                if ((b->stage == GFX_STAGE_FS) != blk) continue;
                int st = nvn_stage_of(b->stage), slot = b->res->slot;
                uint64_t h = st >= 0 && slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
                if (n < RTC_MAX) hs[n] = h;
                if (same && (n >= RTC_MAX || rc->h[n] != h)) same = 0;
                n++;
            }
            if (same) {
                memcpy(rec + 8u * nb, rc->v, 4u * nt);
                rtc_hit++;
            } else {
                rtc_miss++;
                if (!rsp_worker) vk_texture_lock();   // a worker's lookups take the lock only where they need it (tex_info_core)
                for (uint32_t i = 0; i < info->n_bl; i++) {
                    const VkPipeBinding *b = &info->bl[i];
                    if ((b->stage == GFX_STAGE_FS) != blk) continue;
                    const GfxShaderRes *r = b->res;
                    int st = nvn_stage_of(b->stage), slot = r->slot;
                    uint64_t h = st >= 0 && slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
                    VkDescriptorImageInfo ii;
                    tex_info(ex, r, h, 0, r->separate, &ii);
                    int cls = bt_class_of_type(r->type);
                    uint32_t v = 0;
                    if (cls >= 0) {
                        int di = dummy_index(res_view_type(r->type), (r->type & 0x200) != 0);
                        v = vkb_slot((uint32_t)cls, ii.sampler, ii.imageView, ii.imageLayout, r->type & 0x400 ? Z.samp_shadow : Z.samp, Z.img[di].view);
                    }
                    memcpy(rec + 8u * nb + 4u * b->binding, &v, 4);
                }
                if (n <= RTC_MAX && nt <= RTC_MAX && !ex->nfbl) {
                    memcpy(rc->h, hs, 8u * n);
                    memcpy(rc->v, rec + 8u * nb, 4u * nt);
                    rc->info = info, rc->serial = descriptor_serial, rc->g0 = g0, rc->g1 = g1, rc->g2 = g2, rc->g3 = g3;
                } else {
                    rc->info = NULL;
                }
                if (!rsp_worker) vk_texture_unlock();
            }
        }
        { static int dr = -1; if (dr < 0) { const char *e = getenv("TOTK_DRAW_REUSE"); dr = e && !strcmp(e, "1"); } if (dr) draw_reuse_probe(ex, info, blk, rec, nb, bytes); }
        CSA(SP_E_TEX, ct_e);
        ct_e = CST();
        int fresh = B.bl_info != info;
        if (fresh || B.rec_last_n[blk] != bytes || memcmp(B.rec_last[blk], rec, bytes)) {
            VkBuffer sb;
            VkDeviceSize so;
            memcpy(scratch(ex, bytes, &sb, &so), rec, bytes);
            B.rec_last_addr[blk] = vkb_buffer_address(sb) + so;
            memcpy(B.rec_last[blk], rec, bytes), B.rec_last_n[blk] = bytes;
            fresh = 1;
        } else {
            fresh = 0;
        }
        if (fresh || B.bl_info != info) {
            vkCmdPushConstants(ex->cb, info->layout, blk ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT, blk ? GFX_BINDLESS_REC_FS : GFX_BINDLESS_REC_VS, 8,
                               &B.rec_last_addr[blk]);
            ex->recorded++;
        }
        CSA(SP_E_PUSH, ct_e);
    }
}

static void bindless_draw(VkExec *ex, const VkPipeInfo *info) {
    if (info->rec_nb[0] || info->rec_nb[1]) {
        if (B.bl_layout != info->layout) {
            VkDescriptorSet set = vkb_set();
            vkCmdBindDescriptorSets(ex->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, info->layout, VKB_SET, 1, &set, 0, NULL);
            B.bl_layout = info->layout;
            ex->recorded++;
        }
        bindless_records(ex, info);
        B.bl_info = info;
        return;
    }
    uint32_t pc[2][GFX_BINDLESS_SLOTS_FS] = {{0}};
    vk_texture_lock();
    for (uint32_t i = 0; i < info->n_bl; i++) {
        const VkPipeBinding *b = &info->bl[i];
        const GfxShaderRes *r = b->res;
        int st = nvn_stage_of(b->stage), slot = r->slot, blk = b->stage == GFX_STAGE_VS ? 0 : 1;
        uint64_t h = st >= 0 && slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
        VkDescriptorImageInfo ii;
        tex_info(ex, r, h, 0, r->separate, &ii);
        if (census_on()) census_tex(st, slot, 0, h);
        int cls = bt_class_of_type(r->type);
        uint32_t v = 0;
        if (cls >= 0) {
            int di = dummy_index(res_view_type(r->type), (r->type & 0x200) != 0);
            v = vkb_slot((uint32_t)cls, ii.sampler, ii.imageView, ii.imageLayout, r->type & 0x400 ? Z.samp_shadow : Z.samp, Z.img[di].view);
        }
        pc[blk][b->binding] = v;
    }
    vk_texture_unlock();
    static int dbg = -1;   // DKB_BL_DEBUG bits: 1 no set bind, 2 no push constants
    if (dbg < 0) dbg = getenv("DKB_BL_DEBUG") ? atoi(getenv("DKB_BL_DEBUG")) : 0;
    if (dbg & 1) goto pushes;
    if (B.bl_layout != info->layout) {   // the table is bound per layout: a layout that is not stage-independent (a GS pipeline) disturbs set VKB_SET
        VkDescriptorSet set = vkb_set();
        vkCmdBindDescriptorSets(ex->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, info->layout, VKB_SET, 1, &set, 0, NULL);
        B.bl_layout = info->layout;
        ex->recorded++;
    }
pushes:;
    int fresh = B.bl_info != info;
    if (dbg & 2) { B.bl_info = info; return; }
    if (info->bl_vs && (fresh || memcmp(B.bl_pc[0], pc[0], 4u * info->bl_vs))) {
        vkCmdPushConstants(ex->cb, info->layout, VK_SHADER_STAGE_VERTEX_BIT, GFX_BINDLESS_PUSH_VS, 4u * info->bl_vs, pc[0]);
        memcpy(B.bl_pc[0], pc[0], 4u * info->bl_vs);
        ex->recorded++;
    }
    if (info->bl_fs && (fresh || memcmp(B.bl_pc[1], pc[1], 4u * info->bl_fs))) {
        vkCmdPushConstants(ex->cb, info->layout, VK_SHADER_STAGE_FRAGMENT_BIT, GFX_BINDLESS_PUSH_FS, 4u * info->bl_fs, pc[1]);
        memcpy(B.bl_pc[1], pc[1], 4u * info->bl_fs);
        ex->recorded++;
    }
    B.bl_info = info;
}

// TOTK_BINDLESS_TEX=check: the table slot of a texture the replay just resolved (through the memo) must hold what a lookup that
// skips the memo resolves (tex_info_full); the table is entered the same way the bindless path would (vkb_slot). Texture lock held.
static void bindless_check(VkExec *ex, const GfxShaderRes *r, uint64_t h, const VkDescriptorImageInfo *got) {
    int cls = bt_class_of_type(r->type);
    if (cls < 0) return;
    int ms = (r->type & 0x200) != 0;
    int di = dummy_index(res_view_type(r->type), ms);
    VkSampler dsamp = r->type & 0x400 ? Z.samp_shadow : Z.samp;
    uint32_t slot = vkb_slot((uint32_t)cls, got->sampler, got->imageView, got->imageLayout, dsamp, Z.img[di].view);
    VkDescriptorImageInfo full;
    VkImage image;
    tex_info_full(ex, r, h, 0, 0, &full, &image);
    if (slot) vkb_verify((uint32_t)cls, slot, full.sampler, full.imageView, full.imageLayout);
}

static void bind_resources(VkExec *ex, const VkPipeInfo *info) {
    static _Thread_local VkWriteDescriptorSet w[MAX_BINDINGS];
    static _Thread_local VkDescSlot d[MAX_DESC];
    static _Thread_local VkBufferView bv[MAX_DESC];
    if (info->n_bindings > MAX_BINDINGS) rt_fatal("vk: %u descriptor bindings exceed the replay's %d", info->n_bindings, MAX_BINDINGS);
    bind_stats(info);
    uint64_t ct_res = CST(), ct;
    int gfx = info->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS;
    if (gfx && census_on()) {
        unsigned nds = 0;
        for (uint32_t i = 0; i < info->n_bindings; i++) nds += info->bindings[i].count ? info->bindings[i].count : 1;
        census_draw(info->n_bindings, nds);
    }
    int skip = gfx ? bind_skip_on() : 0, same = 0, was_valid = B.res_valid;
    if (skip) {
        ct = CST();
        same = bind_sig(ex, info);
        CSA(SP_RES_SIG, ct);
        if (same && was_valid && skip == 1 && ((!info->n_bl && !info->n_bb) || B.bl_info == info)) {
            if (census_on()) census_desc(6);
            for (int i = 0; i < RS.ncnt; i++) vk_mem_count(RS.cnt[i].cls, RS.cnt[i].use, RS.cnt[i].bytes);
            BS.bind_skip++;
            ex->recorded++;
            CSA(SP_RES, ct_res);
            return;
        }
        B.res_valid = 0;   // until the build below ends
        if (skip == 2) memcpy(RS.pcnt, RS.cnt, sizeof RS.cnt), RS.pncnt = RS.ncnt;
        RS.rec = 1, RS.ncnt = 0, RS.cnt_over = 0, RS.volatile_ = 0, RS.cmds = 0;
    }
    uint32_t n_sets = info->n_sets ? info->n_sets : 1;
    uint32_t lo[2] = {0, 0}, hi[2] = {0, 0}, d0[2] = {0, 0};   // bindings [lo, hi) and first descriptor per set
    int need_w = 0;                                             // a set without a template: fill the write structs
    for (uint32_t s = 0, i0 = 0; s < n_sets; s++) {
        uint32_t i1 = i0;
        while (i1 < info->n_bindings && (n_sets == 1 || info->bindings[i1].set == s)) i1++;
        lo[s] = i0, hi[s] = i1, i0 = i1;
        if (i1 > lo[s] && !info->tpl[s]) need_w = 1;
    }
    uint32_t nd = 0;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        const GfxShaderRes *r = b->res;
        uint32_t cnt = b->count ? b->count : 1;
        if (nd + cnt > MAX_DESC) rt_fatal("vk: %u descriptors exceed the replay's %d", nd + cnt, MAX_DESC);
        if (n_sets == 2 && i == lo[1]) d0[1] = nd;
        VkDescSlot *o = &d[nd];
        int st = nvn_stage_of(b->stage), kind = 0;   // kind: 0 buffer, 1 image, 2 texel buffer view
        if (st < 0) rt_fatal("vk_exec: unsupported descriptor stage %u", b->stage);
        ct = CST();
        if (b->kind == VK_PIPE_SUPPORT) {
            float in[3];
            support_in(ex, in);
            if (!B.sup_valid || memcmp(in, B.sup_in, sizeof in)) {   // support_fill reads only these
                uint8_t tmp[SUPPORT_BYTES];
                support_fill(tmp, in[0], in[1], in[2]);
                memcpy(B.sup_in, in, sizeof in);
                if (!B.sup_valid || memcmp(tmp, B.support, SUPPORT_BYTES)) {   // a new buffer only when the contents changed
                    VkBuffer sb;
                    VkDeviceSize so;
                    memcpy(scratch(ex, SUPPORT_BYTES, &sb, &so), tmp, SUPPORT_BYTES);
                    memcpy(B.support, tmp, SUPPORT_BYTES);
                    B.sup_buf = sb, B.sup_off = so, B.sup_valid = 1;
                }
            }
            o->buffer = (VkDescriptorBufferInfo){B.sup_buf, B.sup_off, SUPPORT_BYTES};
        } else if (b->kind == GFX_RES_UBO) {
            for (uint32_t k = 0; k < cnt; k++) {
                o[k].buffer = ubo_info(ex, st, r->slot + (int)k);
                if (census_on()) census_buf(0, st, r->slot + (int)k, (uint64_t)(uintptr_t)o[k].buffer.buffer * 1000003ull ^ o[k].buffer.offset, o[k].buffer.range);
            }
        } else if (b->kind == GFX_RES_SSBO) {
            for (uint32_t k = 0; k < cnt; k++) {
                o[k].buffer = ssbo_info(ex, st, r, r->flags & 1);
                if (census_on()) census_buf(1, st, r->slot, (uint64_t)(uintptr_t)o[k].buffer.buffer * 1000003ull ^ o[k].buffer.offset, o[k].buffer.range);
            }
        } else if (b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) {
            int img = b->kind == GFX_RES_IMG;
            if (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || b->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                kind = 2;
                for (uint32_t k = 0; k < cnt; k++) {   // buffer textures are not replayed yet
                    o[k].w[0] = 0, o[k].w[1] = 0, o[k].w[2] = 0;
                    o[k].view = Z.zview;
                    if (need_w) bv[nd + k] = Z.zview;
                }
            } else {
                kind = 1;
                vk_texture_lock();   // tex_info looks up with the texture lock held
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    uint64_t h = 0;
                    if (img) h = slot >= 0 && slot < 32 ? ex->img[st][slot] : 0;
                    else h = slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
                    if (census_on()) census_tex(st, slot, img, h);
                    tex_info(ex, r, h, img, r->separate, &o[k].image);
                    if (gfx && !img && !r->separate && vkb_mode() == 2) bindless_check(ex, r, h, &o[k].image);   // TOTK_BINDLESS_TEX=check
                    o[k].w[2] = (uint32_t)o[k].image.imageLayout;   // zero the tail padding
                }
                vk_texture_unlock();
            }
        } else {
            rt_fatal("vk: binding %u has an unknown kind %d", (unsigned)b->binding, (int)b->kind);
        }
        CSA(kind == 1 ? SP_RES_TEX : SP_RES_BUF, ct);
        if (need_w) {
            w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstBinding = b->binding, .descriptorCount = cnt, .descriptorType = b->type};
            if (kind == 0) w[i].pBufferInfo = &o->buffer;   // a VkDescSlot is as large as either info struct
            else if (kind == 1) w[i].pImageInfo = &o->image;
            else w[i].pTexelBufferView = &bv[nd];
        }
        nd += cnt;
    }
    ct = CST();
    // Set 1's first descriptor is noted when the loop reaches its first binding. A set 1 without bindings (a fragment
    // stage with no resources) starts at the end instead, so set 0's byte count (d0[1] - d0[0]) covers its descriptors.
    // Left at 0, it pushed set 0's template with 0 bytes copied into the recording, and the driver read the
    // descriptors from whatever followed in the recording.
    if (n_sets == 2 && lo[1] == info->n_bindings) d0[1] = nd;
    // Under the validation layer, a new graphics layout binds set 1 before set 0 is pushed or bound. The layer (SDK
    // 1.4.363, PipelineLayoutCompatDef::operator==) takes any two INDEPENDENT_SETS layouts as compatible, so
    // vkCmdPushDescriptorSetKHR keeps the push set it made for an earlier layout and drops writes to bindings that
    // layout lacks (false VUID-vkCmdDrawIndexed-None-08114, then an access violation inside the layer). A bind at set 1
    // disturbs set 0 by its canonical id, which drops the layer's push set: the push after it gets a new one.
    int set1_first = 0;
    if (gfx && info->layout != B.ds_layout) {   // another layout: sets bound before may be disturbed
        B.ds_layout = info->layout;
        B.ds_bound[0] = B.ds_bound[1] = VK_NULL_HANDLE;
        B.ds_pushed[0] = B.ds_pushed[1] = 0;
        B.bl_layout = VK_NULL_HANDLE, B.bl_info = NULL;
        set1_first = n_sets == 2 && vk_ctx()->validation;
    }
    for (uint32_t k = 0; k < n_sets; k++) {
        uint32_t s = set1_first ? 1 - k : k, i0 = lo[s], i1 = hi[s];
        const VkDescSlot *ds = &d[d0[s]];
        uint32_t dbytes = (uint32_t)(((s + 1 < n_sets ? d0[s + 1] : nd) - d0[s]) * sizeof *d);
        if (!s && i1 == i0 && info->n_bb) continue;   // TOTK_BINDLESS_BUF: every buffer is in the record, nothing is left in set 0
        if (s && i1 == i0) {               // no fragment stage: set 1 is empty and unused
            if (set1_first) {              // bound only to drop the layer's push set (comment above)
                VkDescriptorSet set = alloc_set(ex, info->set_layout1);
                vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, 1, 1, &set, 0, NULL);
                B.ds_bound[1] = set, B.ds_pushed[1] = 0;
                RS.cmds++;
            }
            continue;
        }
        if (s ? info->push1 : info->push) {
            uint32_t n = gfx && push_skip_on() ? dbytes : 0;
            if (n && B.ds_pushed[s] && DS.pn[s] == n && !memcmp(DS.psig[s], ds, n)) {
                BS.push_skip++;
                if (census_on()) census_desc(1);
                continue;
            }
            if (info->tpl[s]) rec_push_tpl(vk_pipeline_push_tpl_fn(), ex->cb, info->tpl[s], info->layout, s, ds, dbytes);
            else rec_push(vk_pipeline_push_fn(), ex->cb, info->bind_point, info->layout, s, i1 - i0, w + i0);
            BS.push++;
            if (census_on()) census_desc(0);
            RS.cmds++;
            if (gfx) {
                B.ds_bound[s] = VK_NULL_HANDLE, B.ds_pushed[s] = n != 0;
                if (n) DS.pn[s] = n, memcpy(DS.psig[s], ds, n);
            }
        } else if (!gfx || s > 1) {
            VkDescriptorSet set = alloc_set(ex, s ? info->set_layout1 : info->set_layout);
            write_set(info, s, set, ds, w, i0, i1);
            vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, s, 1, &set, 0, NULL);
        } else {
            VkDescriptorSetLayout sl = s ? info->set_layout1 : info->set_layout;
            VkDescriptorSet set = B.ds_last[s];
            if (!set || B.ds_last_layout[s] != sl || DS.n[s] != dbytes || memcmp(DS.sig[s], ds, dbytes)) {
                set = alloc_set(ex, sl);
                write_set(info, s, set, ds, w, i0, i1);
                B.ds_last[s] = set, B.ds_last_layout[s] = sl, DS.n[s] = dbytes;
                memcpy(DS.sig[s], ds, dbytes);
                BS.set_new++;
                RS.cmds++;
            } else {
                BS.set_reuse++;
                if (census_on()) census_desc(3);
            }
            if (B.ds_bound[s] != set) {
                vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, s, 1, &set, 0, NULL);
                B.ds_bound[s] = set, B.ds_pushed[s] = 0;
                RS.cmds++;
            }
        }
    }
    if (gfx && (info->n_bl || info->n_bb)) bindless_draw(ex, info);
    else if (gfx) B.bl_layout = VK_NULL_HANDLE, B.bl_info = NULL;   // a layout without the table disturbs its set
    CSA(SP_RES_EMIT, ct);
    if (skip) {
        RS.rec = 0;
        // An equal next draw records nothing when every push set holds its push (TOTK_PUSH_SKIP on) and nothing made
        // the build unrepeatable; allocated sets are always bound to the one written last.
        int valid = !RS.volatile_ && !RS.cnt_over && RS.hn;
        for (uint32_t s = 0; s < n_sets; s++)
            if (hi[s] > lo[s] && (s ? info->push1 : info->push) && !B.ds_pushed[s]) valid = 0;
        if (skip == 2) {   // with the pipeline info equal (bind_sig), the packed descriptors determine every write
            uint32_t n = (uint32_t)(nd * sizeof *d);
            int wr = n != RS.ckn || memcmp(RS.ck, d, n);
            int cn = RS.ncnt != RS.pncnt || memcmp(RS.cnt, RS.pcnt, sizeof RS.cnt[0] * (size_t)RS.ncnt);
            if (same && was_valid && (wr || cn || RS.cmds))
                rt_fatal("vk_exec: descriptor build skip check: a draw of program %llu with the inputs of the previous one "
                         "gets %s", (unsigned long long)ex->prog->serial,
                         RS.cmds ? "descriptor commands" : wr ? "other writes" : "other census counts");
            if (same && was_valid) BS.bind_skip++;   // the draws a skip would have spared, compared instead
            RS.ckn = n;
            memcpy(RS.ck, d, n);
        }
        B.res_valid = valid;
    }
    ex->recorded++;
    CSA(SP_RES, ct_res);
}

// ---- draws

static void bind_pipeline(VkExec *ex, VkPipelineBindPoint bp, VkPipeline p) {
    int k = bp == VK_PIPELINE_BIND_POINT_COMPUTE;
    if (B.pipe[k] == p) return;
    if (k) B.bl_info = NULL;   // another layout's push constants (a compute pipeline) follow: the slot blocks are pushed again
    vkCmdBindPipeline(ex->cb, bp, p);
    ex->recorded++;
    B.pipe[k] = p;
    if (!k) B.st_valid = 0;
}

static void record_fixed(VkExec *ex) {
    VkCommandBuffer cb = ex->cb;
    if (ex->dirty & VK_EXEC_DIRTY_VIEWPORT) {
        float x = 0, y = 0, w = (float)ex->pass_w, h = (float)ex->pass_h;
        if (ex->has_vp) x = ex->vp[0][0], y = ex->vp[0][1], w = ex->vp[0][2], h = ex->vp[0][3];
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        VkViewport v = {x, y + h, w, -h, ex->depth_range[0], ex->depth_range[1]};   // the y flip: negative height
        vkCmdSetViewportWithCount(cb, 1, &v);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_SCISSOR) {
        VkRect2D r = vk_exec_scissor(ex);
        vkCmdSetScissorWithCount(cb, 1, &r);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_DEPTH_BIAS) {
        vkCmdSetDepthBias(cb, ex->poly_offset.units, ex->poly_offset.clamp, ex->poly_offset.factor);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_BLEND_CONST) {
        vkCmdSetBlendConstants(cb, ex->blend_color);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_STENCIL) {
        for (int f = 0; f < 2; f++) {
            VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
            vkCmdSetStencilCompareMask(cb, face, ex->st_vmask[f]);
            vkCmdSetStencilWriteMask(cb, face, ex->st_wmask[f]);
            vkCmdSetStencilReference(cb, face, ex->st_ref[f]);
            ex->recorded += 3;
        }
    }
    ex->dirty = 0;
}

// A draw rewritten by DKB_LQ_INTERLEAVE (light_quad_match, lq_interleave_plan below): n streams at locations loc[], whose
// instances [0, first+count) are copied into one interleaved buffer. fmt: the NVN format word of each attribute.
typedef struct {
    unsigned n;
    uint32_t loc[VK_LQ_MAX_STREAMS], fmt[VK_LQ_MAX_STREAMS], first, count;
    const uint8_t *src[VK_LQ_MAX_STREAMS];
} LqPlan;

// ps: the pipeline state of the draw (its vstream strides). lq: the interleave plan, whose buffer replaces binding 0.
static void bind_vertex_buffers(VkExec *ex, const VkPipeState *ps, const LqPlan *lq) {
    VkBuffer vb[17];
    VkDeviceSize vo[17], vsz[17], vstr[17];
    int vcls[17] = {0};
    void *vpool[17] = {0};
    int dyn = vk_pipeline_vertex_input_dynamic();
    // A stream whose range is the one its bound buffer was resolved from, with no pool or mirror change since
    // (vk_mem_gen, read before any resolve), resolves to that buffer again: most draws change one stream or none.
    uint64_t gen = vk_mem_gen();
    int same_gen = B.vb_valid && B.vgen == gen;
    for (int b = 0; b < 17; b++) {
        vb[b] = Z.zbuf, vo[b] = 0, vsz[b] = ZERO_BYTES, vstr[b] = b < 16 ? ps->vstream[b].stride : 0;
        if (b == VK_PIPE_ZERO_BINDING) continue;
        const NVNbufferRange *r = &ex->vbuf[b];
        if (same_gen && B.vin[b].address == r->address && B.vin[b].size == r->size) {
            vb[b] = B.vb[b], vo[b] = B.vo[b], vsz[b] = B.vsz[b], vcls[b] = B.vcls[b], vpool[b] = B.vpool[b];
            continue;
        }
        VkMemSpan sp;
        if (!r->address || !vk_mem_resolve(r->address, 1, &sp)) continue;
        VK_MEM_USE(sp);
        uint64_t n = r->size && r->size < sp.avail ? r->size : sp.avail;
        vb[b] = sp.buffer, vo[b] = sp.offset, vsz[b] = n, vcls[b] = sp.cls, vpool[b] = sp.pool;
    }
    if (lq) {   // binding 0 is the interleaved copy; the bound-buffer cache is dropped below, so no later draw reuses it
        uint64_t bytes = 0;
        vk_lq_interleave_bytes(lq->first, lq->count, lq->n, &bytes);
        VkBuffer sb;
        VkDeviceSize so;
        uint8_t *m = scratch(ex, bytes, &sb, &so);
        vk_lq_interleave_copy(m, lq->src, lq->n, lq->first, lq->count);
        vb[0] = sb, vo[0] = so, vsz[0] = bytes, vcls[0] = 0, vpool[0] = NULL;
    }
    memcpy(B.vcls, vcls, sizeof vcls), memcpy(B.vpool, vpool, sizeof vpool);
    memcpy(B.vin, ex->vbuf, sizeof B.vin);
    B.vgen = gen;
    if (!lq && B.vb_valid && !memcmp(B.vb, vb, sizeof vb) && !memcmp(B.vo, vo, sizeof vo) && !memcmp(B.vsz, vsz, sizeof vsz) &&
        (dyn || !memcmp(B.vstr, vstr, sizeof vstr)))
        return;
    vkCmdBindVertexBuffers2(ex->cb, 0, 17, vb, vo, vsz, dyn ? NULL : vstr);
    ex->recorded++;
    for (int b = 0; b < 17; b++)
        if (vcls[b]) vk_mem_count(vcls[b], VK_MUSE_VB, vsz[b]);
    if (rt_trace & RT_TRACE_PERF)
        for (int b = 0; b < 17; b++)
            if (vpool[b]) vk_mem_bound_note(vpool[b]);
    memcpy(B.vb, vb, sizeof vb), memcpy(B.vo, vo, sizeof vo), memcpy(B.vsz, vsz, sizeof vsz), memcpy(B.vstr, vstr, sizeof vstr);
    B.vb_valid = !lq;
}

// TOTK_PIPE_SKIP: which draws are skipped, rather than waited for, while their shaders or pipeline build.
// strict (0): only draws whose pass targets were all rendered in the previous presented frame; a pass with a target
//   that was not counts as one-shot and waits.
// any (1): also draws in a pass with at least one target rendered in the previous frame (a per-frame depth
//   buffer or g-buffer with a colour target the frame before did not use: half-resolution effect buffers in fights).
//   Passes with none (volumes, LUTs, shadow layers seen for the first time) still wait.
// fullscreen (2, default): as any, and full-screen draws into such passes are skipped instead of waiting from their
//   budget. A full-screen pass whose stage libraries are cold in the driver cache took 86 ms on average (up to 194 ms)
//   in a fight on an RX 9070: three frames in a row each spent the 50 ms budget, and some of the draws were skipped
//   after all. Skipping costs the pass for the frames until the pipeline is ready.
static int pipe_skip_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = cfg_value("TOTK_PIPE_SKIP");
        mode = !strcmp(e, "strict") || !strcmp(e, "0") ? 0 : !strcmp(e, "fullscreen") || !strcmp(e, "2") ? 2 : 1;
        static const char *const what[3] = {"only draws into per-frame passes (TOTK_PIPE_SKIP=strict)",
                                            "draws into passes with a target the previous frame rendered",
                                            "those and full-screen draws (TOTK_PIPE_SKIP=fullscreen)"};
        rt_log("vk_exec: draws skipped while their pipeline builds: %s", what[mode]);
    }
    return mode;
}

// The previous draw's shaders, kept while its program, bound pipeline state, the vertex and fragment stages' first 32
// texture and image handles and every slot's resolution (vk_texture_slots_gen, read unlocked: a registration racing
// with it is ordered after this draw, as answers_for's own check would order it) stay the same: answers_for reads
// nothing else, so shader_for would return the same shaders. About 65% of field draws repeat them
// (TOTK_BIND_STATS). TOTK_SHADER_MEMO=0 turns it off; =check looks the shaders up on every hit too, and every slot's
// texture type in answers_for, and stops on a difference.
// tex_serial (bumped by every texture and image bind) equal to the key's means the handles were not rebound since
// they were compared, so the 1 KiB compare is skipped; a rebind to the same handles still matches by the compare.
typedef struct {
    const prog_t *prog;
    uint64_t serial;
    uint32_t gen, tex_serial;
    uint64_t tex[3][32], img[3][32];
    uint8_t geometry_input;
    VkPipeState pipe;
} MemoKey;

// Workaround for the GPU fault after the first guide movie (#9). On NVIDIA (drivers 617.14, 617.42 and 621.01) the
// draws of Bananza's instanced light quads whose vertex stage fetches ten per-instance streams (an input at Location
// 9) and whose fragment stage reads the storage buffer at c0+0x530 fault the device with an invalid instruction
// pointer, or crash the driver's shader compiler. Leaving out one such shader moves the fault to another of the
// family, and the same SPIR-V drawn on its own does not fault, so the cause is still open. Skipping these draws
// loses those lights; the family's quads without an input at Location 9 draw as before. It is a mitigation, not a fix:
// in 600 s runs on an RTX 5060 Ti the fault came minutes later in one run of two (another shader), where without it
// every run faulted right after the first guide movie. TOTK_FIX_VS_FAULT=0 draws them.
//
// The later fault, around seven minutes in, comes from one more quad of the same shape whose fragment stage does not
// read c0+0x530: NVN 2 vertex stage cae0ef8e9fe9147f, nine per-instance RGBA32F streams, one instance. In 9 runs of
// 600 s every run where it drew faulted, with its draw among those in flight at the device loss, and the runs where it
// never drew did not; skipping it gave 0 faults in 4 runs. Quads that never fault have its shape: vertex stages
// 782b2585777d2b4e, 392e2956038a9e64 and 7d85669ac430f01a read Location 9, feed every attribute per instance and carry
// the same integer code (four reads of c9 at a computed index), and skipping every such quad left out ten vertex stages
// that drew in all 17 logged runs. So it is matched by the name of its NVN 2 stage.
static int lq_platform(void) {
    static int on = -1;
    if (on < 0) on = !rt_game_version && vk_ctx()->props.vendorID == 0x10de;   // 0: Bananza, on NVIDIA
    return on;
}

// What the workaround matches: 1 a known faulting NVN 2 vertex stage, 2 the family (fragment stage reads c0+0x530, vertex
// stage reads Location 9), 0 neither. light_quad_fault_skip skips these draws; lq_interleave_* rewrites them.
static int light_quad_match_uncached(const VkExec *ex, const GfxShader *vs, const GfxShader *fs);
static int light_quad_match(const VkExec *ex, const GfxShader *vs, const GfxShader *fs) {
    if (!lq_platform() || !vs || !fs) return 0;
    if (!fastpath_on()) return light_quad_match_uncached(ex, vs, fs);
    int v;
    if (dc_memo_get(&B.lqm, ex->prog, vs, fs, 0, &v)) return v;
    v = light_quad_match_uncached(ex, vs, fs);
    dc_memo_put(&B.lqm, ex->prog, vs, fs, 0, v);
    return v;
}
static int light_quad_match_uncached(const VkExec *ex, const GfxShader *vs, const GfxShader *fs) {
    const uint8_t *ctl, *code = NULL;
    uint32_t len;
    if (find_stage(ex->prog, NVN_STAGE_VS, &ctl, &code, &len)) {
        uint64_t origin = nvn2_origin(code);
        // cae0ef8e9fe9147f: the fault seven minutes in (above); still skipped by name. 7a76ae8de5016fd9: the Ice stage's quad,
        // a four-vertex strip with ten per-instance streams whose fragment stage does not read c0+0x530, so the family match
        // below misses it. Creating its pipeline crashed the driver's compiler (nvgpucomp64.dll, reproduced from the captured
        // pipeline alone), and with that avoided its draws lost the device. Its vertex stage is now folded by default
        // (gfx_shader.c lq_fold_rules, not exact): the folded module compiled without the crash and drew for 90 s without
        // device loss. This match stays as the fallback: light_quad_fault_skip lifts the skip only for a module that
        // gfx_shader_folded reports, which is decided when the shader is loaded, before this decision and before the pipeline
        // is built, so if the fold fails closed the draws are still skipped here.
        if (origin == 0xcae0ef8e9fe9147full || origin == 0x7a76ae8de5016fd9ull) return 1;
    }
    const GfxShaderRes *r;
    int n = gfx_shader_resources(fs, GFX_RES_SSBO, &r), lights = 0;
    for (int i = 0; i < n; i++) lights |= r[i].cbuf_slot == 0 && r[i].handle * 4 == 0x530;
    GfxShaderIo io[32];
    if (!lights || !(gfx_shader_inputs(vs, io) >> 9 & 1)) return 0;
    return 2;
}

// The fold's fail-safe, for every stage with a fold rule (light quad or not: Fort's is not): a vertex module that should have been
// folded and was not is never drawn, whatever path loaded it (gfx_shader.c "the fold, whichever path loaded the module"). Logged once
// per stage.
static int lq_fold_missing(const VkExec *ex, const GfxShader *vs) {
    if (!lq_platform() || !vs || !ex->prog) return 0;
    int v;
    if (fastpath_on() && dc_memo_get(&B.fmm, ex->prog, vs, NULL, 0, &v)) return v;
    const uint8_t *ctl, *code = NULL;
    uint32_t len;
    uint64_t origin = find_stage(ex->prog, NVN_STAGE_VS, &ctl, &code, &len) ? nvn2_origin(code) : 0;
    v = lq_fold_missing_decision(1, gfx_shader_fold_origin(origin), gfx_shader_folded(vs));
    if (v) {
        static uint64_t told[16];
        static int ntold;
        int seen = 0;
        for (int i = 0; i < ntold; i++) seen |= told[i] == origin;
        if (!seen && ntold < 16) {
            told[ntold++] = origin;
            rt_log("vk_exec: draws with vertex stage %016llx are skipped: its module should have been folded and was not",
                   (unsigned long long)origin);
        }
    }
    dc_memo_put(&B.fmm, ex->prog, vs, NULL, 0, v);
    return v;
}

// Diagnostics for the NVIDIA device faults whose draw is not a light quad (Fort, Tower; DIAGNOSTICS.md "Draw first-seen log and
// skip switches"). All off by default and only on Bananza + NVIDIA.
//   DKB_DRAW_FIRSTSEEN=1  one line per (vertex origin, fragment origin) pair, the first time it is drawn, and a heartbeat per
//                         world second ("sec=N firstseen=K"). Every rt_log line is a separate write to the log file, so the
//                         lines survive a device loss.
//   DKB_SKIP_ORIGINS=<hex prefixes>  skip the draws whose vertex or fragment origin starts with one of them (logged once each).
//   DKB_SKIP_BISECT=<mask>:<value>[,<mask>:<value>...]   skip the draws whose vertex origin has (origin & mask) == value for
//                         any of the terms; counted per second.
//   DKB_SKIP_BISECT_FS=<mask>:<value>  the same for the fragment origin; a draw matching either is skipped.
// The skip happens after the shaders are looked up and before the pipeline is built.
#define DD_PAIRS 8192
static int draw_diag(const VkExec *ex, const GfxShader *vs, const GfxShader *fs, int prim, const p_draw *d, const p_draw_idx *di) {
    static int on = -1, first_on;
    static DfOrigins list;
    static DfBisect vb, fb;
    static uint64_t pv[DD_PAIRS], pf[DD_PAIRS];
    static unsigned npairs;
    static uint64_t told[DF_MAX_ORIGINS];
    static unsigned ntold;
    static int64_t sec = -1;
    static unsigned sec_skips, sec_skips_fs;
    static uint64_t t0;
    if (on < 0) {
        const char *f = getenv("DKB_DRAW_FIRSTSEEN"), *o = getenv("DKB_SKIP_ORIGINS"), *b = getenv("DKB_SKIP_BISECT"),
                   *bf = getenv("DKB_SKIP_BISECT_FS");
        first_on = f && strcmp(f, "0");
        if (o && *o && df_parse_origins(&list, o) < 0) {
            rt_log("draw-diag: DKB_SKIP_ORIGINS=%s is not a comma-separated list of 1..16 hex digit prefixes (at most %d); ignored", o, DF_MAX_ORIGINS);
            list.n = 0;
        }
        if (b && *b) {
            if (df_parse_bisect_list(&vb, b) < 0 || !vb.on) rt_log("draw-diag: DKB_SKIP_BISECT=%s is not a comma-separated list of <hex mask>:<hex value> terms (value inside the mask, at most %d); ignored", b, DF_MAX_TERMS);
        }
        if (bf && *bf) {
            if (df_parse_bisect_list(&fb, bf) < 0 || !fb.on) rt_log("draw-diag: DKB_SKIP_BISECT_FS=%s is not a comma-separated list of <hex mask>:<hex value> terms (value inside the mask, at most %d); ignored", bf, DF_MAX_TERMS);
        }
        on = lq_platform() && (first_on || list.n || vb.on || fb.on);
        if (on)
            rt_log("draw-diag: firstseen %d, skip origins %u, bisect %s, fs bisect %s", first_on, list.n, vb.on ? b : "off", fb.on ? bf : "off");
    }
    if (!on || !vs) return 0;
    const uint8_t *ctl, *code = NULL;
    uint32_t len;
    uint64_t vo = find_stage(ex->prog, NVN_STAGE_VS, &ctl, &code, &len) ? nvn2_origin(code) : 0, fo = 0;
    code = NULL;
    if (fs && find_stage(ex->prog, NVN_STAGE_FS, &ctl, &code, &len)) fo = nvn2_origin(code);
    int64_t w = frame_step_world_us(), now = w < 0 ? -1 : w / 1000000;
    if (!t0) t0 = plat_now_ns();
    if (now != sec) {
        if (sec_skips) rt_log("draw-skip: sec=%lld bisect skipped=%u", (long long)sec, sec_skips);
        if (sec_skips_fs) rt_log("draw-skip: sec=%lld bisect-fs skipped=%u", (long long)sec, sec_skips_fs);
        if (first_on && now >= 0)
            rt_log("draw-firstseen: sec=%lld firstseen=%u wall=%.1f", (long long)now, npairs, (double)(plat_now_ns() - t0) / 1e9);
        sec = now, sec_skips = sec_skips_fs = 0;
    }
    if (first_on && vo) {
        unsigned i = 0;
        for (; i < npairs && !(pv[i] == vo && pf[i] == fo); i++) {}
        if (i == npairs && npairs < DD_PAIRS) {
            pv[npairs] = vo, pf[npairs] = fo, npairs++;
            GfxShaderIo io[32];
            uint32_t in = gfx_shader_inputs(vs, io), streams = 0;
            for (unsigned l = 0; l < 16; l++)
                if (in >> l & 1) streams |= 1u << (ex->pipe.vattr[l] >> 27 & 31);
            int32_t inst = d ? d->instances : di ? di->instances : 0;
            rt_log("draw-firstseen: n=%u t=%.3f wall=%.1f vs=%016llx fs=%016llx prim=%d attrs=%d streams=%d inst=%d loc9=%d", npairs,
                   w < 0 ? -1.0 : (double)w / 1e6, (double)(plat_now_ns() - t0) / 1e9, (unsigned long long)vo, (unsigned long long)fo, prim,
                   __builtin_popcount(in & 0xffff), __builtin_popcount(streams), inst > 0 ? inst : 1, (int)(in >> 9 & 1));
        }
    }
    int why;
    if (!df_skip(&list, &vb, &fb, vo, fo, &why)) return 0;
    sec_skips += why >> 1 & 1, sec_skips_fs += why >> 2 & 1;
    if (why & 1) {
        uint64_t hit = df_match_origin(&list, vo) >= 0 ? vo : fo;
        unsigned i = 0;
        for (; i < ntold && told[i] != hit; i++) {}
        if (i == ntold && ntold < DF_MAX_ORIGINS) {
            told[ntold++] = hit;
            rt_log("draw-skip: origin %016llx skipped (DKB_SKIP_ORIGINS)", (unsigned long long)hit);
        }
    }
    return 1;
}

static int lq_skip_enabled(void) {
    static int on = -1;
    if (on < 0) on = lq_platform() && cfg_bool("TOTK_FIX_VS_FAULT");
    return on;
}

static int light_quad_fault_skip(const VkExec *ex, const GfxShader *vs, const GfxShader *fs) {
    if (!lq_skip_enabled()) return 0;
    int m = light_quad_match(ex, vs, fs);
    int folded = m && gfx_shader_folded(vs);
    if (folded) {   // the fold was applied to this vertex stage: the draw is made (gfx_shader.c lq_fold)
        const uint8_t *ctl, *code;
        uint32_t len;
        static volatile uint64_t told[8];
        uint64_t origin = find_stage(ex->prog, NVN_STAGE_VS, &ctl, &code, &len) ? nvn2_origin(code) : 0;
        int seen = 0;
        for (int i = 0; i < 8 && !seen; i++) seen = __atomic_load_n(&told[i], __ATOMIC_RELAXED) == origin;
        for (int i = 0; i < 8 && !seen; i++) {
            uint64_t zero = 0;
            if (__atomic_compare_exchange_n(&told[i], &zero, origin, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                rt_log("light-quad: drawing %016llx with folded vertex stage", (unsigned long long)origin);
                break;
            }
        }
    }
    if (!lq_skip_decision(1, m, folded)) return 0;
    if (m == 1) warn_once(W_FAULT_SKIP2, "a light quad of a known faulting NVN 2 vertex stage is skipped on NVIDIA (#9)");
    else if (m == 2) warn_once(W_FAULT_SKIP, "light-quad draws with ten instance streams are skipped on NVIDIA (#9; TOTK_FIX_VS_FAULT=0 draws them)");
    return 1;
}

// DKB_LQ_INTERLEAVE=1 (off by default; only with TOTK_FIX_VS_FAULT=0, else the skip above takes these draws): the matched
// draw's per-instance streams (one RGBA32F attribute per binding, nine on the faulting quads) are copied into one
// interleaved buffer and bound as a single binding of stride 16*N with the attributes at offsets 16*k, in case the driver
// faults on the many separate instance-rate bindings. The source bytes are read from host-visible pool memory (the way
// the byte-index path reads indices); a stream in device memory leaves the draw alone. The copy covers instances
// [0, base_instance+instances) with only [base_instance, base_instance+instances) written, and firstInstance is kept, so
// gl_InstanceIndex and c0's base instance (c0_binst) are unchanged and Vulkan's fetch of instance base_instance+i lands
// on element base_instance+i of the copy. The copy goes to the command buffer's scratch chunks (scratch()), written by
// the host before the submit, so no barrier is needed. Direct and indexed draws only; indirect draws are left alone.
static void lq_note(const char *why) {
    static volatile int32_t seen;
    if (!__atomic_exchange_n(&seen, 1, __ATOMIC_RELAXED)) rt_log("lq-interleave: draw left alone: %s (logged once)", why);
}

static int lq_interleave_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("DKB_LQ_INTERLEAVE");
        // "2" also skips the matched draws the rewrite has to leave alone (diagnostic: is the remainder what faults?)
        on = e && (!strcmp(e, "1") || !strcmp(e, "2")) && lq_platform() && !cfg_bool("TOTK_FIX_VS_FAULT") ? e[0] - '0' : 0;
    }
    return on;
}

static int lq_interleave_plan(const VkExec *ex, const GfxShader *vs, const GfxShader *fs, const p_draw *d, const p_draw_idx *di,
                              LqPlan *p) {
    if (!lq_interleave_enabled() || !light_quad_match(ex, vs, fs)) return 0;
    if (!d && !di) return lq_note("indirect draw"), 0;
    int32_t base = d ? d->base_instance : di->base_instance, inst = d ? d->instances : di->instances;
    if (base < 0) return lq_note("negative base instance"), 0;
    GfxShaderIo io[32];
    uint32_t used = gfx_shader_inputs(vs, io);
    VkLqAttr a[VK_LQ_MAX_STREAMS];
    unsigned n = 0;
    for (unsigned l = 0; l < 32; l++) {
        if (!(used >> l & 1)) continue;
        if (n == VK_LQ_MAX_STREAMS) return lq_note("more than 16 attributes"), 0;
        va_st va = {0};
        if (l < 16) memcpy(&va, &ex->pipe.vattr[l], sizeof va);
        VkFormat f = va.fmt ? vk_vertex_format((int)va.fmt, NULL) : VK_FORMAT_UNDEFINED;
        if (f == VK_FORMAT_UNDEFINED || va.stream >= 16) return lq_note("an attribute reads the zero binding"), 0;
        a[n] = (VkLqAttr){l, va.stream, va.offset, ex->pipe.vstream[va.stream].stride, ex->pipe.vstream[va.stream].divisor, f};
        p->loc[n] = l, p->fmt[n] = va.fmt;
        n++;
    }
    const char *why;
    if (!vk_lq_interleave_check(a, n, &why)) return lq_note(why), 0;
    p->n = n, p->first = (uint32_t)base, p->count = inst > 0 ? (uint32_t)inst : 1;
    uint64_t bytes;
    if (!vk_lq_interleave_bytes(p->first, p->count, n, &bytes)) return lq_note("base instance plus instances out of range"), 0;
    for (unsigned k = 0; k < n; k++) {
        const NVNbufferRange *r = &ex->vbuf[a[k].stream];
        VkMemSpan sp;
        if (!r->address || !vk_mem_resolve(r->address, 1, &sp)) return lq_note("a stream is not backed"), 0;
        if (sp.alloc || (sp.cls != VK_MCLS_HOST_CACHED && sp.cls != VK_MCLS_HOST_UNC))
            return lq_note("a stream is in device memory (no host copy)"), 0;
        uint64_t have = r->size && r->size < sp.avail ? r->size : sp.avail;
        if (have < ((uint64_t)p->first + p->count) * 16u) return lq_note("a stream is shorter than the instances drawn"), 0;
        p->src[k] = (const uint8_t *)(uintptr_t)r->address;
    }
    return 1;
}

// The pipeline state the plan draws with: every attribute on binding 0 at offset 16*k, stride 16*N, divisor 1. The other
// streams are cleared, so the pipeline key is distinct and (static vertex input) names one binding.
static void lq_interleave_state(VkPipeState *s, const LqPlan *p) {
    memset(s->vattr, 0, sizeof s->vattr);
    memset(s->vstream, 0, sizeof s->vstream);
    for (unsigned k = 0; k < p->n; k++) {
        va_st va = {.fmt = p->fmt[k], .offset = 16u * k, .stream = 0};   // the NVN format word is the original's
        memcpy(&s->vattr[p->loc[k]], &va, sizeof va);
    }
    s->vstream[0] = (vs_st){.stride = 16u * p->n, .divisor = 1};
}

// Inspect matched draws before the workaround returns, without building their pipeline or issuing a draw.
// Each distinct shape (vertex-stage origin, used inputs, attribute and stream state, draw kind) is logged once, as
// computed from recorded state only: alignment and layout per attribute, extent per stream against the bound range
// and its buffer, divisors, the zero-binding fallback and the slots bind_vertex_buffers feeds from the zero buffer.
// Indexed vertex-rate fetches and all indirect ranges need GPU data and are reported as unknown.
static void light_quad_fetch_log(const VkExec *ex, const GfxShader *vs, const p_draw *d, const p_draw_idx *di, int restart) {
    static int enabled = -1;
    static unsigned shapes;
    static uint64_t seen[256];
    if (enabled < 0) enabled = getenv("DKB_LIGHT_QUAD_FETCH") != NULL;
    if (!enabled) return;
    const uint8_t *ctl, *code = NULL;
    uint32_t len;
    uint64_t origin = find_stage(ex->prog, NVN_STAGE_VS, &ctl, &code, &len) ? nvn2_origin(code) : 0;
    GfxShaderIo io[32];
    uint32_t used = gfx_shader_inputs(vs, io);
    uint64_t hash = 1469598103934665603ull;
    const int kind = d ? 0 : di ? 1 : 2;
    const void *parts[] = {&origin, &used, ex->pipe.vattr, ex->pipe.vstream, &kind};
    const size_t sizes[] = {sizeof origin, sizeof used, sizeof ex->pipe.vattr, sizeof ex->pipe.vstream, sizeof kind};
    for (unsigned i = 0; i < sizeof parts / sizeof *parts; i++)
        for (size_t j = 0; j < sizes[i]; j++) hash = (hash ^ ((const uint8_t *)parts[i])[j]) * 1099511628211ull;
    for (unsigned i = 0; i < shapes; i++) if (seen[i] == hash) return;
    if (shapes >= 256) return;
    seen[shapes++] = hash;
    unsigned draw = shapes;
    int32_t count = d ? d->count : di ? di->count : 0;
    int32_t base_instance = d ? d->base_instance : di ? di->base_instance : -1;
    int32_t instances = d ? d->instances : di ? di->instances : 0;
    uint64_t inst = instances > 0 ? (uint64_t)instances : 1;
    uint64_t snapshot_max = 0;
    int snapshot = 0;
    if (di && di->count > 0 && di->count <= 4096 && di->itype >= 0 && di->itype <= 2) {
        unsigned width = 1u << di->itype;
        VkMemSpan index_span;
        if (vk_mem_resolve(di->ibuf, (uint64_t)di->count * width, &index_span) && !index_span.alloc &&
            (index_span.cls == VK_MCLS_HOST_CACHED || index_span.cls == VK_MCLS_HOST_UNC))
            snapshot = vk_vertex_fetch_indices((const void *)(uintptr_t)di->ibuf, (uint32_t)di->count, width,
                                               restart, di->base_vertex, &snapshot_max);
    }
    // The attribute list vertex_input() (vk_pipeline.c) builds, and the streams it references.
    uint32_t streams = 0;
    unsigned zero_attrs = 0, nattr = 0;
    for (unsigned l = 0; l < 32; l++) {
        if (!(used >> l & 1)) continue;
        nattr++;
        va_st a = {0};
        if (l < 16) memcpy(&a, &ex->pipe.vattr[l], sizeof a);
        VkFormat fmt = a.fmt ? vk_vertex_format((int)a.fmt, NULL) : VK_FORMAT_UNDEFINED;
        if (fmt == VK_FORMAT_UNDEFINED || a.stream >= 16) zero_attrs++;
        else streams |= 1u << a.stream;
    }
    unsigned nstreams = (unsigned)__builtin_popcount(streams);
    int multi_div = 0;
    for (unsigned b = 0; b < 16; b++) if ((streams >> b & 1) && ex->pipe.vstream[b].divisor > 1) multi_div = 1;
    rt_log("light-fetch: shape=%u vs=%016llx kind=%s first=%d count=%d base_instance=%d instances=%d attrs=%u streams=%u "
           "zero_binding_attrs=%u divisor_gt1=%d dyn_vi=%d used=%08x index_snapshot=%d snapshot_max=%llu",
           draw, (unsigned long long)origin, d ? "direct" : di ? "indexed" : "indirect", d ? d->first : -1, count,
           base_instance, (d || di) ? (int)inst : -1, nattr, nstreams, zero_attrs, multi_div,
           vk_pipeline_vertex_input_dynamic(), used, snapshot, (unsigned long long)snapshot_max);
    // Per stream: extent against the bound range and its buffer.
    uint64_t bound_end[16] = {0}, bound_off[16] = {0}, bound_size[16] = {0};
    int fed_zero[16] = {0};
    for (unsigned b = 0; b < 16; b++) {
        const NVNbufferRange *r = &ex->vbuf[b];
        VkMemSpan sp = {0};
        int resolved = r->address && vk_mem_resolve(r->address, 1, &sp);
        uint64_t sz = resolved ? (r->size && r->size < sp.avail ? r->size : sp.avail) : ZERO_BYTES;
        bound_off[b] = resolved ? sp.offset : 0, bound_size[b] = sz, fed_zero[b] = !resolved;
        bound_end[b] = resolved ? sp.offset + sp.avail : ZERO_BYTES;
        if (!(streams >> b & 1)) continue;
        unsigned stride = ex->pipe.vstream[b].stride, divisor = ex->pipe.vstream[b].divisor;
        uint64_t alloc_size = 0;
        if (resolved && sp.alloc) {
            VmaAllocationInfo ai;
            vmaGetAllocationInfo(vk_ctx()->vma, sp.alloc, &ai);
            alloc_size = ai.size;
        }
        rt_log("light-fetch: shape=%u stream=%u rate=%s divisor=%u stride=%u stride_mod4=%u stride_zero_instance=%d "
               "raw_addr=%#llx raw_size=%llu resolved=%d fed_by_zero_buffer=%d avail=%llu span_offset=%llu "
               "bound_size=%llu buffer_end=%llu alloc_size=%llu",
               draw, b, divisor ? "instance" : "vertex", divisor, stride, stride & 3, divisor && !stride,
               (unsigned long long)r->address, (unsigned long long)r->size, resolved, !resolved,
               (unsigned long long)(resolved ? sp.avail : 0), (unsigned long long)bound_off[b],
               (unsigned long long)sz, (unsigned long long)bound_end[b], (unsigned long long)alloc_size);
    }
    // Slots that bind_vertex_buffers feeds from the zero buffer with the pipeline's real stride.
    char zline[16 * 24];
    size_t zn = 0;
    zline[0] = 0;
    for (unsigned b = 0; b < 16; b++) {
        unsigned stride = ex->pipe.vstream[b].stride;
        if (!fed_zero[b] || (!stride && !(streams >> b & 1))) continue;
        zn += (size_t)snprintf(zline + zn, sizeof zline - zn, " %u:stride=%u,read=%d", b, stride, (streams >> b & 1) != 0);
    }
    rt_log("light-fetch: shape=%u zero_buffer_slots (stride!=0 or read):%s", draw, zn ? zline : " none");
    for (unsigned l = 0; l < 32; l++) {
        if (!(used >> l & 1)) continue;
        va_st a = {0};
        if (l < 16) memcpy(&a, &ex->pipe.vattr[l], sizeof a);
        VkFormat fmt = a.fmt ? vk_vertex_format((int)a.fmt, NULL) : VK_FORMAT_UNDEFINED;
        if (fmt == VK_FORMAT_UNDEFINED || a.stream >= 16) {
            rt_log("light-fetch: shape=%u location=%u zero-binding(%u) nvn_format=%u", draw, l, VK_PIPE_ZERO_BINDING, a.fmt);
            continue;
        }
        unsigned b = a.stream, stride = ex->pipe.vstream[b].stride, divisor = ex->pipe.vstream[b].divisor;
        unsigned bytes = vk_vertex_fetch_format_size(fmt), comp = vk_vertex_fetch_component_size(fmt);
        uint64_t addr_off = bound_off[b] + a.offset;
        uint64_t end = 0, idx = 0, end_plus = 0;
        int known = (d || di) && vk_vertex_fetch_index(divisor != 0, d != NULL, d ? d->first : 0,
                                                          count, base_instance, instances, &idx);
        if (known) known = vk_vertex_fetch_end(idx, stride, a.offset, bytes, &end);
        // The (base_instance + instances) * stride + attribute end bound, one instance beyond the last.
        int plus_known = divisor && (d || di) && base_instance >= 0 &&
                         vk_vertex_fetch_end((uint64_t)(uint32_t)base_instance + inst, stride, a.offset, bytes, &end_plus);
        uint64_t snapshot_end = 0;
        int snapshot_known = di && !divisor && snapshot == 1 &&
                             vk_vertex_fetch_end(snapshot_max, stride, a.offset, bytes, &snapshot_end);
        rt_log("light-fetch: shape=%u location=%u stream=%u nvn_format=%u vk_format=%u attr_offset=%u bytes=%u comp=%u "
               "addr_mod4=%llu addr_mod_comp=%llu stride_lt_attr_end=%d known=%d end_exclusive=%llu out_of_bounds=%d "
               "end_plus1=%llu plus1_overrun=%d buffer_overrun=%d snapshot_oob=%d",
               draw, l, b, a.fmt, (unsigned)fmt, a.offset, bytes, comp, (unsigned long long)(addr_off & 3),
               (unsigned long long)(comp ? addr_off % comp : 0), stride < (uint64_t)a.offset + bytes, known,
               (unsigned long long)end, known && end > bound_size[b], (unsigned long long)end_plus,
               plus_known && end_plus > bound_size[b], known && !fed_zero[b] && bound_off[b] + end > bound_end[b],
               snapshot_known ? snapshot_end > bound_size[b] : -1);
    }
}

// DKB_LQ_TRACE=1 (off by default, DIAGNOSTICS.md): log-only. For every draw light_quad_fault_skip skips, and every draw
// that is not skipped but runs a fragment program of the same family (it reads the storage buffer at c0+0x530, or
// its NVN 2 origin was seen in a skipped draw), the first appearance of each distinct key is logged:
//   lq-trace: t=<world clock s> key=<hash of the text> <text>
// The key is the text itself: skipped flag, NVN 2 vertex and fragment origins, topology, vertex and instance counts,
// the render targets (NVN format, size, samples, layers), each fragment storage buffer (c0 word, slot, bound address
// and size), c0+0x530 (the u32 vk_c0_fill puts there: the low word of the fragment storage slot 2 address), and each
// texture and image the fragment stage reads (handle, storage address, format, size, levels, layers, view type, the
// view's level and layer range, the sampler's address modes and compare). Once per world-clock second:
//   lq-trace: sec=<n> skipped=<draws skipped in that second> keys=<distinct keys so far>
// The clock is frame_step_world_us(), the one a scenario's --seconds ends on. It is read at draws, so a second in which
// nothing drew still gets its line when the next draw arrives. At most LQT_LINES lines. Texture lookups here are the
// ones bind_resources makes, but for skipped draws they are made earlier than the game would make them.
#define LQT_LINES 2000
#define LQT_KEYS 8192
static void lqt_emit(const char *fmt, ...) {
    static unsigned lines;
    if (lines >= LQT_LINES) return;
    char b[3072];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    rt_log("%s", b);
    if (++lines == LQT_LINES) rt_log("lq-trace: %d lines written: log capped", LQT_LINES);
}

// Bytes [addr, addr+n) of guest memory as the replay's draws see them: the submission's uniform snapshot (what the GPU reads),
// else the recorded buffer updates, else the mapped memory (the way ubo_ptr reads a pointer out of a user block). 0:
// unbound, unmapped or only partly mapped.
static int lqc_read(const VkExec *ex, uint64_t addr, uint64_t n, void *out) {
    VkMemSpan sp;
    if (!addr || !n) return 0;
    if (vk_usnap_read(addr, n, out) || upd_read(ex, addr, out, n)) return 1;
    if (!vk_mem_resolve(addr, n, &sp)) return 0;
    memcpy(out, (const void *)(uintptr_t)addr, n);
    return 1;
}

// Two u32 words of a bound constant buffer (cbuf slot 3..20 of a stage) at a byte offset. 0: unbound, too short or unmapped.
static int lqt_cbuf_words(const VkExec *ex, int st, int cbuf, uint64_t off, uint32_t w[2]) {
    if (cbuf < 3 || cbuf - 3 >= 18) return 0;
    const p_bind_buf *u = &ex->ubo[st][cbuf - 3];
    if (!u->addr || u->size < off + 8) return 0;
    return lqc_read(ex, u->addr + off, 8, w);
}

static void lq_trace(const VkExec *ex, const GfxShader *vs, const GfxShader *fs, int prim, int32_t count,
                     const p_draw *d, const p_draw_idx *di, int skipped) {
    static int enabled = -1;
    static uint64_t keys[LQT_KEYS], fs_skipped[64];
    static unsigned nkeys, nfs_skipped;
    static int64_t sec = -1;
    static unsigned skips;
    if (enabled < 0) {
        const char *e = getenv("DKB_LQ_TRACE");
        enabled = e && strcmp(e, "0");
    }
    if (!enabled || !vs || !fs) return;
    int64_t w = frame_step_world_us(), now = w < 0 ? -1 : w / 1000000;
    if (now != sec) {
        if (sec >= 0)
            for (int64_t s = sec, n = 0; s < now && n < 60; s++, n++)
                lqt_emit("lq-trace: sec=%lld skipped=%u keys=%u", (long long)s, s == sec ? skips : 0, nkeys);
        sec = now, skips = 0;
    }
    const uint8_t *ctl, *code = NULL;
    uint32_t len;
    uint64_t vo = find_stage(ex->prog, NVN_STAGE_VS, &ctl, &code, &len) ? nvn2_origin(code) : 0, fo = 0;
    code = NULL;
    if (find_stage(ex->prog, NVN_STAGE_FS, &ctl, &code, &len)) fo = nvn2_origin(code);
    const GfxShaderRes *sr;
    int nss = gfx_shader_resources(fs, GFX_RES_SSBO, &sr), lights = 0, known = 0;
    for (int i = 0; i < nss; i++) lights |= sr[i].cbuf_slot == 0 && sr[i].handle * 4 == 0x530;
    for (unsigned i = 0; i < nfs_skipped; i++) known |= fs_skipped[i] == fo;
    if (skipped) {
        skips++;
        if (!known && nfs_skipped < 64) fs_skipped[nfs_skipped++] = fo;
    } else if (!lights && !known) {
        return;
    }
    char t[2800];
    size_t n = 0;
#define A(...) (n += n < sizeof t ? (size_t)snprintf(t + n, sizeof t - n, __VA_ARGS__) : 0)
    int32_t inst = d ? d->instances : di ? di->instances : 0;
    A("skip=%d vs=%016llx fs=%016llx prim=%d kind=%s count=%d inst=%d", skipped, (unsigned long long)vo,
      (unsigned long long)fo, prim, d ? "direct" : di ? "indexed" : "indirect", count, inst > 0 ? inst : d || di ? 1 : 0);
    A(" rt=[");
    for (int i = 0; i < 8; i++) {
        const VkExecTarget *c = &ex->color[i];
        if (c->image) A("%d:f%d,%ux%u,s%u,l%u;", i, c->nvn_format, c->width, c->height, c->samples, c->layers);
    }
    A("] ds=");
    if (ex->depth.image) A("f%d,%ux%u,s%u,l%u", ex->depth.nvn_format, ex->depth.width, ex->depth.height, ex->depth.samples, ex->depth.layers);
    else A("none");
    uint8_t bank[VK_C0_BYTES];
    vk_c0_fill(bank, NVN_STAGE_FS, ex->ssbo[NVN_STAGE_FS], ex->ubo[NVN_STAGE_FS], c0_dim, c0_binst);
    uint32_t c530;
    memcpy(&c530, bank + 0x530, 4);
    A(" c0_530=%#x", c530);
    for (int i = 0; i < nss; i++) {
        int ubo = 0, slot = ssbo_slot(&sr[i], NVN_STAGE_FS, &ubo);
        const p_bind_buf *b = slot >= 0 && slot < 16 && !ubo ? &ex->ssbo[NVN_STAGE_FS][slot] : NULL;
        A(" ssbo{c%d+%#x slot=%d addr=%#llx size=%llu}", sr[i].cbuf_slot, sr[i].handle * 4, slot,
          (unsigned long long)(b ? b->addr : 0), (unsigned long long)(b ? b->size : 0));
    }
    for (int kind = GFX_RES_TEX; kind <= GFX_RES_IMG; kind++) {
        const GfxShaderRes *r;
        int nr = gfx_shader_resources(fs, kind, &r), img = kind == GFX_RES_IMG;
        for (int i = 0; i < nr; i++) {
            for (int k = 0; k < (r[i].count ? r[i].count : 1); k++) {
                int slot = r[i].slot + k;
                uint64_t h = 0;
                if (img) h = slot >= 0 && slot < 32 ? ex->img[NVN_STAGE_FS][slot] : 0;
                else h = slot >= 0 && slot < 128 ? ex->tex[NVN_STAGE_FS][slot] : 0;
                A(" %s{slot=%d rtype=%#x h=%#llx", img ? "img" : "tex", slot, r[i].type, (unsigned long long)h);
                if (h) {
                    view_t v = {0};
                    int hv = 0;
                    VkTexImage im = {0};
                    int ok = 0;
                    vk_texture_lock();
                    const VkTexImage *p = vk_texture_by_id_locked(&ex->tex_pool, (int)(h & 0xfffff), &v, &hv);
                    if (p) im = *p, ok = 1;
                    vk_texture_unlock();
                    if (ok) {
                        A(" addr=%#llx fmt=%d %ux%ux%u levels=%u layers=%u samples=%u vt=%d", (unsigned long long)im.addr,
                          (int)im.format, im.width, im.height, im.depth, im.levels, im.layers, im.samples, (int)im.view_type);
                        if (hv) A(" view=lv%u+%u,ly%u+%u,fmt%u", v.set_levels ? v.base_level : 0, v.set_levels ? v.nlevels : 0,
                                  v.set_layers ? v.min_layer : 0, v.set_layers ? v.nlayers : 0, v.set_format ? v.format : 0);
                    } else {
                        A(" unresolved");
                    }
                    samp_par sp;
                    if (!img && vk_sampler_params(&ex->samp_pool, (int)(h >> 20 & 0xfff), &sp))
                        A(" samp=%u:wrap%u,%u,%u,cmp%u,%u,filt%u,%u", (unsigned)(h >> 20 & 0xfff), sp.ws, sp.wt, sp.wr,
                          sp.cmp_mode, sp.cmp_func, sp.minf, sp.magf);
                }
                A("}");
            }
        }
    }
#undef A
    uint64_t hash = 1469598103934665603ull;
    for (size_t j = 0; j < n && t[j]; j++) hash = (hash ^ (uint8_t)t[j]) * 1099511628211ull;
    unsigned at = (unsigned)(hash % LQT_KEYS);
    for (unsigned j = 0; j < LQT_KEYS; j++, at = (at + 1) % LQT_KEYS) {
        if (keys[at] == hash) return;
        if (!keys[at]) break;
    }
    if (nkeys >= LQT_KEYS / 2) return;
    keys[at] = hash ? hash : 1, nkeys++;
    // The vertex stage's constant buffer c9 at row 7 (byte 112; the family's flag words, read as u32 by OpBitwiseAnd). Not part
    // of the key: it changes without the draw's other state changing, and is logged when the key first appears.
    uint32_t c9[2];
    char c9t[96];
    if (lqt_cbuf_words(ex, NVN_STAGE_VS, 9, 7 * 16, c9)) snprintf(c9t, sizeof c9t, " vs_c9_row7={x=%#x y=%#x}", c9[0], c9[1]);
    else snprintf(c9t, sizeof c9t, " vs_c9_row7=unreadable");
    // The c9 field goes right after skip/vs/fs: the logger cuts long lines and the texture list can be long.
    const char *split = strstr(t, " prim=");
    if (!split) split = t + strlen(t);
    lqt_emit("lq-trace: t=%.3f key=%016llx %.*s%s%s", w < 0 ? -1.0 : (double)w / 1e6, (unsigned long long)hash,
             (int)(split - t), t, c9t, split);
}

// DKB_LQ_CAPTURE_VS=<comma list of hex NVN 2 vertex-stage origins> (off by default, log and files only): writes, for the first draws
// of each listed vertex stage whose bound data differs from the earlier ones, a self-contained directory the offline evaluator
// (tools/lq_vseval.py) runs on. Every draw of such a stage is captured whether the light-quad skip drops it or not.
// DKB_CAPTURE_ORIGINS=<hex prefixes> (1..16 digits each, up to 64; may be combined with DKB_LQ_CAPTURE_VS) is the same for any
// vertex stage, light quad or not: instanced, several streams, no Location 9 input. The limits and the layout are the same; the
// per-origin maximum counts each distinct origin a prefix matches. It also writes fragment.spv when the draw has a fragment stage.
// Nothing about the draw changes, and nothing is read back from the GPU. A draw that DKB_SKIP_ORIGINS / DKB_SKIP_BISECT skips is
// not captured (they run first); leave them unset when capturing.
//   DKB_LQ_CAPTURE_DIR   parent directory (default quad-replay-captures, relative to the working directory)
//   DKB_LQ_CAPTURE_TAG   name prefix of the draw directories (default vs): <dir>/<tag>-<origin>-<n>/, never an existing one
//   DKB_LQ_CAPTURE_MAX   draws kept per origin (default 6); DKB_LQ_CAPTURE_MAX_TOTAL all origins (default 64)
// Files: vertex.spv (the module the runtime feeds Vulkan: after spirv_optimize and any DKB_LQ_PATCH), draw.txt (the draw, the
// vertex attributes and streams, the uniform and storage resources with their bindings and slots, the cache key), support.bin (the
// support buffer), ubo-<slot>.bin (each bound vertex-stage uniform block, its full bound range), stream-<n>.bin (each used
// vertex stream from the bound address, as far as the draw reaches), ssbo-<slot>.bin, index.bin. All read on the CPU the way the
// draw reads them (uniform snapshot, recorded updates, guest memory); no GPU read-back. Contents that cannot be read are listed
// as unreadable in draw.txt.
#define LQC_CAP (4u << 20)
static void lqc_file(const char *dir, const char *name, const void *data, size_t n) {
    char p[1100];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    plat_file_write_atomic(p, data, n);
}

static uint64_t lqc_hash(uint64_t h, const void *data, size_t n) {
    for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t *)data)[i]) * 1099511628211ull;
    return h;
}

static void lq_capture(const VkExec *ex, const GfxShader *vs, const GfxShader *fs, int prim, int32_t count, const p_draw *d,
                       const p_draw_idx *di, int skipped) {
    static int enabled = -1;
    static DfOrigins want;
    static uint64_t origins[64], seen_hash[512];
    static unsigned norigins, per_origin[64], nseen, total, max_each = 6, max_total = 64;
    static char out_dir[900], tag[64];
    if (enabled < 0) {
        const char *e1 = getenv("DKB_LQ_CAPTURE_VS"), *e2 = getenv("DKB_CAPTURE_ORIGINS");
        char list[1024];
        snprintf(list, sizeof list, "%s%s%s", e1 ? e1 : "", e1 && *e1 && e2 && *e2 ? "," : "", e2 ? e2 : "");
        enabled = list[0] != 0;
        if (enabled && df_parse_origins(&want, list) <= 0) {
            rt_log("lq-capture: the origin list is not a comma-separated list of 1..16 hex digit prefixes (at most %d); capture off", DF_MAX_ORIGINS);
            enabled = 0;
        }
        if (enabled) {
            norigins = 0;
            const char *dir = getenv("DKB_LQ_CAPTURE_DIR"), *t = getenv("DKB_LQ_CAPTURE_TAG"), *m = getenv("DKB_LQ_CAPTURE_MAX"),
                       *mt = getenv("DKB_LQ_CAPTURE_MAX_TOTAL");
            snprintf(out_dir, sizeof out_dir, "%s", dir && *dir ? dir : "quad-replay-captures");
            snprintf(tag, sizeof tag, "%s", t && *t ? t : "vs");
            if (m && atoi(m) > 0) max_each = (unsigned)atoi(m);
            if (mt && atoi(mt) > 0) max_total = (unsigned)atoi(mt);
            rt_log("lq-capture: %u origin prefix(es), %u draw(s) each (at most %u), into %s/%s-<origin>-<n>", want.n,
                   max_each, max_total, out_dir, tag);
        }
    }
    if (!enabled || !vs || total >= max_total) return;
    const uint8_t *ctl, *code = NULL;
    uint32_t len;
    uint64_t origin = find_stage(ex->prog, NVN_STAGE_VS, &ctl, &code, &len) ? nvn2_origin(code) : 0;
    if (df_match_origin(&want, origin) < 0) return;
    unsigned oi = norigins;
    for (unsigned i = 0; i < norigins; i++) if (origins[i] == origin) oi = i;
    if (oi == norigins) {
        if (norigins >= 64) return;
        origins[norigins++] = origin;
    }
    if (per_origin[oi] >= max_each) return;
    // What the draw reads, from the CPU side.
    GfxShaderIo io[32];
    const uint32_t used = gfx_shader_inputs(vs, io);
    const int32_t first = d ? d->first : 0, base_instance = d ? d->base_instance : di ? di->base_instance : 0;
    const int32_t instances = d ? (d->instances > 0 ? d->instances : 1) : di ? (di->instances > 0 ? di->instances : 1) : 1;
    uint8_t *stream[16] = {0};
    uint64_t stream_n[16] = {0};
    int stream_ok[16] = {0};
    uint32_t streams = 0;
    for (unsigned l = 0; l < 32 && l < 16 + 16; l++) {
        if (!(used >> l & 1) || l >= 16) continue;
        va_st a;
        memcpy(&a, &ex->pipe.vattr[l], sizeof a);
        if (a.fmt && a.stream < 16) streams |= 1u << a.stream;
    }
    uint64_t hash = 1469598103934665603ull;
    hash = lqc_hash(hash, &first, sizeof first), hash = lqc_hash(hash, &count, sizeof count);
    hash = lqc_hash(hash, &instances, sizeof instances), hash = lqc_hash(hash, &base_instance, sizeof base_instance);
    for (unsigned b = 0; b < 16; b++) {
        if (!(streams >> b & 1)) continue;
        const NVNbufferRange *r = &ex->vbuf[b];
        const unsigned stride = ex->pipe.vstream[b].stride, divisor = ex->pipe.vstream[b].divisor;
        const uint64_t units = divisor ? ((uint64_t)base_instance + (uint64_t)instances + divisor - 1) / divisor
                                       : (uint64_t)(first > 0 ? first : 0) + (uint64_t)(count > 0 ? count : 0);
        uint64_t n = (di && !divisor ? (uint64_t)count + 65536 : units) * stride + 64;   // indexed vertex-rate: a generous window
        VkMemSpan sp;
        if (r->address && vk_mem_resolve(r->address, 1, &sp)) {
            if (r->size && n > r->size) n = r->size;
            if (n > sp.avail) n = sp.avail;
        }
        if (n > LQC_CAP) n = LQC_CAP;
        stream[b] = malloc(n ? n : 1);
        stream_n[b] = n;
        stream_ok[b] = stream[b] && lqc_read(ex, r->address, n, stream[b]);
        if (stream_ok[b]) hash = lqc_hash(hash, stream[b], n);
    }
    const GfxShaderRes *ur, *sr;
    const int nu = gfx_shader_resources(vs, GFX_RES_UBO, &ur), ns = gfx_shader_resources(vs, GFX_RES_SSBO, &sr);
    uint8_t *ubo_data[32] = {0};
    uint64_t ubo_n[32] = {0};
    int ubo_ok[32] = {0};
    for (int i = 0; i < nu && i < 32; i++) {
        const int slot = ur[i].slot;
        if (slot < 3 || slot - 3 >= 18) continue;
        const p_bind_buf *b = &ex->ubo[NVN_STAGE_VS][slot - 3];
        uint64_t n = b->size < LQC_CAP ? b->size : LQC_CAP;
        VkMemSpan sp;
        if (b->addr && vk_mem_resolve(b->addr, 1, &sp) && n > sp.avail) n = sp.avail;
        ubo_n[i] = n;
        ubo_data[i] = malloc(n ? n : 1);
        ubo_ok[i] = ubo_data[i] && b->addr && lqc_read(ex, b->addr, n, ubo_data[i]);
        if (ubo_ok[i]) hash = lqc_hash(hash, ubo_data[i], n);
    }
    uint8_t *idx = NULL;
    uint64_t idx_n = 0;
    int idx_ok = 0;
    if (di && di->count > 0 && di->itype >= 0 && di->itype <= 2) {
        idx_n = (uint64_t)di->count << di->itype;
        if (idx_n > LQC_CAP) idx_n = LQC_CAP;
        idx = malloc(idx_n);
        idx_ok = idx && lqc_read(ex, di->ibuf, idx_n, idx);
        if (idx_ok) hash = lqc_hash(hash, idx, idx_n);
    }
    int fresh = 1;
    for (unsigned i = 0; i < nseen; i++) fresh &= seen_hash[i] != hash;
    if (fresh && nseen < 512) {
        seen_hash[nseen++] = hash;
        char dir[1000];
        int made = 0;
        plat_file_mkdir_p(out_dir);
        for (unsigned n = per_origin[oi]; n < per_origin[oi] + 100 && !made; n++) {
            snprintf(dir, sizeof dir, "%s/%s-%016llx-%02u", out_dir, tag, (unsigned long long)origin, n);
            made = plat_fs_mkdir(dir) != PLAT_FS_EXISTS;
            if (!made) continue;
            plat_file_mkdir_p(dir);
        }
        if (made) {
            per_origin[oi]++, total++;
            uint64_t words;
            const uint32_t *spv = gfx_shader_spirv(vs, &words);
            lqc_file(dir, "vertex.spv", spv, (size_t)words * 4);
            if (fs) {
                uint64_t fw;
                const uint32_t *fspv = gfx_shader_spirv(fs, &fw);
                lqc_file(dir, "fragment.spv", fspv, (size_t)fw * 4);
            }
            uint8_t sup[SUPPORT_BYTES];
            float in[3];
            support_in(ex, in);
            support_fill(sup, in[0], in[1], in[2]);
            lqc_file(dir, "support.bin", sup, sizeof sup);
            char *t = malloc(65536);
            size_t n = 0;
#define W(...) (n += n < 65000 ? (size_t)snprintf(t + n, 65536 - n, __VA_ARGS__) : 0)
            char key[65] = {0};
            for (int i = 0; i < 32; i++) sprintf(key + 2 * i, "%02x", gfx_shader_key(vs)[i]);
            W("format=lq-vs-capture-v1\norigin=%016llx\ncache_key=%s\nskipped=%d\nprim=%d\nkind=%s\ncount=%d\nfirst=%d\ninstances=%d\n"
              "base_instance=%d\nbase_vertex=%d\nitype=%d\nsupport_bytes=%u\n",
              (unsigned long long)origin, key, skipped, prim, d ? "direct" : di ? "indexed" : "indirect", count, first, instances,
              base_instance, di ? di->base_vertex : 0, di ? di->itype : -1, (unsigned)SUPPORT_BYTES);
            for (unsigned l = 0; l < 16; l++) {
                if (!(used >> l & 1)) continue;
                va_st a;
                memcpy(&a, &ex->pipe.vattr[l], sizeof a);
                W("attr location=%u nvn_format=%u offset=%u stream=%u type=%u comps=%u\n", l, (unsigned)a.fmt, (unsigned)a.offset,
                  (unsigned)a.stream, (unsigned)io[l].type, (unsigned)io[l].comps);
            }
            for (unsigned b = 0; b < 16; b++) {
                if (!(streams >> b & 1)) continue;
                W("stream index=%u stride=%u divisor=%u address=%#llx bound_size=%llu captured=%llu readable=%d file=stream-%u.bin\n", b,
                  ex->pipe.vstream[b].stride, ex->pipe.vstream[b].divisor, (unsigned long long)ex->vbuf[b].address,
                  (unsigned long long)ex->vbuf[b].size, (unsigned long long)stream_n[b], stream_ok[b], b);
                char name[32];
                snprintf(name, sizeof name, "stream-%u.bin", b);
                if (stream_ok[b]) lqc_file(dir, name, stream[b], (size_t)stream_n[b]);
            }
            for (int i = 0; i < nu && i < 32; i++) {
                const int slot = ur[i].slot;
                const p_bind_buf *b = slot >= 3 && slot - 3 < 18 ? &ex->ubo[NVN_STAGE_VS][slot - 3] : NULL;
                W("ubo binding=%u cbuf_slot=%d address=%#llx bound_size=%llu captured=%llu readable=%d file=ubo-%d.bin\n", ur[i].binding,
                  slot, (unsigned long long)(b ? b->addr : 0), (unsigned long long)(b ? b->size : 0), (unsigned long long)ubo_n[i],
                  ubo_ok[i], slot);
                char name[32];
                snprintf(name, sizeof name, "ubo-%d.bin", slot);
                if (ubo_ok[i]) lqc_file(dir, name, ubo_data[i], (size_t)ubo_n[i]);
            }
            for (int i = 0; i < ns; i++) {
                int ubo = 0, slot = ssbo_slot(&sr[i], NVN_STAGE_VS, &ubo);
                const p_bind_buf *b = slot >= 0 && slot < 16 && !ubo ? &ex->ssbo[NVN_STAGE_VS][slot] : NULL;
                uint64_t sn = b ? (b->size < LQC_CAP ? b->size : LQC_CAP) : 0;
                uint8_t *sd = sn ? malloc(sn) : NULL;
                int ok = sd && lqc_read(ex, b->addr, sn, sd);
                W("ssbo binding=%u storage_slot=%d address=%#llx bound_size=%llu captured=%llu readable=%d file=ssbo-%d.bin\n", sr[i].binding,
                  slot, (unsigned long long)(b ? b->addr : 0), (unsigned long long)(b ? b->size : 0), (unsigned long long)sn, ok, slot);
                char name[32];
                snprintf(name, sizeof name, "ssbo-%d.bin", slot);
                if (ok) lqc_file(dir, name, sd, (size_t)sn);
                free(sd);
            }
            if (di) {
                W("index address=%#llx bytes=%llu readable=%d file=index.bin\n", (unsigned long long)di->ibuf, (unsigned long long)idx_n, idx_ok);
                if (idx_ok) lqc_file(dir, "index.bin", idx, (size_t)idx_n);
            }
#undef W
            lqc_file(dir, "draw.txt", t, n);
            free(t);
            rt_log("lq-capture: %s (draw %d x %d instance(s), skipped %d)", dir, count, instances, skipped);
        }
    }
    for (unsigned b = 0; b < 16; b++) free(stream[b]);
    for (int i = 0; i < 32; i++) free(ubo_data[i]);
    free(idx);
}

static _Thread_local struct { int on, valid, has_fs, has_gs; MemoKey k; const GfxShader *vs, *fs, *gs; } SM = {-1};

// Numeric handles are relative to the active descriptor tables. Keep the per-handle memos only while those tables
// and their address mappings stay unchanged. A sampler table switch also changes combined image descriptors.
static void descriptor_state(VkExec *ex) {
    static _Thread_local struct { const VkExec *ex; dpool_ref tex, samp; uint64_t mem_gen; uint32_t slots_gen; } last;
    uint64_t mg = vk_mem_gen();
    uint32_t sg = vk_texture_slots_gen();
    if (last.ex != ex || last.tex.addr != ex->tex_pool.addr || last.tex.count != ex->tex_pool.count ||
        last.samp.addr != ex->samp_pool.addr || last.samp.count != ex->samp_pool.count || last.mem_gen != mg) {
        if (!++descriptor_serial) rt_fatal("vk_exec: descriptor context serial exhausted");
        SM.valid = 0;
        ex->fb_fs = NULL;
        last.ex = ex, last.tex = ex->tex_pool, last.samp = ex->samp_pool, last.mem_gen = mg;
    }
    if (last.slots_gen != sg) ex->fb_fs = NULL;
    last.slots_gen = sg;
}

static void memo_key(const VkExec *ex, uint32_t gen, MemoKey *k) {
    k->prog = ex->prog, k->serial = ex->prog->serial, k->gen = gen, k->tex_serial = ex->tex_serial;
    memcpy(k->tex[0], ex->tex[NVN_STAGE_VS], sizeof k->tex[0]), memcpy(k->tex[1], ex->tex[NVN_STAGE_FS], sizeof k->tex[1]);
    memcpy(k->img[0], ex->img[NVN_STAGE_VS], sizeof k->img[0]), memcpy(k->img[1], ex->img[NVN_STAGE_FS], sizeof k->img[1]);
    k->geometry_input = geometry_input;
    if (geometry_input) {
        memcpy(k->tex[2], ex->tex[NVN_STAGE_GS], sizeof k->tex[2]);
        memcpy(k->img[2], ex->img[NVN_STAGE_GS], sizeof k->img[2]);
    }
    memcpy(&k->pipe, &ex->pipe, sizeof k->pipe);
}

static int memo_match(VkExec *ex, uint32_t gen, MemoKey *k) {
    descriptor_state(ex);
    if (!SM.valid) return 0;
    if (k->prog != ex->prog || k->serial != ex->prog->serial || k->gen != gen || k->geometry_input != geometry_input ||
        memcmp(&k->pipe, &ex->pipe, sizeof k->pipe))
        return 0;
    if (geometry_input && (memcmp(k->tex[2], ex->tex[NVN_STAGE_GS], sizeof k->tex[2]) ||
                           memcmp(k->img[2], ex->img[NVN_STAGE_GS], sizeof k->img[2]))) return 0;
    if (k->tex_serial == ex->tex_serial) return 1;
    if (memcmp(k->tex[0], ex->tex[NVN_STAGE_VS], sizeof k->tex[0]) || memcmp(k->tex[1], ex->tex[NVN_STAGE_FS], sizeof k->tex[1]) ||
        memcmp(k->img[0], ex->img[NVN_STAGE_VS], sizeof k->img[0]) || memcmp(k->img[1], ex->img[NVN_STAGE_FS], sizeof k->img[1]))
        return 0;
    k->tex_serial = ex->tex_serial;
    return 1;
}

static int shader_memo_on(void) {
    if (SM.on < 0) {
        const char *e = cfg_value("TOTK_SHADER_MEMO");
        SM.on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
    }
    return SM.on;
}

static _Thread_local struct { int on, valid; VkPipeState s; const VkPipeInfo *info; } PM = {-1};

static int pipe_memo_on(void) {
    if (PM.on < 0) PM.on = cfg_bool("TOTK_PIPE_MEMO");
    return PM.on;
}

static int experimental_geometry_on(void) {
    static int on = -1;
    if (on < 0) on = cfg_bool("TOTK_EXPERIMENTAL_GEOMETRY");
    return on;
}

// 1: an opted-in ordinary GS is required; 0: preserve the existing VS/FS path; -1: this opted-in draw is unsupported.
// Inspect raw NVN2 markers before any NVN1 control offsets. TOTK_EXPERIMENTAL_GEOMETRY admits no native shader; only
// DKB_NATIVE_GS does, for the passthrough subset checked first.
static int geometry_draw(VkExec *ex, uint32_t prim) {
    geometry_input = 0;
    if (native_gs_on() && ex->prog) {   // native failures keep VS/FS; ordinary stages retain their separate opt-in
        const uint8_t *ctl, *code, *fctl, *fcode;
        uint32_t len, flen, sph[5];
        if (!find_stage(ex->prog, NVN_STAGE_GS, &ctl, &code, &len)) return 0;
        if (nvn2_origin(code)) {
            if (len < 0x30 + 0x60 || !find_stage(ex->prog, NVN_STAGE_FS, &fctl, &fcode, &flen) ||
                vk_geometry_input(prim) != GFX_GEOMETRY_TRIANGLES) return 0;
            memcpy(sph, code + 0x30, sizeof sph);
            if (!(sph[0] & (1u << 24)) || sph[2] >> 24 != 1 || (sph[3] >> 24 & 15) != 7 || (sph[4] & 0xfff) != 1) {
                warn_once(W_GEOMETRY, "native GS: a stage outside the passthrough subset keeps the vertex and fragment path");
                return 0;
            }
            geometry_input = GFX_GEOMETRY_TRIANGLES;
            return 1;
        }
        // An ordinary GS still follows its own opt-in and validation below.
    }
    if (!experimental_geometry_on() || !ex->prog) return 0;
    int found = 0, portable = 0;
    uint32_t stages = 0;
    if (ex->prog->nstage > 6) goto unsupported;
    for (uint32_t i = 0; i < ex->prog->nstage; i++) {
        const uint8_t *ctl = ex->prog->sd[i].control;
        const uint8_t *code = (const uint8_t *)(uintptr_t)ex->prog->sd[i].data;
        if (!ctl || !code || gfx_shader_is_nvn2(ctl)) goto unsupported;
        uint32_t stage, len;
        memcpy(&stage, ctl + 0x714, 4);
        if (stage == NVN_STAGE_VS || stage == NVN_STAGE_FS) { stages |= 1u << stage; continue; }
        if (stage == NVN_STAGE_CS) continue;
        if (stage != NVN_STAGE_GS || found++) goto unsupported;   // no tessellation, duplicate or unknown stage
        stages |= 1u << stage;
        memcpy(&len, ctl + 0x704, 4);
        if (len < 0x30 + 0x60 || nvn2_origin(code) || c0_on()) goto unsupported;
        uint32_t sph;
        memcpy(&sph, code + 0x30, 4);
        if ((sph >> 10 & 15) != 4) goto unsupported;
        if (sph & (1u << 24)) {
            uint32_t skip[8];
            if (nvn_geometry_control_get(ctl, 0x880, code, len, skip) != 1 ||
                vk_geometry_input(prim) != GFX_GEOMETRY_TRIANGLES) goto unsupported;
            portable = 1;
        }
    }
    if (found) {
        // Replay currently replaces a complete program, not individual stages. Partial GS binds need a separate
        // merging contract; do not execute a GS which this bind did not request.
        if (!(stages & (1u << NVN_STAGE_VS)) || ((uint32_t)ex->prog_stages & stages) != stages) goto unsupported;
        // Portable metadata currently follows the verified whole-program SDK binding path.
        if (portable && ((uint32_t)ex->prog_stages != NVN_SHADER_STAGE_ALL_GRAPHICS_BITS ||
                         !(stages & (1u << NVN_STAGE_FS)))) goto unsupported;
        geometry_input = vk_geometry_input(prim);
        if (!geometry_input) goto unsupported;
    }
    return found;
unsupported:
    geometry_input = 0;
    warn_once(W_GEOMETRY, "experimental geometry draw rejected: ordinary GS, complete stage mask, supported primitive and FIX_C0=0 are required");
    return -1;
}

// Pass, shaders, pipeline, fixed state and resources for a draw. 0: the draw is skipped.
typedef struct { VkPipeState state; VkPipeline pipeline; const VkPipeInfo *info; } DumpDrawState;

static int prepare_draw(VkExec *ex, int prim, int32_t count, int restart, DumpDrawState *dump, const p_draw *direct, const p_draw_idx *indexed) {
    uint64_t ct_prep = CST(), ct_x;
    if (ex->render_disabled) return 0;
    static int skip_all = -1;   // DKB_SKIP_DRAWS=1: drop every draw (bisecting GPU faults)
    if (skip_all < 0) skip_all = getenv("DKB_SKIP_DRAWS") != NULL;
    if (skip_all) return 0;
    int want_gs = geometry_draw(ex, (uint32_t)prim);
    if (want_gs < 0) return 0;
    cb_res(ex);
    dummies_init();
    if (ex->pass_noatt) {   // no attachment: the pass covers this draw's viewport or scissor, reopened when it needs more
        uint32_t w, h;
        vk_exec_noatt_dims(ex, &w, &h);
        if (ex->in_pass && (w > ex->pass_w || h > ex->pass_h)) vk_exec_pass_end(ex);
        if (!ex->in_pass) ex->rt_resolved = 0;
    }
    if (!ex->in_pass) ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
    vk_exec_pass_begin(ex);
    VkRect2D sc = vk_exec_scissor(ex);
    if (!sc.extent.width || !sc.extent.height) return 0;
    if (!ex->prog) return warn_once(W_PROGRAM, "a draw without a bound program is skipped"), 0;
    int has_vs, has_fs, has_gs = 0, vs_wait = 0, fs_wait = 0, gs_wait = 0, memo = shader_memo_on();
    const GfxShader *vs, *fs, *gs = NULL;
    uint32_t gen = memo ? vk_texture_slots_gen() : 0;   // before shader_for, whose lookups may bump it
    if (memo && SM.valid && memo_match(ex, gen, &SM.k)) {
        vs = SM.vs, fs = SM.fs, gs = SM.gs, has_vs = 1, has_fs = SM.has_fs, has_gs = SM.has_gs;
        if (memo == 2) {
            int hv, hf, hg = 0, wv, wf, wg = 0;
            const GfxShader *v2 = shader_for(ex, NVN_STAGE_VS, &hv, &wv, 0), *f2 = shader_for(ex, NVN_STAGE_FS, &hf, &wf, 0);
            const GfxShader *g2 = want_gs ? shader_for(ex, NVN_STAGE_GS, &hg, &wg, 0) : NULL;
            if (v2 != vs || f2 != fs || g2 != gs || hf != has_fs || hg != has_gs || wv || wf || wg)
                rt_fatal("vk_exec: shader memo of program %llu differs from the lookup", (unsigned long long)ex->prog->serial);
        }
    } else {
        SM.valid = 0;
        vs = shader_for(ex, NVN_STAGE_VS, &has_vs, &vs_wait, 0);
        fs = shader_for(ex, NVN_STAGE_FS, &has_fs, &fs_wait, 0);
        native_gs_vs = vs, native_gs_fs = fs;
        if (want_gs) gs = shader_for(ex, NVN_STAGE_GS, &has_gs, &gs_wait, 0);
        if (memo && vs && !vs_wait && !fs_wait && !gs_wait && (!has_fs || fs) && (!want_gs || (has_gs && gs)))
            memo_key(ex, gen, &SM.k), SM.vs = vs, SM.fs = fs, SM.gs = gs, SM.has_fs = has_fs, SM.has_gs = has_gs, SM.valid = 1;
    }
    if (!has_vs) return warn_once(W_NO_VS, "a draw with a program that has no vertex stage is skipped"), 0;
    static int bis_lo = -1, bis_hi = 255;   // DKB_DRAW_BISECT=lo-hi (hex): only draws whose program byte is in range
    if (bis_lo < 0) {
        const char *e = getenv("DKB_DRAW_BISECT");
        bis_lo = 0;
        if (e) sscanf(e, "%x-%x", &bis_lo, &bis_hi);
    }
    if (vs && (bis_lo > 0 || bis_hi < 255)) {
        int b = gfx_shader_input_id(vs)[0] ^ (fs ? gfx_shader_input_id(fs)[0] : 0);
        if (b < bis_lo || b > bis_hi) return 0;
        if (bis_lo == bis_hi) {
            static int told;
            if (told++ < 4) {   // the cache file names (shader-cache/<key>.spv)
                char k[2][65] = {{0}};
                for (int i = 0; i < 32; i++) {
                    sprintf(k[0] + 2 * i, "%02x", gfx_shader_key(vs)[i]);
                    if (fs) sprintf(k[1] + 2 * i, "%02x", gfx_shader_key(fs)[i]);
                }
                const uint8_t *ctl, *vcode = NULL, *fcode = NULL;
                uint32_t len;
                find_stage(ex->prog, NVN_STAGE_VS, &ctl, &vcode, &len), find_stage(ex->prog, NVN_STAGE_FS, &ctl, &fcode, &len);
                rt_log("bisect: draw byte %#x vs %s (NVN 2 dump %016llx) fs %s (%016llx)", b, k[0],
                       (unsigned long long)nvn2_origin(vcode), k[1], (unsigned long long)nvn2_origin(fcode));
            }
        }
    }
    // A draw into a target the previous frame did not render (or into none) waits for a pending shader or pipeline:
    // one-shot draws (volumes and LUTs generated once, later only sampled) would leave their target empty for the
    // session. A draw into per-frame targets is skipped until they are ready and comes back next frame.
    int oneshot = -1, skip = pipe_skip_mode();   // -1: not looked up
    if (vs_wait || fs_wait || gs_wait) {
        if (!(oneshot = !vk_exec_targets_recent(ex, skip >= 1))) return perf_count(PERF_CNT_SHADER_SKIP), 0;   // skipped until ready
        perf_count(PERF_CNT_ONESHOT_WAIT);
        uint64_t sw0 = kern_now_ns();
        if (vs_wait) vs = shader_for(ex, NVN_STAGE_VS, &has_vs, NULL, 1);
        wait_log(ex, "one-shot draw", "shader translation", NVN_STAGE_VS, vs_wait ? kern_now_ns() - sw0 : 0);
        sw0 = kern_now_ns();
        if (fs_wait) fs = shader_for(ex, NVN_STAGE_FS, &has_fs, NULL, 1);
        wait_log(ex, "one-shot draw", "shader translation", NVN_STAGE_FS, fs_wait ? kern_now_ns() - sw0 : 0);
        native_gs_vs = vs, native_gs_fs = fs;
        sw0 = kern_now_ns();
        if (gs_wait) gs = shader_for(ex, NVN_STAGE_GS, &has_gs, NULL, 1);
        wait_log(ex, "one-shot draw", "shader translation", NVN_STAGE_GS, gs_wait ? kern_now_ns() - sw0 : 0);
    }
    if (draw_diag(ex, vs, fs, prim, direct, indexed)) return 0;
    int lq_skip = light_quad_fault_skip(ex, vs, fs);
    lq_trace(ex, vs, fs, prim, count, direct, indexed, lq_skip);
    lq_capture(ex, vs, fs, prim, count, direct, indexed, lq_skip);
    if (lq_skip) { light_quad_fetch_log(ex, vs, direct, indexed, restart); return 0; }
    if (lq_fold_missing(ex, vs)) return 0;
    if (!vs || (has_fs && !fs) || (want_gs && (!has_gs || !gs))) return 0;
    if (!geometry_feedback_safe(ex, vs, gs)) return 0;
    uint32_t fbl = feedback_prepare(ex, fs);
    VkPipeState s = ex->pipe;
    LqPlan lq;
    int lq_on = lq_interleave_plan(ex, vs, fs, direct, indexed, &lq);
    if (!lq_on && lq_interleave_enabled() == 2 && light_quad_match(ex, vs, fs)) {
        static volatile uint32_t left;
        uint32_t k = __atomic_add_fetch(&left, 1, __ATOMIC_RELAXED);
        if (!(k & (k - 1))) rt_log("lq-interleave: skipped %u draw(s) it could not rewrite (DKB_LQ_INTERLEAVE=2)", k);
        return 0;
    }
    if (lq_on) lq_interleave_state(&s, &lq);
    s.vs = vs, s.fs = fs, s.gs = gs;
    s.feedback = (uint8_t)((fbl & 0xff ? 1 : 0) | (fbl & 0x100 ? 2 : 0));
    s.prim = (uint8_t)prim;
    s.depth_minus_w = (uint8_t)(nvn_depth_minus_w() != 0);
    s.y_flip = 1;
    s.prim_restart = (uint8_t)(restart != 0);
    const VkPipeInfo *info = NULL;
    int pending;
    // A draw of at most 6 vertices over half the render area or more (a full-screen triangle or quad, a tile) waits for
    // its pipeline: skipping it can leave a black tile. The waits share 50 ms per presented frame
    // (TOTK_PIPE_SKIP=fullscreen skips them instead). A one-shot draw waits without that budget. Everything else is
    // skipped while its pipeline is built.
    static _Thread_local uint64_t wait_frame;
    static _Thread_local uint64_t wait_left_ns;
    uint32_t wait_ms = 0;
    uint64_t t0 = 0;
    if (oneshot > 0) {
        wait_ms = ONESHOT_WAIT_MS;
    } else if (skip < 2 && count <= 6 && (uint64_t)sc.extent.width * sc.extent.height * 2 >= (uint64_t)ex->pass_w * ex->pass_h) {
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (f != wait_frame) wait_frame = f, wait_left_ns = 50000000;
        wait_ms = (uint32_t)(wait_left_ns / 1000000);
        if (wait_ms) t0 = kern_now_ns();
    }
    // The previous draw's pipeline while its state repeats (PM, TOTK_PIPE_MEMO=0 turns it off): the cache key is made
    // from the state alone, and a pipeline once found only changes from its fast link to its optimised build, read
    // from the info the cache entry keeps.
    VkPipeline p;
    const char *why = NULL;
    uint64_t w0 = 0;
    if (pipe_memo_on() && PM.valid && !memcmp(&PM.s, &s, sizeof s)) {
        info = PM.info, pending = 0;
        p = __atomic_load_n(&info->pipeline, __ATOMIC_ACQUIRE);
    } else {
        w0 = t0 ? t0 : wait_ms ? kern_now_ns() : 0;   // the clock is read only for draws that may wait
        p = vk_pipeline_graphics_async(&s, &info, &pending, wait_ms);
        if (t0) {
            uint64_t spent = kern_now_ns() - t0;
            wait_left_ns = spent < wait_left_ns ? wait_left_ns - spent : 0;
        }
        why = oneshot > 0 ? "one-shot, shader" : wait_ms ? "full-screen" : NULL;
        if (pending && oneshot < 0 && !vk_exec_targets_recent(ex, skip >= 1)) {
            perf_count(PERF_CNT_ONESHOT_WAIT);
            why = "one-shot";
            if (!w0) w0 = kern_now_ns();
            p = vk_pipeline_graphics_async(&s, &info, &pending, ONESHOT_WAIT_MS);
        }
        PM.valid = !pending && p && info;
        if (PM.valid) PM.s = s, PM.info = info;
    }
    uint64_t waited = w0 ? kern_now_ns() - w0 : 0;
    static int wait_logs;
    if (why && waited >= 2000000 && wait_logs < 200) {
        char tg[512];
        wait_logs++;
        vk_exec_targets_describe(ex, tg, sizeof tg);
        rt_log("vk_exec: frame %llu: a draw waited %.1f ms for its pipeline (%s%s): %u vertices, pass %ux%u, %s",
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1, waited / 1e6, why,
               pending ? ", skipped after all" : "", count, ex->pass_w, ex->pass_h, tg);
    }
    if (pending) return 0;   // a background worker is building it: the draw is skipped until it is ready
    if (!p || !info) return warn_once(W_PIPELINE, "a graphics pipeline could not be created: draws with it are skipped"), 0;
    CSA(SP_PREP, ct_prep);
    ct_x = CST();
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    if (!B.st_valid || memcmp(&B.st, &s, sizeof s)) {
        vk_pipeline_set_dynamic(ex->cb, &s);
        ex->recorded++;
        B.st = s, B.st_valid = 1;
    }
    record_fixed(ex);
    repeat_stats(ex, info);
    CSA(SP_PIPE, ct_x);
    bind_resources(ex, info);
    ct_x = CST();
    if (fbl & ex->fb_dirty) feedback_barrier(ex);
    feedback_written(ex, &s);
    CSA(SP_POST, ct_x);
    ct_x = CST();
    bind_vertex_buffers(ex, &s, lq_on ? &lq : NULL);
    CSA(SP_VB, ct_x);
    if (lq_on) {
        static uint32_t applied;
        applied++;
        if (!(applied & (applied - 1)))
            rt_log("lq-interleave: applied to draw %u: %u instance streams in one binding of stride %u (DKB_LQ_INTERLEAVE=1)",
                   applied, lq.n, 16 * lq.n);
    }
    if (vk_dump_armed) *dump = (DumpDrawState){s, p, info};
    vk_exec_oq_draw(ex);
    return 1;
}

// TOTK_TRACE=perf: the draw or dispatch just recorded, with the NVN 2 names of its stages, for the slow-part line
// (vk_sync.c vk_sync_note). kind: 0 draw, 1 indirect draw, 2 dispatch, 3 indirect dispatch.
static void note_work(const VkExec *ex, int kind, uint64_t work) {
    if (rsp_worker || !vk_sync_noting() || !ex->prog) return;
    const uint8_t *ctl, *code;
    uint32_t len;
    int cs = kind >= 2;
    uint64_t a = find_stage(ex->prog, cs ? NVN_STAGE_CS : NVN_STAGE_VS, &ctl, &code, &len) ? nvn2_origin(code) : 0;
    uint64_t b = !cs && find_stage(ex->prog, NVN_STAGE_FS, &ctl, &code, &len) ? nvn2_origin(code) : 0;
    vk_sync_note(kind, a, b, work);
}

static void bind_index(VkExec *ex, const void *data, uint64_t bytes, VkIndexType type) {
    VkBuffer b;
    VkDeviceSize o;
    memcpy(scratch(ex, bytes, &b, &o), data, bytes);
    vkCmdBindIndexBuffer(ex->cb, b, o, type);
    dc_index_reset(&B.idx);
    ex->recorded++;
}

// The Bananza reset vertex program loses stores through pointers loaded from a list. Match both locally confirmed
// stages and the observed one-instance point-draw form. Other programs and parameter forms keep the shader path.
static _Thread_local int rsp_reset_probe;   // rsp_draw_serial: only ask whether indirect_reset_draw would take the draw
static int indirect_reset_draw(VkExec *ex, const p_draw *d) {
    // Both settings are fixed for the run (TOTK_FIX_INDIRECT_RESET needs a restart), so read them once: this
    // runs ahead of every draw.
    static int on = -1;
    if (on < 0) on = !rt_game_version && cfg_bool("TOTK_FIX_INDIRECT_RESET") && !getenv("DKB_SKIP_DRAWS");
    if (!on || !c0_on() || ex->render_disabled ||
        !ex->prog || ex->prog->nstage != 2 || d->prim != NVN_DRAW_PRIMITIVE_POINTS ||
        d->count <= 0 || d->first < 0 || d->base_instance || d->instances != 1) return 0;
    const uint8_t *ctl, *code;
    uint32_t length;
    if (!find_stage(ex->prog, NVN_STAGE_VS, &ctl, &code, &length) || nvn2_origin(code) != 0xe78abde51f2d6bfaull ||
        !find_stage(ex->prog, NVN_STAGE_FS, &ctl, &code, &length) || nvn2_origin(code) != 0x57b0d7b4959f03c0ull) return 0;
    if (rsp_split_on && ex->prog && !rsp_worker && rsp_reset_probe) return 1;
    const p_bind_buf *list = &ex->ssbo[NVN_STAGE_VS][0];
    // c0_table encodes the binding's size as u32 at +0x118; use the same extent, including its zero rejection.
    if (!vk_exec_reset_indirect(ex, list->addr, (uint32_t)list->size, d->first, d->count)) {
        warn_once(W_RESET, "the matched indirect reset has an unsupported list or record: using its shader");
        return 0;
    }
    return 1;
}

static void op_draw(VkExec *ex, const p_draw *d) {
    c0_binst = (uint32_t)d->base_instance;
    if (indirect_reset_draw(ex, d)) return;
    DumpDrawState dump;
    if (d->count <= 0 || !prepare_draw(ex, d->prim, d->count, 0, &dump, d, NULL)) return;
    if (!rsp_worker) vk_exec_frame.draws++;
    uint32_t inst = d->instances > 0 ? (uint32_t)d->instances : 1, first = (uint32_t)d->first, n = (uint32_t)d->count;
    if (d->prim == NVN_DRAW_PRIMITIVE_QUADS && n >= 4) {   // quads become two triangles each
        uint32_t q = n / 4;
        uint32_t *ix = malloc(sizeof *ix * 6 * q);
        if (!ix) rt_fatal("vk: out of memory");
        for (uint32_t i = 0; i < q; i++) {
            uint32_t v = first + 4 * i, k[6] = {0, 1, 2, 0, 2, 3};
            for (int j = 0; j < 6; j++) ix[6 * i + j] = v + k[j];
        }
        bind_index(ex, ix, sizeof *ix * 6 * q, VK_INDEX_TYPE_UINT32);
        free(ix);
        vkCmdDrawIndexed(ex->cb, 6 * q, inst, 0, 0, (uint32_t)d->base_instance);
    } else if (d->prim == NVN_DRAW_PRIMITIVE_LINE_LOOP && n >= 2) {   // a strip that returns to its first vertex
        uint32_t *ix = malloc(sizeof *ix * (n + 1));
        if (!ix) rt_fatal("vk: out of memory");
        for (uint32_t i = 0; i < n; i++) ix[i] = first + i;
        ix[n] = first;
        bind_index(ex, ix, sizeof *ix * (n + 1), VK_INDEX_TYPE_UINT32);
        free(ix);
        vkCmdDrawIndexed(ex->cb, n + 1, inst, 0, 0, (uint32_t)d->base_instance);
    } else {
        vkCmdDraw(ex->cb, n, inst, first, (uint32_t)d->base_instance);
    }
    ex->recorded++;
    if (vk_dump_armed) vk_dump_draw(ex, &dump.state, dump.pipeline, dump.info, OP_DRAW, d);
    note_work(ex, 0, (uint64_t)n * inst);
}

// Primitive restart of an indexed draw. NVN restarts strips wherever an index equals the SetPrimitiveRestart value;
// Vulkan restarts only at the all-ones value of the bound index type, and only strips and fans without
// primitiveTopologyListRestart. Byte indices are widened to 16 bits, so their restart value becomes 0xffff
// (op_draw_idx). Without restart the 0xffff entries of TOTK's terrain and cliff strips fetch vertex 65535 + base and
// the triangles around them drop out.
static int restart_for(const VkExec *ex, int prim, int itype) {
    if (!ex->prim_restart.enable) return 0;
    switch (vk_pipeline_topology(prim)) {
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: break;
    default: return 0;   // lists: a restart index only drops the unfinished primitive
    }
    static const uint32_t all_ones[3] = {0xffu, 0xffffu, 0xffffffffu};
    if (ex->prim_restart.index == all_ones[itype]) return 1;
    warn_once(W_RESTART, "primitive restart index %#x with %d-byte indices is not replayed: strips drawn without restart",
              ex->prim_restart.index, 1 << itype);
    return 0;
}

static void op_draw_idx(VkExec *ex, const p_draw_idx *d) {
    c0_binst = (uint32_t)d->base_instance;
    static const uint32_t isize[3] = {1, 2, 4};
    if (d->itype < 0 || d->itype > 2) return warn_once(W_INDEX, "index type %d is unknown: draw skipped", d->itype);
    int restart = restart_for(ex, d->prim, d->itype);
    DumpDrawState dump;
    if (d->count <= 0 || !prepare_draw(ex, d->prim, d->count, restart, &dump, NULL, d)) return;
    if (!rsp_worker) vk_exec_frame.draws++;
    uint64_t bytes = (uint64_t)d->count * isize[d->itype];
    VkMemSpan sp;
    if (!vk_mem_resolve(d->ibuf, bytes, &sp)) return warn_once(W_INDEX, "index buffer %#llx is not backed: draws with it are skipped", (unsigned long long)d->ibuf);
    if (d->itype == NVN_INDEX_TYPE_UNSIGNED_BYTE) {   // no uint8 index type here: widen (host-readable backing only)
        if (sp.alloc) return warn_once(W_INDEX, "byte indices in device-local memory: draw skipped");
        vk_mem_count(sp.cls, VK_MUSE_CPU_READ, bytes);
        vk_mem_count(VK_MCLS_SCRATCH, VK_MUSE_IB, 2 * bytes);
        uint16_t *ix = malloc(sizeof *ix * (size_t)d->count);
        if (!ix) rt_fatal("vk: out of memory");
        for (int i = 0; i < d->count; i++) {
            uint8_t v = ((const uint8_t *)(uintptr_t)d->ibuf)[i];
            ix[i] = restart && v == 0xff ? 0xffff : v;
        }
        bind_index(ex, ix, sizeof *ix * (size_t)d->count, VK_INDEX_TYPE_UINT16);
        free(ix);
    } else if (sp.offset % isize[d->itype]) {
        return warn_once(W_INDEX, "index buffer %#llx is not aligned to its index size: draw skipped", (unsigned long long)d->ibuf);
    } else {
        VK_MEM_USE(sp);
        VkIndexType ity = d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        if (!fastpath_on() || !dc_index_redundant(&B.idx, (uint64_t)sp.buffer, sp.offset, (int)ity)) {
            vkCmdBindIndexBuffer(ex->cb, sp.buffer, sp.offset, ity);
            ex->recorded++;
        }
        vk_mem_count(sp.cls, VK_MUSE_IB, bytes);
        VK_MEM_BOUND(sp);
    }
    vkCmdDrawIndexed(ex->cb, (uint32_t)d->count, d->instances > 0 ? (uint32_t)d->instances : 1, 0, d->base_vertex,
                     (uint32_t)d->base_instance);
    ex->recorded++;
    if (vk_dump_armed) vk_dump_draw(ex, &dump.state, dump.pipeline, dump.info, OP_DRAW_IDX, d);
    note_work(ex, 0, (uint64_t)d->count * (uint64_t)(d->instances > 0 ? d->instances : 1));
}

// DrawElementsIndirect: the arguments are read by the GPU from the pool buffer (they are often written by an earlier
// dispatch), as nxnative nvn_mtl.m does. The base instance stays out of c0 (0x644): c0 cannot be patched by a transfer
// inside a pass; gl_BaseInstance of the Vulkan draw is the argument's. Byte indices need the count to be widened: skipped.
static void op_draw_idx_ind(VkExec *ex, const p_draw_idx_ind *d) {
    c0_binst = 0;
    if (d->itype != NVN_INDEX_TYPE_UNSIGNED_SHORT && d->itype != NVN_INDEX_TYPE_UNSIGNED_INT)
        return warn_once(W_INDEX, "indirect draw with index type %d: skipped", d->itype);
    VkMemSpan ib, ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDrawIndexedIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect draw arguments at %#llx are not backed or not aligned: draws skipped", (unsigned long long)d->ind);
    if (!vk_mem_resolve(d->ibuf, 1, &ib) || ib.offset % (d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4))
        return warn_once(W_INDEX, "index buffer %#llx is not backed or not aligned: indirect draws skipped", (unsigned long long)d->ibuf);
    DumpDrawState dump;
    if (!prepare_draw(ex, d->prim, 0x7fffffff, restart_for(ex, d->prim, d->itype), &dump, NULL, NULL)) return;
    if (!rsp_worker) vk_exec_frame.draws++;
    VK_MEM_USE(ip);
    VK_MEM_USE(ib);
    {
        VkIndexType ity = d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        if (!fastpath_on() || !dc_index_redundant(&B.idx, (uint64_t)ib.buffer, ib.offset, (int)ity)) {
            vkCmdBindIndexBuffer(ex->cb, ib.buffer, ib.offset, ity);
            ex->recorded++;
        }
    }
    vkCmdDrawIndexedIndirect(ex->cb, ip.buffer, ip.offset, 1, sizeof(VkDrawIndexedIndirectCommand));
    ex->recorded += 2;
    if (vk_dump_armed) vk_dump_draw(ex, &dump.state, dump.pipeline, dump.info, OP_DRAW_IDX_IND, d);
    note_work(ex, 1, 0);
    // An imported pool's GPU address is its host address (bind_range reads it the same way); device memory is not.
    if (!rsp_worker && !ip.alloc && vk_sync_noting()) vk_sync_note_args(d->ind);
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDrawIndexedIndirectCommand));
    VK_MEM_BOUND(ip);
    VK_MEM_BOUND(ib);
}

// ---- dispatch and uniform buffer updates

static void op_dispatch(VkExec *ex, const p_dispatch *d) {
    if (d->x <= 0 || d->y <= 0 || d->z <= 0) return;
    static int skip = -1;   // DKB_SKIP_COMPUTE=1: drop every direct dispatch (bisecting GPU faults)
    if (skip < 0) skip = getenv("DKB_SKIP_COMPUTE") != NULL;
    if (skip) return;
    if (!ex->prog) return warn_once(W_PROGRAM, "a dispatch without a bound program is skipped");
    cb_res(ex);
    dummies_init();
    int present;
    uint64_t cw0 = kern_now_ns();
    const GfxShader *cs = shader_for(ex, NVN_STAGE_CS, &present, NULL, 0);   // synchronous: a skipped dispatch changes results
    wait_log(ex, "dispatch", "compute shader translation or load", NVN_STAGE_CS, kern_now_ns() - cw0);
    if (!cs) return;
    const VkPipeInfo *info = NULL;
    cw0 = kern_now_ns();
    VkPipeline p = vk_pipeline_compute(cs, &info);
    wait_log(ex, "dispatch", "compute pipeline build", NVN_STAGE_CS, kern_now_ns() - cw0);
    if (!p || !info) return warn_once(W_PIPELINE, "a compute pipeline could not be created: dispatches with it are skipped");
    vk_exec_barrier(ex);   // ends the pass; earlier writes are visible to the shader
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    c0_dim[0] = (uint32_t)d->x, c0_dim[1] = (uint32_t)d->y, c0_dim[2] = (uint32_t)d->z;
    c0_lay_set(ex, cs);
    twin_dispatch = 1;
    bind_resources(ex, info);
    twin_dispatch = 0;
    if (vk_dump_armed) vk_dump_boundary(ex, "dispatch_before", 0);
    vkCmdDispatch(ex->cb, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z);
    ex->recorded++;
    if (vk_dump_armed) vk_dump_boundary(ex, "dispatch_after", 0);
    note_work(ex, 2, (uint64_t)d->x * (uint64_t)d->y * (uint64_t)d->z);
    if (vk_dump_armed) vk_dump_dispatch(ex, cs, p, info, d);
}

// DispatchComputeIndirect: vkCmdDispatchIndirect from the pool buffer. NVN 1 shaders read the group counts at c0+0x200,
// so they are copied there on the GPU before dispatch. That NVN 1 offset can hold declared roots or resource tables
// in NVN 2; leave its packed bank unchanged rather than overwrite it with the legacy copy.
static void op_dispatch_ind(VkExec *ex, const p_dispatch_ind *d) {
    if (!ex->prog) return warn_once(W_PROGRAM, "a dispatch without a bound program is skipped");
    VkMemSpan ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDispatchIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect dispatch arguments at %#llx are not backed or not aligned: skipped", (unsigned long long)d->ind);
    cb_res(ex);
    dummies_init();
    int present;
    uint64_t cw0 = kern_now_ns();
    const GfxShader *cs = shader_for(ex, NVN_STAGE_CS, &present, NULL, 0);
    wait_log(ex, "indirect dispatch", "compute shader translation or load", NVN_STAGE_CS, kern_now_ns() - cw0);
    if (!cs) return;
    const VkPipeInfo *info = NULL;
    cw0 = kern_now_ns();
    VkPipeline p = vk_pipeline_compute(cs, &info);
    wait_log(ex, "indirect dispatch", "compute pipeline build", NVN_STAGE_CS, kern_now_ns() - cw0);
    if (!p || !info) return warn_once(W_PIPELINE, "a compute pipeline could not be created: dispatches with it are skipped");
    vk_exec_barrier(ex);
    VK_MEM_USE(ip);
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    c0_dim[0] = c0_dim[1] = c0_dim[2] = 0;
    c0_lay_set(ex, cs);
    B.c0_valid[2] = 0;   // a fresh c0 range: the copy below must not change one an earlier dispatch reads
    twin_dispatch = 1;
    bind_resources(ex, info);
    twin_dispatch = 0;
    if (!c0_lay && B.c0_valid[2]) {   // the NVN 1 shader binds c0
        VkBufferCopy cp = {ip.offset, B.c0_off[2] + 0x200, sizeof(VkDispatchIndirectCommand)};
        vkCmdCopyBuffer(ex->cb, ip.buffer, B.c0_buf[2], 1, &cp);
        ex->recorded++;
        vk_exec_barrier(ex);
        B.c0_valid[2] = 0;   // the range no longer holds B.c0[2]
    }
    vkCmdDispatchIndirect(ex->cb, ip.buffer, ip.offset);
    ex->recorded++;
    note_work(ex, 3, 0);
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDispatchIndirectCommand));
    VK_MEM_BOUND(ip);
    if (vk_dump_armed) vk_dump_dispatch(ex, cs, p, info, &(p_dispatch){0, 0, 0, d->caller});
}

static void op_update_ub(VkExec *ex, const p_update_ub *u) {
    if (!u->size) return;
    VkMemSpan sp;
    if (!vk_mem_resolve(u->dst, u->size, &sp)) return warn_once(W_UB, "UpdateUniformBuffer to %#llx is not backed: dropped", (unsigned long long)u->dst);
    cb_res(ex);
    vk_exec_barrier(ex);   // earlier draws finished with the old contents
    VkBuffer b;
    VkDeviceSize o;
    memcpy(scratch(ex, u->size, &b, &o), u->data, u->size);
    VK_MEM_PICK(sp);
    VkBufferCopy cp = {o, sp.offset, u->size};
    vkCmdCopyBuffer(ex->cb, b, sp.buffer, 1, &cp);
    vk_mem_gpu_wrote(&sp, u->size);
    vk_usnap_drop(u->dst, u->size);
    ex->recorded++;
    vk_mem_count(sp.cls, VK_MUSE_GPU_WRITE, u->size);
    note_written(ex, u->dst, u->size);
    upd_note(ex, u);
}

// ---- bind ops

static void geometry_context_buffer_log(unsigned record, const char *name, const p_bind_buf *binding) {
    VkMemInspect view = {0};
    int backed = binding->addr && binding->size && vk_mem_inspect(binding->addr, binding->size, &view);
    rt_log("geometry_buffer_context: record=%u range=%s snapshot=passive requested=%llu full_range_backed_now=%d "
           "pool_kind=%u pool_flags=%#x class_before_mirror_pick=%d offset=%llu available=%llu "
           "mirror_state=%d mirror_present=%d shadow_storage=%d buffer_present=%d contents=unread descriptor=unobserved",
           record, name, (unsigned long long)binding->size, backed, view.pool_kind, view.pool_flags, view.cls,
           (unsigned long long)view.offset, (unsigned long long)view.avail,
           view.mirror_state, view.mirror_present, view.shadow_storage, view.buffer_present);
}

// Opt-in draw-ingress metadata only. Replay is serialized; the cap is process-wide and later draws are unobserved.
static void geometry_context_log(const VkExec *ex, uint32_t op, const void *payload) {
    if (op != OP_DRAW && op != OP_DRAW_IDX && op != OP_DRAW_IDX_IND) return;
    static int on = -1, filtered;
    static uint64_t filter;
    static unsigned records;
    if (on < 0) {
        const char *e = getenv("DKB_GEOMETRY_CONTEXT"), *f = getenv("DKB_GEOMETRY_CONTEXT_ORIGIN");
        on = e && !strcmp(e, "1");
        if (on && f) {
            filtered = 1;
            if (f[0] == '0' && (f[1] == 'x' || f[1] == 'X')) f += 2;
            unsigned digits = 0;
            for (; *f; f++) {
                unsigned d = *f >= '0' && *f <= '9' ? (unsigned)(*f - '0') :
                             *f >= 'a' && *f <= 'f' ? (unsigned)(*f - 'a' + 10) :
                             *f >= 'A' && *f <= 'F' ? (unsigned)(*f - 'A' + 10) : 16;
                if (d == 16 || digits == 16) { on = 0; break; }
                filter = (filter << 4) | d;
                digits++;
            }
            if (!digits) on = 0;
            if (!on) rt_log("geometry_context: invalid origin filter; metadata logging disabled");
        }
    }
    if (!on || records == 12 || !ex || !payload) return;

    uint64_t origin[3] = {0};
    unsigned present = 0;
    int metadata = ex->prog && ex->prog->nstage <= 6;
    // Failed native lowering can leave a compact control: do not send it to the legacy-shaped stage getter.
    if (metadata)
        for (uint32_t i = 0; i < ex->prog->nstage; i++) {
            const uint8_t *ctl = ex->prog->sd[i].control;
            if (ctl && gfx_shader_is_nvn2(ctl)) { metadata = 0; break; }
        }
    if (metadata)
        for (int stage = 0; stage < 3; stage++) {
            const uint8_t *ctl, *code;
            uint32_t len;
            if (find_stage(ex->prog, stage, &ctl, &code, &len)) {
                present |= 1u << stage;
                origin[stage] = nvn2_origin(code);   // pointer-to-origin lookup, no shader bytes read
            }
        }
    // Reserve the bounded log for programs with a geometry stage. Retain unavailable metadata records so a
    // failed native lowering is not silently hidden; never infer a stage from an unreadable control block.
    if (metadata && !(present & (1u << NVN_STAGE_GS))) return;
    if (filtered && !((present & 1) && origin[0] == filter) &&
        !((present & 2) && origin[1] == filter) && !((present & 4) && origin[2] == filter)) return;

    const char *kind;
    int prim;
    char draw[256], viewport[192], attachments[512], bindings[384];
    if (op == OP_DRAW) {
        const p_draw *d = payload;
        kind = "direct", prim = d->prim;
        snprintf(draw, sizeof draw, "first=%d count=%d instances_raw=%d base_instance=%d",
                 d->first, d->count, d->instances, d->base_instance);
    } else if (op == OP_DRAW_IDX) {
        const p_draw_idx *d = payload;
        kind = "indexed", prim = d->prim;
        snprintf(draw, sizeof draw, "count=%d instances_raw=%d base_vertex=%d base_instance=%d index_type=%d index_addr=%#llx",
                 d->count, d->instances, d->base_vertex, d->base_instance, d->itype, (unsigned long long)d->ibuf);
    } else {
        const p_draw_idx_ind *d = payload;
        kind = "indexed_indirect", prim = d->prim;
        snprintf(draw, sizeof draw, "index_type=%d index_addr=%#llx indirect_addr=%#llx count=unknown instances=unknown",
                 d->itype, (unsigned long long)d->ibuf, (unsigned long long)d->ind);
    }
    if (ex->has_vp)
        snprintf(viewport, sizeof viewport, "has_vp=1 viewport0_stored=[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g] depth_range=[%.9g,%.9g]",
                 ex->vp[0][0], ex->vp[0][1], ex->vp[0][2], ex->vp[0][3], ex->vp[0][4], ex->vp[0][5],
                 ex->depth_range[0], ex->depth_range[1]);
    else
        snprintf(viewport, sizeof viewport, "has_vp=0 viewport0_stored=unreported depth_range=[%.9g,%.9g]", ex->depth_range[0], ex->depth_range[1]);
    if (ex->rt_resolved) {
        size_t n = (size_t)snprintf(attachments, sizeof attachments, "attachments=resolved pass_noatt=%d pass_layers=%u layer_views=mip:base:count",
                                  ex->pass_noatt, ex->pass_layers);
        for (int i = 0; i < 8; i++)
            if (ex->color[i].image && n < sizeof attachments) {
                const VkExecTarget *t = &ex->color[i];
                n += (size_t)snprintf(attachments + n, sizeof attachments - n, " color%d=%u:%u:%u", i, t->level, t->base_layer, t->layers);
            }
        if (ex->depth.image && n < sizeof attachments)
            snprintf(attachments + n, sizeof attachments - n, " depth=%u:%u:%u", ex->depth.level, ex->depth.base_layer, ex->depth.layers);
    } else {
        snprintf(attachments, sizeof attachments, "attachments=unresolved pass_layers=unknown");
    }
    const p_bind_buf *s0 = &ex->ssbo[NVN_STAGE_GS][0], *s1 = &ex->ssbo[NVN_STAGE_GS][1], *u0 = &ex->ubo[NVN_STAGE_GS][0];
    snprintf(bindings, sizeof bindings, "gs_ranges=recorded_pod_unresolved gs_ssbo0=%#llx:%llu:%s gs_ssbo1=%#llx:%llu:%s gs_ubo0=%#llx:%llu:%s",
             (unsigned long long)s0->addr, (unsigned long long)s0->size, s0->addr && s0->size ? "range" : "empty_or_unset",
             (unsigned long long)s1->addr, (unsigned long long)s1->size, s1->addr && s1->size ? "range" : "empty_or_unset",
             (unsigned long long)u0->addr, (unsigned long long)u0->size, u0->addr && u0->size ? "range" : "empty_or_unset");
    rt_log("geometry_context: record=%u limit=12 kind=%s op=%u prim=%d %s fence=%llu program=%llu "
           "bound_stage_mask=%#x stage_metadata=%s present_stage_mask=%#x vs=%016llx gs=%016llx fs=%016llx "
           "render_disabled=%d render_cond_present=%d raster_discard=%u pass_active=%d %s %s %s "
           "clip_cull=untracked xfb=untracked guest_viewport_count=unknown guest_viewport_mask=unknown viewport0_update_provenance=untracked layer_routing=unknown",
           ++records, kind, op, prim, draw, (unsigned long long)ex->fence,
           ex->prog ? (unsigned long long)ex->prog->serial : 0ull, (unsigned)ex->prog_stages,
           metadata ? "available" : "unavailable", present,
           (unsigned long long)origin[0], (unsigned long long)origin[2], (unsigned long long)origin[1],
           ex->render_disabled, ex->has_render_cond, (unsigned)ex->pipe.raster_discard, ex->in_pass, viewport, attachments, bindings);
    geometry_context_buffer_log(records, "gs_ssbo0", s0);
    geometry_context_buffer_log(records, "gs_ssbo1", s1);
    geometry_context_buffer_log(records, "gs_ubo0", u0);
}

static void bind_handles(uint64_t *dst, int cap, const p_bind_handles *b) {
    if (b->stage < 0 || b->stage >= 6 || b->first < 0 || b->n < 0 || b->n > 32 || b->first + b->n > cap) {
        warn_once(W_BIND, "a bind of %d handles at %d of stage %d is out of range: ignored", b->n, b->first, b->stage);
        return;
    }
    memcpy(&dst[b->first], b->h, sizeof(uint64_t) * (size_t)b->n);
}

#define DRAW_TIMED(call) do { if (split_on()) split_begin_draw(); uint64_t ct_all = CST(); call; CSA(SP_TOTAL, ct_all); split_end_draw(); } while (0)

// The state ops that decide what a draw binds (program, uniform and storage buffers, texture, sampler and image handles, root push constants).
// live: the replay itself (they also bump the shared bind serials); 0: a shadow state kept by the replay pre-pass (vk_exec.c).
void vk_exec_apply_binding(VkExec *ex, uint32_t op, const void *payload, int live) {
    switch (op) {
    case OP_BIND_PROGRAM: {
        const p_bind_program *b = payload;
        ex->prog = b->prog, ex->prog_stages = b->stages;
        if (live) bind_gen++;
        break;
    }
    case OP_BIND_UBO: {
        const p_bind_buf *b = payload;
        if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 18) ex->ubo[b->stage][b->index] = *b;
        if (live) bind_gen++;
        break;
    }
    case OP_PUSH_CONST: {
        const p_push_const *p = payload;
        if (p->table < 0 || p->table >= NVN2_ROOT_TABLES || (!p->compute && p->table != 0)) {
            static int warned;
            if (!warned++) rt_log("vk: push constants into root table %d dropped", p->table);
            break;
        }
        int k = p->compute != 0, w0 = p->offset / 4, n = p->bytes / 4;
        memcpy(&ex->push[k][p->table][w0], p->w, 4 * (size_t)n);
        ex->push_set[k][p->table] |= (n >= 64 ? ~0ull : ((1ull << n) - 1)) << w0;
        if (live) bind_gen++;
        break;
    }
    case OP_BIND_SSBO: {
        const p_bind_buf *b = payload;
        if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16) ex->ssbo[b->stage][b->index] = *b;
        else {   // past the table: dropped (NVN 2 binds more storage buffers per stage than NVN 1's 16)
            static uint64_t seen[6];
            if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 64 && !(seen[b->stage] >> b->index & 1))
                seen[b->stage] |= 1ull << b->index, rt_log("vk: storage buffer %d of stage %d dropped", b->index, b->stage);
        }
        if (live) bind_gen++;
        break;
    }
    case OP_BIND_TEX: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 6 ? ex->tex[b->stage] : NULL, 128, b);
        if (live) ex->tex_serial++;
        if (live) bind_gen++;
        break;
    }
    case OP_BIND_SAMP: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 6 ? ex->samp[b->stage] : NULL, 128, b);
        break;
    }
    case OP_BIND_IMG: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 6 ? ex->img[b->stage] : NULL, 32, b);
        if (live) ex->tex_serial++;   // the shader memo's handle check (feedback_scan reads only textures: a rescan at most)
        if (live) bind_gen++;
        break;
    }
    default: break;
    }
}

// The fixed-function and vertex state ops (no side effect but ex): 1 when op is one of them.
static int apply_fixed_state(VkExec *ex, uint32_t op, const void *payload) {
    switch (op) {
    case OP_BIND_STATE: {
        const p_bind_state *b = payload;
        blend_st bs;
        switch (b->kind) {
        case 0:
            memcpy(&bs, b->bytes, sizeof bs);
            if (bs.target < 8) ex->pipe.blend[bs.target] = bs;
            break;
        case 1: memcpy(&ex->pipe.color, b->bytes, sizeof ex->pipe.color); break;
        case 2: memcpy(&ex->pipe.chmask, b->bytes, sizeof ex->pipe.chmask); break;
        case 3: memcpy(&ex->pipe.ms, b->bytes, sizeof ex->pipe.ms); break;
        case 4: memcpy(&ex->pipe.poly, b->bytes, sizeof ex->pipe.poly); break;
        case 5: memcpy(&ex->pipe.ds, b->bytes, sizeof ex->pipe.ds); break;
        case 6: memcpy(&ex->point_size, b->bytes, sizeof ex->point_size); break;
        case 7: memcpy(&ex->pipe.alpha_ref, b->bytes, sizeof ex->pipe.alpha_ref); break;
        }
        break;
    }
    case OP_BIND_VATTR: {
        const p_bind_vattr *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memset(ex->pipe.vattr, 0, sizeof ex->pipe.vattr);   // MNG: the new list replaces the old one
        memcpy(&ex->pipe.vattr[b->first], b->st, sizeof(uint32_t) * (size_t)b->n);
        break;
    }
    case OP_BIND_VSTREAM: {
        const p_bind_vstream *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memset(ex->pipe.vstream, 0, sizeof ex->pipe.vstream);
        memcpy(&ex->pipe.vstream[b->first], b->st, sizeof(vs_st) * (size_t)b->n);
        break;
    }
    case OP_BIND_VBUF: {
        const p_bind_vbuf *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memcpy(&ex->vbuf[b->first], b->r, sizeof(NVNbufferRange) * (size_t)b->n);
        break;
    }
    default: return 0;
    }
    return 1;
}

// TOTK_REPLAY_SPLIT: a bind op's effect on the state alone, for the walk thread (which only keeps the state) and the workers.
void vk_exec_split_bind(VkExec *ex, uint32_t op, const void *payload) {
    if (!apply_fixed_state(ex, op, payload)) vk_exec_apply_binding(ex, op, payload, 1);
}

void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload) {
    if (census_on()) census_op(op);
    geometry_context_log(ex, op, payload);
    if (apply_fixed_state(ex, op, payload)) return;
    switch (op) {
    case OP_BIND_PROGRAM: case OP_BIND_UBO: case OP_PUSH_CONST: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG:
        vk_exec_apply_binding(ex, op, payload, 1);
        break;
    case OP_UPDATE_UB: op_update_ub(ex, payload); break;
    case OP_DRAW: DRAW_TIMED(op_draw(ex, payload)); break;
    case OP_DRAW_IDX: DRAW_TIMED(op_draw_idx(ex, payload)); break;
    case OP_DISPATCH: op_dispatch(ex, payload); break;
    case OP_DRAW_IDX_IND: DRAW_TIMED(op_draw_idx_ind(ex, payload)); break;
    case OP_DISPATCH_IND: op_dispatch_ind(ex, payload); break;
    default: rt_fatal("vk: op %s is not a draw op", nvn_op_name(op));
    }
}

// ---- TOTK_REPLAY_SPLIT=<threads>: the walk itself split across threads (MT_DESIGN.md, version (ii)). Needs TOTK_BINDLESS_TEX=1 and
// TOTK_BINDLESS_BUF=1 and the recorder thread; replaces TOTK_REPLAY_MT while on.
// The walk thread still walks the whole op stream, but a run of draws and the binds and state ops between them only update the state
// (vk_exec_split_bind, vk_exec_state_op) and go into a chunk (about TOTK_REPLAY_SPLIT_DRAWS draws, default 1000): the ops' addresses and a
// copy of the state at the chunk's first draw. Everything else (render target changes, clears, barriers, copies, dispatches, indirect
// draws, uniform updates, draws that need the walk thread) ends the chunk and runs on the walk thread as before, in order.
// A worker replays a chunk on its copy of the state with the ordinary draw path (thread-local B, RS, BS, DS, memos and descriptor pools:
// that is the "own execution context") into a RecChunk, a packet buffer of vk_rec.c; the walk thread put the chunk's place into the
// recorder's ring when it closed the chunk, and the recorder makes the chunk's calls when the worker published it. The chunk starts as a
// new command buffer does (every dynamic state, the pipeline, the sets and the vertex buffers are recorded again), inside the pass the
// walk thread began. Draws are numbered in the submission; uniform snapshot lookups and drops (vk_usnap.c) are made as of that number.
// What a worker cannot do: end the pass (a feedback loop's restart: the target is learned for the next pass and the draw gets a
// by-region barrier instead, counted), insert an alias copy (the alias starts empty, counted).
// Known gaps: drops made by a draw on another worker (a shader-written storage buffer over a snapshot uniform range) are seen only
// when they were made before the lookup in time, not in draw order; fb_dirty at a chunk start is assumed full.
#define RSP_MAX_WORKERS 16
typedef struct RspChunk {
    struct RspChunk *qnext, *lnext;
    VkExec start;                        // the state at the chunk's first draw
    const op_hdr **ops;
    uint32_t nops, cap, seq0, ndraws, upd_n;
    uint64_t bind_gen0;
    RecChunk *rec;
} RspChunk;

int vk_exec_split_active;
static struct {
    int state;                           // 0 not read, 1 on, -1 off
    int workers;
    PlatMutex m;
    PlatCond work_cv, idle_cv;
    RspChunk *qh, *qt, *freel, *all, *cur;
    int pending, stop;
    uint32_t seq, pass_serial, chunk_draws, ext;
    int after_chunk;
    uint64_t n[RSPN_N];
    uint64_t subs, chunks, draws, run_n, run_ns, run_max, join_n, join_ns, serial_draws[5], cut[OP_COUNT_], bind_ops, state_ops;
} RSP;

static void rsp_worker_main(void *arg);

static int rsp_on(void) {
    if (RSP.state) return RSP.state > 0;
    RSP.state = -1;
    const char *e = getenv("TOTK_REPLAY_SPLIT");
    int n = e ? atoi(e) : 0;
    if (n <= 0) return 0;
    if (!vkb_buf_mode()) {
        rt_log("TOTK_REPLAY_SPLIT=%d needs TOTK_BINDLESS_TEX=1 and TOTK_BINDLESS_BUF=1: off", n);
        return 0;
    }
    if (n > RSP_MAX_WORKERS) n = RSP_MAX_WORKERS;
    // The lazily read switches and tables the draw path shares: made once here, on the walk thread, before a worker reads them.
    dummies_init(), limits_init();
    (void)c0_on(), (void)native_gs_on(), (void)bind_skip_on(), (void)fastpath_on(), (void)td_mode(), (void)bc_on();
    (void)shader_memo_on(), (void)pipe_memo_on(), (void)pipe_skip_mode(), (void)census_on();
    const char *d = getenv("TOTK_REPLAY_SPLIT_DRAWS");
    { const char *x = getenv("TOTK_REPLAY_SPLIT_EXT"); RSP.ext = x ? (uint32_t)atoi(x) : 3; }   // bit 0: indirect draws in chunks; bit 1: uniform snapshot begun on a helper thread
    RSP.chunk_draws = d && atoi(d) > 0 ? (uint32_t)atoi(d) : 1000;
    plat_mutex_init(&RSP.m), plat_cond_init(&RSP.work_cv), plat_cond_init(&RSP.idle_cv);
    RSP.workers = n;
    rsp_split_on = 1;
    __atomic_store_n(&vk_split_active, 1, __ATOMIC_RELEASE);
    RSP.state = 1;
    for (int i = 0; i < n; i++) plat_thread_spawn(rsp_worker_main, NULL, "replay-split", 0);
    rt_log("TOTK_REPLAY_SPLIT: %d replay worker threads, chunks of %u draws", n, RSP.chunk_draws);
    return 1;
}

int vk_exec_split_ext(uint32_t bit) { return RSP.state > 0 && (RSP.ext & bit); }
static void rsp_note(int kind) { __atomic_fetch_add(&RSP.n[kind], 1, __ATOMIC_RELAXED); }
void vk_exec_split_refused(void) { rsp_note(RSPN_ALIAS); }

static void rsp_begin(int work) {
    vk_exec_split_active = 0;
    if (work || !rsp_on() || vk_dump_armed || !rec_split_ok()) return;
    RSP.cur = NULL, RSP.seq = 0, RSP.pass_serial = 0, RSP.after_chunk = 0;
    vk_usnap_split_seq(0);
    vk_exec_split_active = 1;
    RSP.subs++;
}

// A chunk: the first draw's state, then the ops after it.
static void rsp_run(RspChunk *c) {
    static _Thread_local VkExec wx;
    VkExec *x = &wx;
    uint64_t t0 = plat_now_ns();
    *x = c->start;
    x->dirty = VK_EXEC_DIRTY_ALL, x->recorded = 0, x->fb_fs = NULL;
    memset(&B, 0, sizeof B);   // the chunk starts as a new command buffer does
    bind_gen = c->bind_gen0;
    upd_cap = c->upd_n;
    rec_chunk_begin(c->rec);
    uint32_t seq = c->seq0;
    for (uint32_t i = 0; i < c->nops; i++) {
        const op_hdr *h = c->ops[i];
        const void *a = nvn_op_payload(h);
        if (h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND) {
            vk_usnap_split_seq(++seq);
            vk_exec_op_draw(x, h->op, a);
        } else if (!vk_exec_state_op(x, h->op, a)) {
            vk_exec_split_bind(x, h->op, a);
        }
    }
    rec_chunk_end();
    vk_usnap_split_seq(0);
    vk_usnap_split_flush(), vk_mem_split_flush();
    rec_chunk_publish(c->rec);
    uint64_t ns = plat_now_ns() - t0;
    __atomic_fetch_add(&RSP.run_ns, ns, __ATOMIC_RELAXED), __atomic_fetch_add(&RSP.run_n, 1, __ATOMIC_RELAXED);
    uint64_t m = __atomic_load_n(&RSP.run_max, __ATOMIC_RELAXED);
    while (ns > m && !__atomic_compare_exchange_n(&RSP.run_max, &m, ns, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}

static void rsp_worker_main(void *arg) {
    (void)arg;
    rsp_worker = 1;
    vk_mem_worker = 1;
    vk_exec_set_worker(1);
    plat_mutex_lock(&RSP.m);
    for (;;) {
        while (!RSP.qh && !RSP.stop) plat_cond_wait(&RSP.work_cv, &RSP.m);
        if (RSP.stop) break;
        RspChunk *c = RSP.qh;
        RSP.qh = c->qnext;
        if (!RSP.qh) RSP.qt = NULL;
        plat_mutex_unlock(&RSP.m);
        rsp_run(c);
        plat_mutex_lock(&RSP.m);
        if (!--RSP.pending) plat_cond_broadcast(&RSP.idle_cv);
    }
    plat_mutex_unlock(&RSP.m);
}

// The walk thread closes the open chunk: its place goes into the recorder's ring, the chunk to the workers.
static void rsp_flush(VkExec *ex) {
    RspChunk *c = RSP.cur;
    if (!c) return;
    RSP.cur = NULL;
    c->rec = rec_chunk_alloc();
    c->lnext = RSP.all, RSP.all = c;
    c->qnext = NULL;
    plat_mutex_lock(&RSP.m);
    if (RSP.qt) RSP.qt->qnext = c; else RSP.qh = c;
    RSP.qt = c;
    RSP.pending++;
    plat_cond_signal(&RSP.work_cv);
    plat_mutex_unlock(&RSP.m);
    rec_chunk_enqueue(c->rec);
    ex->recorded += (int)(c->ndraws ? c->ndraws : 1);   // what the chunk records, for the barriers that skip when nothing was
    RSP.after_chunk = 1, RSP.chunks++;
}

// Before an op the walk thread runs itself: the chunk before it is closed, and the command buffer state the walk thread kept (B) is
// forgotten, as the chunk's draws changed what the command buffer holds.
static void rsp_serial(VkExec *ex) {
    rsp_flush(ex);
    if (RSP.after_chunk) {
        RSP.after_chunk = 0;
        memset(&B, 0, sizeof B);
        ex->dirty = VK_EXEC_DIRTY_ALL;
    }
    vk_usnap_split_seq(RSP.seq);
}

static void rsp_append(const op_hdr *h) {
    RspChunk *c = RSP.cur;
    if (c->nops == c->cap) {
        c->cap = c->cap ? c->cap * 2 : 4096;
        c->ops = realloc((void *)c->ops, sizeof *c->ops * c->cap);
        if (!c->ops) rt_fatal("vk: out of memory");
    }
    c->ops[c->nops++] = h;
}

static void rsp_join(VkExec *ex) {
    rsp_flush(ex);
    uint64_t t0 = plat_now_ns();
    plat_mutex_lock(&RSP.m);
    while (RSP.pending) plat_cond_wait(&RSP.idle_cv, &RSP.m);
    plat_mutex_unlock(&RSP.m);
    for (RspChunk *c = RSP.all, *next; c; c = next) next = c->lnext, c->lnext = RSP.freel, RSP.freel = c;
    RSP.all = NULL;
    RSP.join_ns += plat_now_ns() - t0, RSP.join_n++;
}

// 1: the draw goes into a chunk; 0: the walk thread runs it itself (reason counted).
static int rsp_draw(VkExec *ex, const op_hdr *h, const void *a) {
    int why = 0;
    if (ex->render_disabled) why = 1;
    else if (ex->pass_noatt) why = 2;
    else if (vk_exec_oq_counting()) why = 3;
    else if (h->op == OP_DRAW) {
        rsp_reset_probe = 1;
        int m = indirect_reset_draw(ex, a);
        rsp_reset_probe = 0;
        if (m) why = 4;
    }
    if (!why && !RSP.cur) {
        if (!ex->in_pass) {   // what prepare_draw does for the first draw of a pass
            cb_res(ex);
            dummies_init();
            ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
            vk_exec_pass_begin(ex);
            if (ex->pass_noatt) why = 2;
        }
    }
    if (why) {
        rsp_serial(ex);
        RSP.serial_draws[why]++;
        RSP.seq++;
        vk_usnap_split_seq(RSP.seq);
        return 0;
    }
    if (!RSP.cur) {
        RspChunk *c = RSP.freel;
        if (c) RSP.freel = c->lnext;
        else if (!(c = calloc(1, sizeof *c))) rt_fatal("vk: out of memory");
        c->nops = c->ndraws = 0;
        c->start = *ex;
        if (ex->pass_serial == RSP.pass_serial) {   // draws of this pass came before: assume they wrote every attachment
            uint32_t m = ex->depth.image ? 0x100u : 0;
            for (uint32_t i = 0; i < ex->pass_ncolor && i < 8; i++) m |= ex->color[i].image ? 1u << i : 0;
            c->start.fb_dirty = m;
        }
        RSP.pass_serial = ex->pass_serial;
        c->seq0 = RSP.seq, c->bind_gen0 = bind_gen;
        c->upd_n = UPD.fence == ex->fence ? UPD.n : 0;
        RSP.cur = c;
    }
    rsp_append(h);
    RSP.cur->ndraws++, RSP.seq++, RSP.draws++;
    vk_exec_frame.draws_seen++;
    if (RSP.cur->ndraws >= RSP.chunk_draws) rsp_flush(ex);
    return 1;
}

int vk_exec_split_op(VkExec *ex, const op_hdr *h, const void *a) {
    uint32_t op = h->op;
    switch (op) {
    case OP_END: case OP_BEGIN: case OP_JUMP: case OP_CALL: case OP_WAIT_SYNC: return 0;
    case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF: case OP_BIND_PROGRAM: case OP_BIND_UBO:
    case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_PUSH_CONST:
        vk_exec_split_bind(ex, op, a);
        if (RSP.cur) rsp_append(h);
        RSP.bind_ops++;
        return 1;
    case OP_DRAW_IDX_IND: if (!(RSP.ext & 1)) goto serial_op; return rsp_draw(ex, h, a);   // TOTK_REPLAY_SPLIT_EXT bit 0
    case OP_DRAW: case OP_DRAW_IDX: return rsp_draw(ex, h, a);
    default:
    serial_op:
        if (vk_exec_state_op(ex, op, a)) {
            if (RSP.cur) rsp_append(h);
            RSP.state_ops++;
            return 1;
        }
        if (RSP.cur) RSP.cut[op]++;
        rsp_serial(ex);
        if (op == OP_DRAW_IDX_IND || op == OP_DISPATCH || op == OP_DISPATCH_IND) RSP.seq++, vk_usnap_split_seq(RSP.seq);
        return 0;
    }
}

// Walk-thread time per op, split into what went to a chunk (0) and what ran on the walk thread (1).
static uint64_t rsp_opns[2][OP_COUNT_], rsp_opn[2][OP_COUNT_];
void vk_exec_split_time(uint32_t op, uint64_t ns, int serial) {
    if (op < OP_COUNT_) rsp_opns[serial][op] += ns, rsp_opn[serial][op]++;
}
static void rsp_time_log(double nf) {
    char buf[900];
    int n = 0;
    for (int s = 0; s < 2; s++) {
        uint64_t tot = 0;
        for (uint32_t o = 0; o < OP_COUNT_; o++) tot += rsp_opns[s][o];
        n += snprintf(buf + n, sizeof buf - (size_t)n, "%s %.2f ms [", s ? "serial" : "chunked", (double)tot / nf / 1e6);
        for (int k = 0; k < 6; k++) {
            uint32_t bo = 0;
            for (uint32_t o = 0; o < OP_COUNT_; o++) if (rsp_opns[s][o] > rsp_opns[s][bo]) bo = o;
            if (!rsp_opns[s][bo]) break;
            n += snprintf(buf + n, sizeof buf - (size_t)n, "%s %.2f ms/%.0f ", nvn_op_name(bo), (double)rsp_opns[s][bo] / nf / 1e6, (double)rsp_opn[s][bo] / nf);
            rsp_opns[s][bo] = 0;   // taken
        }
        n += snprintf(buf + n, sizeof buf - (size_t)n, "] ");
    }
    rt_log("perf: split walk-thread time per frame: %s", buf);
    memset(rsp_opns, 0, sizeof rsp_opns), memset(rsp_opn, 0, sizeof rsp_opn);
}

// TOTK_TRACE=perf, every 60 frames (vk_exec.c resub_log).
void vk_exec_split_log(uint64_t frame, double nf) {
    static uint64_t last_run_n, last_run_ns, last_join_ns, last_subs, last_draws, last_chunks, last_sd, last_lw2;
    if (RSP.state <= 0) return;
    rsp_time_log(nf);
    uint64_t rn = RSP.run_n - last_run_n, rns = RSP.run_ns - last_run_ns, jn = RSP.join_ns - last_join_ns, subs = RSP.subs - last_subs;
    uint64_t draws = RSP.draws - last_draws, chunks = RSP.chunks - last_chunks;
    uint64_t sd = RSP.serial_draws[1] + RSP.serial_draws[2] + RSP.serial_draws[3] + RSP.serial_draws[4];
    uint64_t sdn = sd - last_sd;
    last_run_n = RSP.run_n, last_run_ns = RSP.run_ns, last_join_ns = RSP.join_ns, last_subs = RSP.subs;
    last_draws = RSP.draws, last_chunks = RSP.chunks, last_sd = sd;
    uint64_t top[3] = {0}, topop[3] = {0};
    for (uint32_t o = 0; o < OP_COUNT_; o++)
        for (int k = 0; k < 3; k++)
            if (RSP.cut[o] > top[k]) {
                for (int j = 2; j > k; j--) top[j] = top[j - 1], topop[j] = topop[j - 1];
                top[k] = RSP.cut[o], topop[k] = o;
                break;
            }
    rt_log("perf: split frame %llu: per frame: split submissions %.1f, chunks %.1f (%.0f draws, %.0f per chunk), run %.2f ms per chunk (max %.2f), "
           "worker time %.2f ms, join wait %.2f ms; serial draws %.1f (since start: disabled %llu, no-attachment %llu, counting %llu, reset %llu); "
           "chunk cuts since start by op %s %llu, %s %llu, %s %llu; feedback events %llu, refused alias copies %llu; split-lock wait %.2f ms",
           (unsigned long long)frame, (double)subs / nf, (double)chunks / nf, (double)draws / nf, chunks ? (double)draws / (double)chunks : 0.0,
           rn ? (double)rns / (double)rn / 1e6 : 0.0, (double)RSP.run_max / 1e6, (double)rns / nf / 1e6, (double)jn / nf / 1e6, (double)sdn / nf,
           (unsigned long long)RSP.serial_draws[1], (unsigned long long)RSP.serial_draws[2], (unsigned long long)RSP.serial_draws[3], (unsigned long long)RSP.serial_draws[4],
           top[0] ? nvn_op_name((uint32_t)topop[0]) : "-", (unsigned long long)top[0], top[1] ? nvn_op_name((uint32_t)topop[1]) : "-", (unsigned long long)top[1],
           top[2] ? nvn_op_name((uint32_t)topop[2]) : "-", (unsigned long long)top[2], (unsigned long long)RSP.n[RSPN_FB_EVENT], (unsigned long long)RSP.n[RSPN_ALIAS],
           (double)(vk_lock_wait_ns[2] - last_lw2) / nf / 1e6);
    last_lw2 = vk_lock_wait_ns[2];
    RSP.run_max = 0;
}
