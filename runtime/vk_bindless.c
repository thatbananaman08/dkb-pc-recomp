// The global sampled-texture table: see vk.h (VKB_SET) and bindless_table.h.
//
// Descriptor indexing gives a descriptor set whose bindings are arrays of combined image samplers, written while the set is bound
// and in use (UPDATE_AFTER_BIND, UPDATE_UNUSED_WHILE_PENDING) and read only where the shader indexes (PARTIALLY_BOUND). A slot is
// written when its (view, sampler, layout) first appears and again with the stand-in descriptor when its view is destroyed; the
// views are destroyed after the last submission that used them (vk_memory.c collect_locked), so no pending draw reads a slot that is
// rewritten. The sampler pool is never destroyed during a run (vk_texture.c), so sampler slots stay valid.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bindless_table.h"
#include "cfg.h"
#include "host.h"
#include "mtstat.h"
#include "platform/platform.h"
#include "vk.h"

static struct {
    PlatMutex lock;
    int mode;                         // -1 not read yet
    int ready;                        // table made
    VkDescriptorSetLayout layout;
    VkDescriptorPool pool;
    VkDescriptorSet set;
    BtTable t;
    uint8_t dummy_done[BT_CLASSES];
    VkSampler dummy_samp[BT_CLASSES];
    VkImageView dummy_view[BT_CLASSES];
    uint64_t verified, mismatches;
    uint32_t logged_full;
} V = {.lock = PLAT_MUTEX_INIT, .mode = -1};

// Slots per class (including the dummy slot 0). The 2D class is the common one; the shadow, array and cube classes are few.
static uint32_t class_cap(int c) {
    if (c == 1) return 16384;                                  // 2D
    if (c == 5 || c == 17 || c == 21) return 4096;             // 2D array, 2D shadow, 2D array shadow
    if (c == 0 || c == 2 || c == 3 || c == 4 || c == 7 || c == 9 || c == 13) return 2048;   // 1D, 3D, cube, 1D array, cube array, 2D multisample, 2D multisample array
    return 256;
}

// Descriptor indexing with update-after-bind for the whole table, push constants for the slots of both stages, a free descriptor set index
// and the stage-independent layouts the pipelines use on this path (vk_caps.gpl: vertex set 0, fragment set 1). Sets *why on a refusal.
static int supported_why(VkCtx *c, const char **why) {
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(c->phys, &f2);
    VkPhysicalDeviceVulkan12Properties p12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p12};
    vkGetPhysicalDeviceProperties2(c->phys, &p2);
    uint64_t total = 0;
    for (int i = 0; i < BT_CLASSES; i++) total += class_cap(i);
    *why = NULL;
    if (!f12.descriptorIndexing || !f12.runtimeDescriptorArray) *why = "descriptor indexing or runtimeDescriptorArray missing";
    else if (!f12.descriptorBindingSampledImageUpdateAfterBind || !f12.descriptorBindingPartiallyBound ||
             !f12.descriptorBindingUpdateUnusedWhilePending) *why = "update-after-bind, partially bound or update-unused-while-pending missing";
    else if (c->props.limits.maxBoundDescriptorSets <= VKB_SET) *why = "too few bound descriptor sets";
    else if (c->props.limits.maxPushConstantsSize < GFX_BINDLESS_PUSH_FS + 4 * GFX_BINDLESS_SLOTS_FS) *why = "maxPushConstantsSize below the slot ranges";
    else if (p12.maxDescriptorSetUpdateAfterBindSampledImages < total || p12.maxDescriptorSetUpdateAfterBindSamplers < total ||
             p12.maxPerStageDescriptorUpdateAfterBindSampledImages < total || p12.maxPerStageDescriptorUpdateAfterBindSamplers < total)
        *why = "the update-after-bind limits are below the table size";
    else if (!vk_caps.gpl) *why = "stage-independent layouts (graphics pipeline library) are not in use";
    return *why == NULL;
}

int vkb_supported(VkCtx *c) {
    static int cached = -1;
    if (cached < 0) {
        const char *why;
        cached = supported_why(c, &why);
        const char *e = cfg_get("TOTK_BINDLESS_TEX");
        if (!cached && e && *e && strcmp(e, "0")) rt_log("bindless: TOTK_BINDLESS_TEX is off: %s", why);
    }
    return cached;
}

int vkb_mode(void) {
    if (V.mode >= 0) return V.mode;
    plat_mutex_lock(&V.lock);
    if (V.mode < 0) {
        const char *e = cfg_get("TOTK_BINDLESS_TEX");
        int m = !e || !*e || !strcmp(e, "0") ? 0 : !strcmp(e, "check") ? 2 : 1;
        VkCtx *c = vk_ctx();
        if (m && !c) {   // asked before the device exists: the first draw asks again
            plat_mutex_unlock(&V.lock);
            return 0;
        }
        if (m == 1 && !gfx_shader_bindless()) m = 0;   // the shaders are not bindless ones (no translator, or the device said no)
        if (m && !vkb_supported(c)) m = 0;
        if (m) {
            uint32_t cap[BT_CLASSES];
            uint64_t total = 0;
            for (int i = 0; i < BT_CLASSES; i++) total += cap[i] = class_cap(i);
            const char *why = NULL;
            VkDescriptorBindingFlags fl = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                          VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
            VkDescriptorSetLayoutBinding b[BT_CLASSES];
            VkDescriptorBindingFlags bf[BT_CLASSES];
            for (int i = 0; i < BT_CLASSES; i++) {
                b[i] = (VkDescriptorSetLayoutBinding){(uint32_t)i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, cap[i],
                                                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                                                      VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_COMPUTE_BIT, NULL};
                bf[i] = fl;
            }
            VkDescriptorSetLayoutBindingFlagsCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                              .bindingCount = BT_CLASSES, .pBindingFlags = bf};
            VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = &fi,
                                                  .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                                  .bindingCount = BT_CLASSES, .pBindings = b};
            VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, (uint32_t)total};
            VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                             .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1,
                                             .poolSizeCount = 1, .pPoolSizes = &ps};
            VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            if (vkCreateDescriptorSetLayout(c->device, &li, NULL, &V.layout) != VK_SUCCESS) why = "the set layout of the table could not be made";
            else if (vkCreateDescriptorPool(c->device, &pi, NULL, &V.pool) != VK_SUCCESS) why = "the descriptor pool of the table could not be made";
            else {
                ai.descriptorPool = V.pool, ai.descriptorSetCount = 1, ai.pSetLayouts = &V.layout;
                if (vkAllocateDescriptorSets(c->device, &ai, &V.set) != VK_SUCCESS) why = "the descriptor set of the table could not be allocated";
            }
            if (!why && !bt_init(&V.t, cap)) why = "out of memory";
            if (why) {
                rt_log("bindless: TOTK_BINDLESS_TEX is off: %s", why);
                m = 0;
            } else {
                V.ready = 1;
                rt_log("bindless: global texture table on (%s): %llu combined image samplers in %d classes, set %d",
                       m == 2 ? "check" : "TOTK_BINDLESS_TEX=1", (unsigned long long)total, BT_CLASSES, VKB_SET);
            }
        }
        V.mode = m;
    }
    plat_mutex_unlock(&V.lock);
    return V.mode;
}

VkDescriptorSetLayout vkb_layout(void) { return V.ready ? V.layout : VK_NULL_HANDLE; }
VkDescriptorSet vkb_set(void) { return V.ready ? V.set : VK_NULL_HANDLE; }

static void write_slot(uint32_t cls, uint32_t slot, VkSampler samp, VkImageView view, VkImageLayout layout) {
    static int dbg = -1;
    if (dbg < 0) dbg = getenv("DKB_BL_DEBUG") && (atoi(getenv("DKB_BL_DEBUG")) & 4);
    if (dbg) rt_log("bindless: write class %u slot %u view %#llx sampler %#llx layout %d", cls, slot, (unsigned long long)(uintptr_t)view,
                    (unsigned long long)(uintptr_t)samp, (int)layout);
    VkDescriptorImageInfo ii = {samp, view, layout};
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = V.set, .dstBinding = cls, .dstArrayElement = slot,
                              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    vkUpdateDescriptorSets(vk_ctx()->device, 1, &w, 0, NULL);
}

static void write_dummy(uint32_t cls, uint32_t slot) {
    if (V.dummy_done[cls]) write_slot(cls, slot, V.dummy_samp[cls], V.dummy_view[cls], VK_IMAGE_LAYOUT_GENERAL);
}

// A direct-mapped cache per thread in front of the table: a hit costs a compare, no lock. An entry is valid while no slot has been freed
// since it was made (BtTable.gen, bumped by every release; slots are never moved or reused otherwise).
typedef struct { uint64_t view, samp, gen; uint32_t layout, cls, slot; } SlotCache;
static _Thread_local SlotCache slot_cache[256];

uint32_t vkb_slot(uint32_t cls, VkSampler samp, VkImageView view, VkImageLayout layout, VkSampler dummy_samp, VkImageView dummy_view) {
    if (!V.ready || cls >= BT_CLASSES) return 0;
    uint64_t hv = (uint64_t)(uintptr_t)view, hs = (uint64_t)(uintptr_t)samp;
    SlotCache *c = &slot_cache[(hv * 0x9E3779B97F4A7C15ull ^ hs * 0xC2B2AE3D27D4EB4Full ^ (uint64_t)layout << 5 ^ cls) >> 56];
    uint64_t gen = __atomic_load_n(&V.t.gen, __ATOMIC_ACQUIRE);
    if (c->slot && c->gen == gen && c->view == hv && c->samp == hs && c->layout == (uint32_t)layout && c->cls == cls) return c->slot;
    plat_mutex_lock(&V.lock);
    if (!V.dummy_done[cls] && V.t.cap[cls]) {
        V.dummy_done[cls] = 1, V.dummy_samp[cls] = dummy_samp, V.dummy_view[cls] = dummy_view;
        write_slot(cls, 0, dummy_samp, dummy_view, VK_IMAGE_LAYOUT_GENERAL);
    }
    BtKey k = {(uint64_t)(uintptr_t)view, (uint64_t)(uintptr_t)samp, (uint32_t)layout};
    int fresh;
    uint32_t s = bt_acquire(&V.t, cls, &k, &fresh);
    if (fresh) write_slot(cls, s, samp, view, layout), MTC(MT_TEX_SLOT_NEW);
    else if (!s && V.logged_full++ < 8)
        rt_log("bindless: class %u of the texture table is full (%u slots): draws use its stand-in", cls, V.t.cap[cls]);
    uint64_t gen_now = V.t.gen;
    plat_mutex_unlock(&V.lock);
    if (s) *c = (SlotCache){hv, hs, gen_now, (uint32_t)layout, cls, s};
    return s;
}

int vkb_verify(uint32_t cls, uint32_t slot, VkSampler samp, VkImageView view, VkImageLayout layout) {
    if (!V.ready) return 1;
    plat_mutex_lock(&V.lock);
    const BtKey *k = bt_key(&V.t, cls, slot);
    int ok = k && k->view == (uint64_t)(uintptr_t)view && k->samp == (uint64_t)(uintptr_t)samp && k->layout == (uint32_t)layout;
    V.verified++;
    if (!ok) {
        V.mismatches++;
        if (V.mismatches <= 20)
            rt_log("bindless check: class %u slot %u holds view %#llx sampler %#llx layout %u, the draw path binds view %#llx sampler %#llx layout %u", cls, slot,
                   k ? (unsigned long long)k->view : 0ull, k ? (unsigned long long)k->samp : 0ull, k ? k->layout : 0u, (unsigned long long)(uintptr_t)view,
                   (unsigned long long)(uintptr_t)samp, (unsigned)layout);
    }
    plat_mutex_unlock(&V.lock);
    return ok;
}

static void rewrite_dummy(void *u, uint32_t cls, uint32_t slot) {
    (void)u;
    write_dummy(cls, slot);
}

void vkb_view_gone(VkImageView view) {
    if (!V.ready) return;
    plat_mutex_lock(&V.lock);
    bt_release_view(&V.t, (uint64_t)(uintptr_t)view, rewrite_dummy, NULL);
    plat_mutex_unlock(&V.lock);
}

void vkb_sampler_gone(VkSampler samp) {
    if (!V.ready) return;
    plat_mutex_lock(&V.lock);
    bt_release_sampler(&V.t, (uint64_t)(uintptr_t)samp, rewrite_dummy, NULL);
    plat_mutex_unlock(&V.lock);
}

void vkb_log(uint64_t frame) {
    if (!V.ready) return;
    plat_mutex_lock(&V.lock);
    char live[256];
    int o = 0;
    live[0] = 0;
    for (uint32_t c = 0; c < BT_CLASSES && o < (int)sizeof live - 24; c++)
        if (bt_live(&V.t, c)) o += snprintf(live + o, sizeof live - (size_t)o, " c%u:%u", c, bt_live(&V.t, c));
    rt_log("bindless %llu: table lookups %llu (hits %llu, new slots %llu, released %llu, class full %llu), live slots%s; check: %llu compared, %llu mismatches",
           (unsigned long long)frame, (unsigned long long)V.t.acquired, (unsigned long long)V.t.hits, (unsigned long long)V.t.fresh,
           (unsigned long long)V.t.released, (unsigned long long)V.t.full, live, (unsigned long long)V.verified, (unsigned long long)V.mismatches);
    plat_mutex_unlock(&V.lock);
}

// ---- bindless buffers
static int buf_requested(void) {
    static int r = -1;
    if (r < 0) {
        const char *b = cfg_get("TOTK_BINDLESS_BUF"), *t = cfg_get("TOTK_BINDLESS_TEX");
        r = b && !strcmp(b, "1") && t && !strcmp(t, "1");
    }
    return r;
}

VkBufferUsageFlags vkb_buf_usage(void) {
    VkCtx *c = vk_ctx();
    return buf_requested() && c && c->has_buffer_device_address ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0;
}

int vkb_buf_mode(void) {
    return buf_requested() && gfx_shader_bindless_buf() && vkb_mode() == 1 && vkb_buf_usage();
}

typedef struct { VkBuffer buffer; VkDeviceAddress address; } AddrCache;
static _Thread_local AddrCache addr_cache[512];

VkDeviceAddress vkb_buffer_address(VkBuffer buffer) {
    AddrCache *e = &addr_cache[((uintptr_t)buffer * 0x9E3779B97F4A7C15ull) >> 55];
    if (e->buffer == buffer) return e->address;
    VkBufferDeviceAddressInfo ai = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buffer};
    VkDeviceAddress a = vkGetBufferDeviceAddress(vk_ctx()->device, &ai);
    *e = (AddrCache){buffer, a};
    return a;
}

uint64_t vkb_gen(void) { return V.ready ? __atomic_load_n(&V.t.gen, __ATOMIC_ACQUIRE) : 0; }
