#ifndef VK_VERTEX_FETCH_H
#define VK_VERTEX_FETCH_H
#include <stdint.h>
#include <string.h>
#include <vulkan/vulkan.h>

// Byte size of the formats emitted by vk_vertex_format; zero means unknown.
static inline unsigned vk_vertex_fetch_format_size(VkFormat f) {
    if (f >= VK_FORMAT_R8_UNORM && f <= VK_FORMAT_R8_SRGB) return 1;
    if (f >= VK_FORMAT_R8G8_UNORM && f <= VK_FORMAT_R8G8_SRGB) return 2;
    if (f >= VK_FORMAT_R8G8B8_UNORM && f <= VK_FORMAT_R8G8B8_SRGB) return 3;
    if (f >= VK_FORMAT_R8G8B8A8_UNORM && f <= VK_FORMAT_B8G8R8A8_SRGB) return 4;
    if (f >= VK_FORMAT_A2R10G10B10_UNORM_PACK32 && f <= VK_FORMAT_A2B10G10R10_SINT_PACK32) return 4;
    if (f == VK_FORMAT_B10G11R11_UFLOAT_PACK32) return 4;
    if (f >= VK_FORMAT_R16_UNORM && f <= VK_FORMAT_R16_SFLOAT) return 2;
    if (f >= VK_FORMAT_R16G16_UNORM && f <= VK_FORMAT_R16G16_SFLOAT) return 4;
    if (f >= VK_FORMAT_R16G16B16_UNORM && f <= VK_FORMAT_R16G16B16_SFLOAT) return 6;
    if (f >= VK_FORMAT_R16G16B16A16_UNORM && f <= VK_FORMAT_R16G16B16A16_SFLOAT) return 8;
    if (f >= VK_FORMAT_R32_UINT && f <= VK_FORMAT_R32_SFLOAT) return 4;
    if (f >= VK_FORMAT_R32G32_UINT && f <= VK_FORMAT_R32G32_SFLOAT) return 8;
    if (f >= VK_FORMAT_R32G32B32_UINT && f <= VK_FORMAT_R32G32B32_SFLOAT) return 12;
    if (f >= VK_FORMAT_R32G32B32A32_UINT && f <= VK_FORMAT_R32G32B32A32_SFLOAT) return 16;
    return 0;
}

// Byte size of one component (the alignment unit of an attribute address); packed formats use their 4-byte word.
static inline unsigned vk_vertex_fetch_component_size(VkFormat f) {
    unsigned n = vk_vertex_fetch_format_size(f);
    if (!n) return 0;
    if ((f >= VK_FORMAT_A2R10G10B10_UNORM_PACK32 && f <= VK_FORMAT_A2B10G10R10_SINT_PACK32) ||
        f == VK_FORMAT_B10G11R11_UFLOAT_PACK32) return 4;
    if (f >= VK_FORMAT_R8_UNORM && f <= VK_FORMAT_B8G8R8A8_SRGB) return 1;
    if (f >= VK_FORMAT_R16_UNORM && f <= VK_FORMAT_R16G16B16A16_SFLOAT) return 2;
    return 4;
}

// Match replay's effective divisor of one. Indexed vertex-rate fetches need the index data.
static inline int vk_vertex_fetch_index(int instance_rate, int direct, int32_t first, int32_t count,
                                        int32_t base_instance, int32_t instances, uint64_t *index) {
    if (count <= 0) return 0;
    if (instance_rate) {
        if (base_instance < 0) return 0;
        *index = (uint64_t)(uint32_t)base_instance + (uint32_t)(instances > 0 ? instances : 1) - 1;
        return 1;
    }
    if (!direct || first < 0) return 0;
    *index = (uint64_t)(uint32_t)first + (uint32_t)count - 1;
    return 1;
}

// Inspect a host snapshot, never asserting that it matches pending GPU writes. Return 1 for a usable range,
// 0 if every index restarts or the format is invalid, -1 for a negative effective vertex index.
static inline int vk_vertex_fetch_indices(const void *data, uint32_t count, unsigned width, int restart,
                                          int32_t base, uint64_t *maximum) {
    if (width != 1 && width != 2 && width != 4) return 0;
    const uint8_t *p = data;
    uint32_t restart_index = width == 1 ? 0xffu : width == 2 ? 0xffffu : UINT32_MAX;
    uint64_t max = 0;
    int any = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t index = 0;
        memcpy(&index, p + (uint64_t)i * width, width);
        if (restart && index == restart_index) continue;
        int64_t effective = (int64_t)index + base;
        if (effective < 0) return -1;
        if (!any || (uint64_t)effective > max) max = (uint64_t)effective;
        any = 1;
    }
    if (any) *maximum = max;
    return any;
}

// Exclusive end relative to the bound range; return zero if it cannot be represented.
static inline int vk_vertex_fetch_end(uint64_t index, uint64_t stride, uint64_t offset, unsigned bytes, uint64_t *end) {
    if (!bytes || (stride && index > (UINT64_MAX - offset) / stride)) return 0;
    uint64_t start = index * stride + offset;
    if (start > UINT64_MAX - bytes) return 0;
    *end = start + bytes;
    return 1;
}

// ---- DKB_LQ_INTERLEAVE (vk_exec_draw.c lq_interleave_*): N per-instance RGBA32F streams, one attribute each, become
// one interleaved stream. Pure helpers, tested by tests/vertex_fetch_check.c.
#define VK_LQ_MAX_STREAMS 16
#define VK_LQ_MAX_ELEMENTS (1u << 14)   // base_instance + instances the copy will cover; larger draws are left alone

typedef struct { uint32_t location, stream, offset, stride, divisor; VkFormat format; } VkLqAttr;

// 1 when every attribute is instance rate with divisor 1, stride 16, offset 0, R32G32B32A32_SFLOAT, each on a stream of
// its own (n in 1..VK_LQ_MAX_STREAMS). Otherwise 0 and *why names the first violated condition.
static inline int vk_lq_interleave_check(const VkLqAttr *a, unsigned n, const char **why) {
    const char *w = NULL;
    if (n < 2 || n > VK_LQ_MAX_STREAMS) w = "needs 2 to 16 attributes";
    for (unsigned i = 0; !w && i < n; i++) {
        if (a[i].format != VK_FORMAT_R32G32B32A32_SFLOAT) w = "an attribute is not R32G32B32A32_SFLOAT";
        else if (a[i].offset) w = "an attribute has a nonzero offset";
        else if (a[i].stride != 16) w = "a stream stride is not 16";
        else if (a[i].divisor != 1) w = "a stream is not instance rate with divisor 1";
        for (unsigned j = 0; !w && j < i; j++)
            if (a[j].stream == a[i].stream) w = "two attributes share a stream";
    }
    if (why) *why = w;
    return !w;
}

// Bytes of the interleaved buffer: elements 0..first+count-1 of n streams, stride 16*n (the first `first` elements stay
// unwritten so firstInstance and gl_InstanceIndex keep their values). 0 when empty, over VK_LQ_MAX_ELEMENTS or overflowing.
static inline int vk_lq_interleave_bytes(uint32_t first, uint32_t count, unsigned n, uint64_t *bytes) {
    uint64_t total = (uint64_t)first + count;
    if (!count || !n || total > VK_LQ_MAX_ELEMENTS) return 0;
    *bytes = total * 16u * n;
    return 1;
}

// dst element i (i in [first, first+count)) = src[0][i], src[1][i], ... (16 bytes each). dst and src[k] start at element 0.
static inline void vk_lq_interleave_copy(uint8_t *dst, const uint8_t *const *src, unsigned n, uint32_t first, uint32_t count) {
    for (uint64_t i = first; i < (uint64_t)first + count; i++)
        for (unsigned k = 0; k < n; k++) memcpy(dst + (i * n + k) * 16u, src[k] + i * 16u, 16u);
}
#endif
