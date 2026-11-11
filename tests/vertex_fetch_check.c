#include "vk_vertex_fetch.h"
#include <assert.h>
int main(void) {
    uint64_t end = 0, index = 0;
    assert(vk_vertex_fetch_index(1, 0, 0, 4, 5, 3, &index) && index == 7);
    assert(vk_vertex_fetch_index(1, 0, 0, 4, 5, 0, &index) && index == 5);
    assert(!vk_vertex_fetch_index(0, 0, 0, 4, 5, 3, &index));
    assert(vk_vertex_fetch_index(0, 1, 2, 4, -1, 3, &index) && index == 5);
    assert(!vk_vertex_fetch_index(1, 0, 0, 4, -1, 1, &index));
    assert(!vk_vertex_fetch_index(1, 0, 0, 0, 0, 1, &index));
    const uint16_t indices[] = {3, 7, 0xffff, 1};
    assert(vk_vertex_fetch_indices(indices, 4, 2, 1, -1, &index) == 1 && index == 6);
    assert(vk_vertex_fetch_indices(indices, 4, 2, 0, 0, &index) == 1 && index == 0xffff);
    assert(vk_vertex_fetch_indices(indices, 4, 2, 1, -2, &index) == -1);
    assert(vk_vertex_fetch_indices(indices + 2, 1, 2, 1, 0, &index) == 0);
    assert(vk_vertex_fetch_indices(indices, 4, 3, 0, 0, &index) == 0);
    assert(vk_vertex_fetch_format_size(VK_FORMAT_R32G32B32A32_SFLOAT) == 16);
    assert(vk_vertex_fetch_format_size(VK_FORMAT_R8G8B8_UNORM) == 3);
    assert(vk_vertex_fetch_format_size(VK_FORMAT_R16G16B16A16_SFLOAT) == 8);
    assert(vk_vertex_fetch_format_size(VK_FORMAT_A2B10G10R10_UNORM_PACK32) == 4);
    assert(vk_vertex_fetch_format_size(VK_FORMAT_UNDEFINED) == 0);
    assert(vk_vertex_fetch_component_size(VK_FORMAT_R32G32B32A32_SFLOAT) == 4);
    assert(vk_vertex_fetch_component_size(VK_FORMAT_R16G16B16_SFLOAT) == 2);
    assert(vk_vertex_fetch_component_size(VK_FORMAT_R8G8B8A8_UNORM) == 1);
    assert(vk_vertex_fetch_component_size(VK_FORMAT_A2B10G10R10_UNORM_PACK32) == 4);
    assert(vk_vertex_fetch_component_size(VK_FORMAT_UNDEFINED) == 0);
    assert(vk_vertex_fetch_end(3, 16, 4, 16, &end) && end == 68);
    assert(vk_vertex_fetch_end(99, 0, 2, 4, &end) && end == 6);
    assert(!vk_vertex_fetch_end(UINT64_MAX, 2, 0, 4, &end));
    assert(!vk_vertex_fetch_end(UINT64_MAX, 1, 0, 1, &end));
    assert(!vk_vertex_fetch_end(1, 1, 0, 0, &end));
    // DKB_LQ_INTERLEAVE helpers
    VkLqAttr at[3];
    for (unsigned i = 0; i < 3; i++)
        at[i] = (VkLqAttr){i + 1, i, 0, 16, 1, VK_FORMAT_R32G32B32A32_SFLOAT};
    const char *why = NULL;
    assert(vk_lq_interleave_check(at, 3, &why) && !why);
    VkLqAttr bad = at[1];
    bad.stride = 32;
    at[1] = bad;
    assert(!vk_lq_interleave_check(at, 3, &why) && why);
    at[1] = (VkLqAttr){2, 1, 0, 16, 1, VK_FORMAT_R32G32B32A32_SFLOAT};
    at[2].stream = 0;
    assert(!vk_lq_interleave_check(at, 3, &why));
    at[2].stream = 2, at[2].divisor = 0;
    assert(!vk_lq_interleave_check(at, 3, &why));
    at[2].divisor = 1, at[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    assert(!vk_lq_interleave_check(at, 3, &why));
    at[2].format = VK_FORMAT_R32G32B32A32_SFLOAT, at[2].offset = 16;
    assert(!vk_lq_interleave_check(at, 3, &why));
    assert(!vk_lq_interleave_check(at, 1, &why));
    uint64_t bytes = 0;
    assert(vk_lq_interleave_bytes(2, 3, 9, &bytes) && bytes == 5 * 16 * 9);
    assert(!vk_lq_interleave_bytes(0, 0, 9, &bytes));
    assert(!vk_lq_interleave_bytes(VK_LQ_MAX_ELEMENTS, 1, 9, &bytes));
    assert(!vk_lq_interleave_bytes(UINT32_MAX, UINT32_MAX, 9, &bytes));
    // streams of 5 elements; elements 1..3 are copied (first 1, count 3) into a 4-element buffer of 3 streams
    uint8_t s0[5 * 16], s1[5 * 16], s2[5 * 16], dst[4 * 48];
    for (unsigned i = 0; i < 5 * 16; i++) s0[i] = (uint8_t)(i + 1), s1[i] = (uint8_t)(i + 101), s2[i] = (uint8_t)(i + 201);
    memset(dst, 0xee, sizeof dst);
    const uint8_t *src[3] = {s0, s1, s2};
    vk_lq_interleave_copy(dst, src, 3, 1, 3);
    for (unsigned i = 0; i < 16 * 3; i++) assert(dst[i] == 0xee);   // element 0 is left unwritten
    for (unsigned e = 1; e < 4; e++)
        for (unsigned k = 0; k < 3; k++)
            assert(!memcmp(dst + (e * 3 + k) * 16, src[k] + e * 16, 16));
    return 0;
}
