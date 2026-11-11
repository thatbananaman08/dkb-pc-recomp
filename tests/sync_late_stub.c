// The Vulkan back-end and platform calls that the TOTK_SYNC_EARLY / TOTK_SYNC_LATE_SMALL paths of
// runtime/nvn_api_core.c (nvnSyncWait) reference, as stand-ins for CPU tests that link the NVN front end without the
// back end. Both switches are off in those tests, so these are never reached on a normal path.
#include <stddef.h>
#include <stdint.h>

uint64_t vk_sync_issued(void) { return 0; }
int vk_sync_fence_wait(uint64_t value, uint64_t timeout_ns) { (void)value, (void)timeout_ns; return 1; }
int vk_texture_small_dirty_class(uint64_t fence, uint64_t tiny) { (void)fence, (void)tiny; return 2; }
void vk_texture_write_back_small(uint64_t fence) { (void)fence; }
int plat_thread_spawn(void *fn, void *arg, const char *name, size_t stack) { (void)fn, (void)arg, (void)name, (void)stack; return -1; }
