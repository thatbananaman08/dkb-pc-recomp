// Game-free exercise of the real event writer and platform append implementation.
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "event_log.h"
#include "vk.h"

typedef struct GfxShader GfxShader;
static volatile int64_t n_failures;
void rt_log(const char *fmt, ...) { (void)fmt; }
void rt_log_raw(const char *data, size_t n) { (void)fwrite(data, 1, n, stderr); }
void rt_vlogf(const char *fmt, va_list args) { (void)fmt; (void)args; }
#include "event_failure.inc"  // the actual fail() function extracted from runtime/gfx_shader.c
#include "event_fatal.inc"    // the actual rt_fatal() function extracted from runtime/dispatch.c

static int crumbs;
VkCtx *vk_ctx(void) { return NULL; }
void vk_crumbs_report(void) { crumbs++; }
void vmaGetHeapBudgets(VmaAllocator allocator, VmaBudget *budgets) { (void)allocator; (void)budgets; abort(); }
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice p, VkPhysicalDeviceMemoryProperties *m) {
    (void)p; (void)m; abort();
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice d, const char *name) {
    (void)d; (void)name; abort();
}
#include "event_device.inc"   // the actual fault-report entry point, with a null device and no driver calls

static void worker(void *arg) {
    uintptr_t id = (uintptr_t)arg;
    for (unsigned i = 0; i < 200; i++) {
        char reason[48];
        snprintf(reason, sizeof reason, "%llu:%u", (unsigned long long)id, i);
        rt_event_shader_failure(reason, "synthetic");
    }
}

static void lost_worker(void *arg) {
    (void)arg;
    for (int i = 0; i < 50; i++) vk_device_fault_report(VK_ERROR_DEVICE_LOST);
}

static void crash(const PlatCrash *c) {
    rt_event_crash(c);
    plat_exit(71);
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    // No output before opening and no second open within a process.
    rt_event_shader_failure("before open", NULL);
    if (!rt_event_open(argv[1])) return 3;
    if (rt_event_open(argv[1])) return 4;
    if (!strcmp(argv[2], "fatal")) {
        rt_fatal("fixture fatal %s %d", "value", 42);
    }
    if (!strcmp(argv[2], "device")) {
        vk_device_fault_report(VK_SUCCESS);
        PlatThread *threads[8];
        for (int i = 0; i < 8; i++) {
            threads[i] = plat_thread_new(lost_worker, NULL, NULL);
            if (!threads[i]) return 10;
        }
        for (int i = 0; i < 8; i++) plat_thread_join(threads[i]);
        return crumbs == 1 ? 0 : 9;
    }
    if (!strcmp(argv[2], "crash")) {
        plat_crash_hook(crash);
        *(volatile int *)(uintptr_t)0 = 1;
        return 5;
    }
    if (!strcmp(argv[2], "abort")) {
        plat_crash_hook(crash);
        abort();
    }
    if (!strcmp(argv[2], "threads")) {
        PlatThread *threads[8];
        for (uintptr_t i = 0; i < 8; i++) {
            threads[i] = plat_thread_new(worker, (void *)i, NULL);
            if (!threads[i]) return 6;
        }
        for (int i = 0; i < 8; i++) plat_thread_join(threads[i]);
        return 0;
    }
    if (strcmp(argv[2], "events")) return 7;
    if (fail("quote \" slash \\ newline\n utf8 \xc3\xa9 byte \xff", "abc123") || n_failures != 1) return 8;
    rt_event_shader_failure(NULL, NULL);
    rt_event_device_lost(-4);
    PlatCrash sample = {.what = "fault", .code = 0xc0000005u, .rip = 0x1234, .offset = 0x34,
                        .access = 1, .addr = 0x28, .module = "test.dll", .dump = "local.dmp"};
    rt_event_crash(&sample);
    char long_message[500];
    memset(long_message, 'x', sizeof long_message - 1);
    long_message[sizeof long_message - 1] = 0;
    rt_event_fatal(long_message);
    memset(long_message, 1, sizeof long_message - 1);
    rt_event_shader_failure(long_message, long_message);
    return 0;
}
