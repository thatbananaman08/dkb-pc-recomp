"""Bananza's one-shot zstd decoder on libzstd (runtime/ov_zstd.c native_oneshot): the production source against host stubs.

Compiles ov_zstd.c with a stub patches.h, runs native_oneshot on frames libzstd makes (one frame; a skippable frame, then a
frame; two frames) and checks the bytes written and the return value, then the calls that must stay with the guest decoder
(flag set, unaligned or missing context memory, a corrupt frame, a frame that does not fit, an in-place call), and TOTK_ZSTD_CHECK
(both ways, the guest's result stands). The guest decoder is a stub that returns a marker. Skips without libzstd or off Windows.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <zstd.h>
#include "ov_zstd.c"

uint64_t rt_guest_start_ns;
void rt_log(const char *fmt, ...) { va_list a; va_start(a, fmt); fputs("[log] ", stdout); vprintf(fmt, a); fputs("\n", stdout); va_end(a); }
void rt_fatal(const char *fmt, ...) { (void)fmt; abort(); }
static const char *env_check, *env_zstd;
const char *cfg_get(const char *n) { return !strcmp(n, "TOTK_ZSTD_CHECK") ? env_check : !strcmp(n, "TOTK_ZSTD") ? env_zstd : NULL; }
int cfg_bool(const char *n) { const char *v = cfg_get(n); return v && strcmp(v, "0"); }
void rt_register_host(uint64_t a, void (*f)(Cpu *)) { (void)a; (void)f; }
uint64_t kern_now_ns(void) { return 1000; }
int kern_core_away(void) { return 0; }
void kern_core_back(int *a) { (void)a; }
int kern_host_track;
void kern_host_left(void *t) { (void)t; }
int plat_mem_readable(uint64_t a, uint64_t n) { (void)a; (void)n; return 1; }
void plat_mutex_lock(PlatMutex *m) { (void)m; }
void plat_mutex_unlock(PlatMutex *m) { (void)m; }
void plat_mutex_lock_shared(PlatMutex *m) { (void)m; }
void plat_mutex_unlock_shared(PlatMutex *m) { (void)m; }
// The "guest decoder": returns a marker and writes a marker byte, so a test can tell whether it ran.
uint64_t rt_call(Cpu *c, uint64_t fn, int n, ...) {
    (void)fn; (void)n;
    ((uint8_t *)(uintptr_t)c->x[0])[0] = 0xEE;
    c->x[0] = 0x1234;
    return 0x1234;
}

static int fails;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } } while (0)

static uint8_t *arena;   // guest addresses are host addresses below 2^39
static uint64_t bump;
static uint8_t *take(size_t n) { uint8_t *p = arena + bump; bump += (n + 255) & ~(size_t)255; return p; }

static int run(Cpu *c, uint8_t *dst, uint32_t cap, const uint8_t *src, uint32_t size, uint64_t ws, uint64_t flag) {
    memset(c, 0, sizeof *c);
    c->x[0] = (uint64_t)(uintptr_t)dst, c->x[1] = cap, c->x[2] = (uint64_t)(uintptr_t)src, c->x[3] = size, c->x[4] = ws, c->x[5] = flag;
    return native_oneshot(c);
}

int main(int argc, char **argv) {
    int check = argc > 1 && !strcmp(argv[1], "check");
    if (check) env_check = "1";
    arena = VirtualAlloc((void *)0x2000000000ull, 64u << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!arena) { printf("SKIP: cannot map the arena\n"); return 77; }
    static Cpu cpu;
    Cpu *c = &cpu;
    enum { N = 300000 };
    uint8_t *plain = take(N), *plain2 = take(N);
    uint32_t s = 12345;
    for (int i = 0; i < N; i++) { s = s * 1664525u + 1013904223u; plain[i] = (uint8_t)((s >> 24) & 0x0f) + (i % 7 == 0 ? 3 : 0); plain2[i] = (uint8_t)(i * 31); }
    size_t bound = ZSTD_compressBound(N);
    uint8_t *f1 = take(bound), *f2 = take(bound), *all = take(2 * bound + 64), *dst = take(2 * N), *ctx = take(0x8000);
    size_t n1 = ZSTD_compress(f1, bound, plain, N, 3), n2 = ZSTD_compress(f2, bound, plain2, N, 5);
    CHECK(!ZSTD_isError(n1) && !ZSTD_isError(n2), "libzstd makes the test frames");
    ctx = (uint8_t *)(((uintptr_t)ctx + 7) & ~7ull);

    if (check) {   // TOTK_ZSTD_CHECK: both ways, the guest's result stands (the stub differs from libzstd, so a mismatch is reported)
        memset(dst, 0, 2 * N);
        CHECK(run(c, dst, N, f1, (uint32_t)n1, (uint64_t)(uintptr_t)ctx, 0) == 1, "check mode answers the call");
        CHECK(c->x[0] == 0x1234 && dst[0] == 0xEE, "check mode keeps the guest's result and output");
        printf("fails %d\n", fails);
        return fails ? 1 : 0;
    }
    // one frame
    memset(dst, 0, 2 * N);
    CHECK(run(c, dst, N, f1, (uint32_t)n1, (uint64_t)(uintptr_t)ctx, 0) == 1, "one frame is taken");
    CHECK(c->x[0] == N && !memcmp(dst, plain, N), "one frame: size and bytes");
    // a skippable frame, then a frame, then another frame
    uint32_t skip[3] = {0x184D2A53u, 4, 0xDEADBEEFu};
    size_t o = 0;
    memcpy(all + o, skip, 12), o += 12;
    memcpy(all + o, f1, n1), o += n1;
    memcpy(all + o, f2, n2), o += n2;
    memset(dst, 0, 2 * N);
    CHECK(run(c, dst, 2 * N, all, (uint32_t)o, (uint64_t)(uintptr_t)ctx, 0) == 1, "skippable + two frames are taken");
    CHECK(c->x[0] == 2 * (uint64_t)N && !memcmp(dst, plain, N) && !memcmp(dst + N, plain2, N), "multi-frame: size and bytes");
    // stays with the guest decoder: flag, unaligned or no context memory, no room, a corrupt frame, in place
    CHECK(run(c, dst, N, f1, (uint32_t)n1, (uint64_t)(uintptr_t)ctx, 1) == 0, "flag set: guest");
    CHECK(run(c, dst, N, f1, (uint32_t)n1, (uint64_t)(uintptr_t)ctx + 4, 0) == 0, "unaligned context memory: guest");
    CHECK(run(c, dst, N, f1, (uint32_t)n1, 0, 0) == 0, "no context memory: guest");
    CHECK(run(c, dst, N - 1, f1, (uint32_t)n1, (uint64_t)(uintptr_t)ctx, 0) == 0, "capacity too small: guest");
    uint8_t *bad = take(n1);
    memcpy(bad, f1, n1), bad[n1 / 2] ^= 0x55, bad[n1 / 2 + 1] ^= 0xAA, bad[n1 - 1] ^= 0xFF;
    CHECK(run(c, dst, N, bad, (uint32_t)n1, (uint64_t)(uintptr_t)ctx, 0) == 0, "corrupt frame: guest");
    CHECK(run(c, f1 + 100, (uint32_t)n1, f1, (uint32_t)n1, (uint64_t)(uintptr_t)ctx, 0) == 0, "in place: guest");
    CHECK(run(c, dst, 0, f1, (uint32_t)n1, (uint64_t)(uintptr_t)ctx, 0) == 0, "zero capacity: guest");
    printf("fails %d\n", fails);
    return fails ? 1 : 0;
}
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    args = ap.parse_args()
    if os.name != 'nt' or not args.cc:
        print('SKIP: needs Windows and a C compiler')
        return 77
    out = ROOT / 'build' / 'test' / 'zstd-oneshot'
    out.mkdir(parents=True, exist_ok=True)
    stub = out / 'stub'
    (stub / 'platform').mkdir(parents=True, exist_ok=True)
    (stub / 'patches.h').write_text('#pragma once\n#define RT_PATCH_COUNT 1\n#define RT_PATCH_PAGE 0x7002450000ull\n'
                                    '#define RT_PATCH_SETS(X) X(0, zstdprobe, "DKB_ZSTD_PROBE", 0)\n')
    src = out / 'oneshot.c'
    src.write_text(HARNESS.replace('    check_mode_reset:;\n    {\n        extern int check_mode_forced;\n    }\n', ''))
    exe = out / 'oneshot.exe'
    inc = ['-I', str(stub), '-I', str(ROOT / 'runtime')]
    p = subprocess.run([args.cc, '-std=gnu11', '-O1', '-Wall', '-Wno-unused-function', '-Wno-unused-variable', *inc, str(src), '-o', str(exe), '-lzstd'],
                       capture_output=True, text=True)
    if p.returncode:
        if 'zstd.h' in p.stderr or 'lzstd' in p.stderr:
            print('SKIP: libzstd is not available')
            return 77
        print(p.stdout, p.stderr)
        return 1
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    print(run.stdout, run.stderr)
    if run.returncode:
        return run.returncode
    run = subprocess.run([str(exe), 'check'], capture_output=True, text=True, timeout=120)
    print(run.stdout, run.stderr)
    if run.returncode == 0 and 'MISMATCH one-shot' not in run.stdout:
        print('FAIL: check mode did not report the difference')
        return 1
    return run.returncode


if __name__ == '__main__':
    sys.exit(main())
