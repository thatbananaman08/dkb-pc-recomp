"""TOTK_HOSTPROF accounting (runtime/hostprof.c): top-N ordering, wait shares, the guest wait table, the slow-frame summary.

Compiles the production hostprof.c with stubs for the logger, clock, config and host function names.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hostprof.c"

static char logbuf[1 << 16];
void rt_log(const char *fmt, ...) { va_list a; va_start(a, fmt); size_t n = strlen(logbuf); vsnprintf(logbuf + n, sizeof logbuf - n, fmt, a); strcat(logbuf, "\n"); va_end(a); }
const char *cfg_get(const char *n) { return !strcmp(n, "TOTK_HOSTPROF") ? "1" : NULL; }
const char *hostfn_name(uint64_t addr) { static char b[8][32]; static int k; k = (k + 1) & 7; snprintf(b[k], sizeof b[k], "host_%llu", (unsigned long long)((addr - RT_HOSTFN_BASE) >> 4)); return b[k]; }
uint64_t kern_now_ns(void) { return 0; }
void plat_mutex_lock(PlatMutex *m) { (void)m; }
void plat_mutex_unlock(PlatMutex *m) { (void)m; }

static int fails;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } } while (0)

int main(void) {
    hostprof_init();
    CHECK(g_hostprof == 1, "TOTK_HOSTPROF=1 turns it on");
    for (int i = 0; i < 12; i++) hp_host_call(i, (uint64_t)(i + 1) * 1000000, i == 11 ? 6000000 : 0);   // slot 11: 12 ms, 6 ms of it waiting
    hp_host_call(11, 1000000, 0);                                                                        // a second call: 13 ms
    hp_svc(0x1A, 40000000, 0, 1, 0x7100001000ull);   // ArbitrateLock on one address, twice
    hp_svc(0x1A, 20000000, 0, 1, 0x7100001000ull);
    hp_svc(0x18, 5000000, 0, 1, 0x1234);
    hp_svc(0x01, 3000000, 1000000, 0, 0);
    hp_scope_add(HPS_ROMFS_READ, 8000000, 2000000);
    hp_wait_add(HPW_FRZ, 4000000);
    hp_wait_add(HPW_CORE, 1000000);
    char s[400];
    hostprof_slow_summary(s, sizeof s);
    CHECK(strstr(s, "svc ArbitrateLock 60.0 ms"), "the frame summary's top entry is the 60 ms of ArbitrateLock");
    CHECK(strstr(s, "host_11 13.0 ms"), "then host_11");
    CHECK(strstr(s, "lock and core waits 5.0 ms"), "waits are summed");
    hostprof_log(60);
    if (getenv("HP_DEBUG")) fputs(logbuf, stdout);
    char *p1 = strstr(logbuf, "#1 svc ArbitrateLock"), *p2 = strstr(logbuf, "#2 host_11"), *p3 = strstr(logbuf, "#3 host_10");
    CHECK(p1 && p2 && p3 && p1 < p2 && p2 < p3, "top ten ordered by total time");
    CHECK(strstr(logbuf, "#2 host_11: 2 calls, 13.0 ms total, max 12.00 ms, waiting 6.0 ms (46%)"), "calls, total, max and wait share");
    CHECK(strstr(logbuf, "romfs_read: 1 calls, 8.0 ms total"), "scopes are listed by name");
    CHECK(strstr(logbuf, "frz_lock (mem_thaw) 4.0 ms in 1 waits"), "lock waits by kind");
    CHECK(strstr(logbuf, "guest waits: 3 waits, 65.0 ms; top: svc ArbitrateLock on 0x7100001000: 2 waits, 60.0 ms (max 40.00)"),
          "guest waits by address");
    CHECK(strstr(logbuf, "svc WaitSynchronization on 0x1234"), "WaitSynchronization is listed by its handle");
    logbuf[0] = 0;
    hostprof_log(120);
    CHECK(!strstr(logbuf, "#1 "), "the window restarts");
    hostprof_frame_reset();
    hostprof_slow_summary(s, sizeof s);
    CHECK(strstr(s, "lock and core waits 0.0 ms") && !strstr(s, "svc"), "the frame restarts");
    printf("fails %d\n", fails);
    return fails ? 1 : 0;
}
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    args = ap.parse_args()
    if not args.cc:
        print('SKIP: no C compiler')
        return 77
    out = ROOT / 'build' / 'test' / 'hostprof'
    out.mkdir(parents=True, exist_ok=True)
    src = out / 'hostprof_test.c'
    src.write_text(HARNESS)
    exe = out / 'hostprof_test.exe'
    p = subprocess.run([args.cc, '-std=gnu11', '-O1', '-Wall', '-Wno-unused-function', '-I', str(ROOT / 'runtime'), str(src), '-o', str(exe)],
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stdout, p.stderr)
        return 1
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    print(run.stdout, run.stderr)
    return run.returncode


if __name__ == '__main__':
    sys.exit(main())
