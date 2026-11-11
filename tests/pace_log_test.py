"""TOTK_PACE_LOG statistics (runtime/pace_log.c): histogram bins, percentiles, the uneven-frame count and the summary line.

Compiles the production pace_log.c with stubs for the logger, config and clock.
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
#include "pace_log.c"

static char logbuf[1 << 14];
void rt_log(const char *fmt, ...) { va_list a; va_start(a, fmt); size_t n = strlen(logbuf); vsnprintf(logbuf + n, sizeof logbuf - n, fmt, a); strcat(logbuf, "\n"); va_end(a); }
const char *cfg_get(const char *n) { return !strcmp(n, "TOTK_PACE_LOG") ? "1" : NULL; }
uint64_t plat_process_uptime_ns(void) { return 12500000000ull; }
void plat_mutex_lock(PlatMutex *m) { (void)m; }
void plat_mutex_unlock(PlatMutex *m) { (void)m; }

static int fails;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } } while (0)

int main(void) {
    // 16.7 / 33.3 alternation: every frame is uneven; steady 33.3 is not
    double alt[8] = {16.7, 33.3, 16.7, 33.3, 16.7, 33.3, 16.7, 33.3}, steady[8] = {33.3, 33.3, 33.3, 33.4, 33.3, 33.2, 33.3, 33.3};
    PaceStats s;
    pace_stats(alt, 8, &s);
    CHECK(s.uneven == 7 && s.hist[1] == 4 && s.hist[3] == 4, "alternation: 7 uneven, bins 12-20 and 28-37");
    CHECK(s.p50 == 16.7 && s.p95 == 33.3 && s.mean > 24.9 && s.mean < 25.1, "alternation: percentiles and mean");
    pace_stats(steady, 8, &s);
    CHECK(s.uneven == 0 && s.hist[3] == 8, "steady: no uneven frames");
    double edges[6] = {11.99, 12.0, 19.99, 28.0, 37.0, 50.0};
    pace_stats(edges, 6, &s);
    CHECK(s.hist[0] == 1 && s.hist[1] == 2 && s.hist[3] == 1 && s.hist[4] == 1 && s.hist[5] == 1, "bin edges: lower bound included");
    pace_stats(NULL, 0, &s);
    CHECK(s.n == 0 && s.mean == 0, "empty series");
    // the summary line: 61 host presents (60 intervals) with no display source, then the requests
    for (int i = 0; i < 60; i++) pace_log_request(i % 2 ? 2 : 1, 16666666, 3000000);
    uint64_t t = 1000000000ull;
    for (int i = 0; i < 61; i++) {
        pace_log_host_present(t);
        t += (i % 2 ? 33300000ull : 16700000ull);
    }
    CHECK(strstr(logbuf, "pace t=12.5 s, 60 frames: host present: <12 0, 12-20 30, 20-28 0, 28-37 30"), "summary: histogram");
    CHECK(strstr(logbuf, "uneven 59"), "summary: uneven count");
    CHECK(strstr(logbuf, "game asked interval 1/2/3+ 30/30/0, pacer period 16.7 ms, slot wait 3.0 ms"), "summary: the game's requests");
    // with a display source the summary is cut by display events
    logbuf[0] = 0;
    for (int i = 0; i < 61; i++) pace_log_display(2000000000ull + (uint64_t)i * 16666666ull);
    CHECK(strstr(logbuf, "display (present_wait): <12 0, 12-20 60") && strstr(logbuf, "uneven 0"), "display summary");
    if (fails) fputs(logbuf, stdout);
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
    out = ROOT / 'build' / 'test' / 'pace-log'
    out.mkdir(parents=True, exist_ok=True)
    src = out / 'pace_log_test.c'
    src.write_text(HARNESS)
    exe = out / 'pace_log_test.exe'
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
