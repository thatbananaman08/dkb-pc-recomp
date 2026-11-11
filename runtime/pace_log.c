// TOTK_PACE_LOG: see pace_log.h.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "pace_log.h"
#include "platform/platform.h"

#define WINDOW 60

int pace_log_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("TOTK_PACE_LOG");
        on = e && !strcmp(e, "1");
    }
    return on;
}

static int dcmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

void pace_stats(const double *ms, int n, PaceStats *o) {
    memset(o, 0, sizeof *o);
    o->n = n;
    if (n <= 0) return;
    static const double edge[5] = {12, 20, 28, 37, 50};
    double sum = 0;
    double *s = malloc((size_t)n * sizeof *s);
    for (int i = 0; i < n; i++) {
        int b = 0;
        while (b < 5 && ms[i] >= edge[b]) b++;
        o->hist[b]++;
        sum += ms[i];
        if (i && fabs(ms[i] - ms[i - 1]) > 4.0) o->uneven++;
        s[i] = ms[i];
    }
    qsort(s, (size_t)n, sizeof *s, dcmp);
    o->p50 = s[(size_t)((n - 1) * 0.50)];
    o->p95 = s[(size_t)((n - 1) * 0.95)];
    o->p99 = s[(size_t)((n - 1) * 0.99)];
    o->mean = sum / n;
    free(s);
}

static PlatMutex lk = PLAT_MUTEX_INIT;
static double disp[WINDOW], host[WINDOW], period[WINDOW], wait_ms[WINDOW];
static double held_ms[WINDOW];
static int n_disp, n_host, n_req, n_held, req_iv[4];
static uint64_t last_disp, last_host;
static int have_display;

static void summary(const char *src, const double *v, int n, char *out, size_t sz) {
    PaceStats s;
    pace_stats(v, n, &s);
    snprintf(out, sz, "%s: <12 %d, 12-20 %d, 20-28 %d, 28-37 %d, 37-50 %d, >50 %d; p50 %.1f p95 %.1f p99 %.1f ms, mean %.1f ms (%.1f fps), uneven %d", src,
             s.hist[0], s.hist[1], s.hist[2], s.hist[3], s.hist[4], s.hist[5], s.p50, s.p95, s.p99, s.mean, s.mean > 0 ? 1000.0 / s.mean : 0.0, s.uneven);
}

// Called with lk held when the leading series has WINDOW samples.
static void emit(void) {
    char a[360], b[360];
    double wm = 0, pm = 0, hm = 0;
    for (int i = 0; i < n_held; i++) hm += held_ms[i];
    if (n_held) hm /= n_held;
    for (int i = 0; i < n_req; i++) wm += wait_ms[i], pm += period[i];
    if (n_req) wm /= n_req, pm /= n_req;
    summary("host present", host, n_host, b, sizeof b);
    if (have_display && n_disp) {
        summary("display (present_wait)", disp, n_disp, a, sizeof a);
        rt_log("pace t=%.1f s, %d frames: %s | %s | game asked interval 1/2/3+ %d/%d/%d, pacer period %.1f ms, slot wait %.1f ms, flip hold %.1f ms",
               (double)plat_process_uptime_ns() / 1e9, n_disp, a, b, req_iv[1], req_iv[2], req_iv[3], pm, wm, hm);
    } else {
        rt_log("pace t=%.1f s, %d frames: %s | game asked interval 1/2/3+ %d/%d/%d, pacer period %.1f ms, slot wait %.1f ms, flip hold %.1f ms",
               (double)plat_process_uptime_ns() / 1e9, n_host, b, req_iv[1], req_iv[2], req_iv[3], pm, wm, hm);
    }
    n_disp = n_host = n_req = n_held = 0;
    memset(req_iv, 0, sizeof req_iv);
}

void pace_log_request(int interval, uint64_t period_ns, uint64_t wait_ns) {
    if (!pace_log_on()) return;
    plat_mutex_lock(&lk);
    if (n_req < WINDOW) {
        period[n_req] = (double)period_ns / 1e6;
        wait_ms[n_req++] = (double)wait_ns / 1e6;
    }
    req_iv[interval < 1 ? 1 : interval > 3 ? 3 : interval]++;
    plat_mutex_unlock(&lk);
}

void pace_log_hold(uint64_t held_ns) {
    if (!pace_log_on()) return;
    plat_mutex_lock(&lk);
    if (n_held < WINDOW) held_ms[n_held++] = (double)held_ns / 1e6;
    plat_mutex_unlock(&lk);
}

void pace_log_host_present(uint64_t t_ns) {
    if (!pace_log_on()) return;
    plat_mutex_lock(&lk);
    if (last_host && n_host < WINDOW) host[n_host++] = (double)(t_ns - last_host) / 1e6;
    last_host = t_ns;
    if (!have_display && n_host == WINDOW) emit();
    plat_mutex_unlock(&lk);
}

void pace_log_display(uint64_t t_ns) {
    if (!pace_log_on()) return;
    plat_mutex_lock(&lk);
    have_display = 1;
    if (last_disp && n_disp < WINDOW) disp[n_disp++] = (double)(t_ns - last_disp) / 1e6;
    last_disp = t_ns;
    if (n_disp == WINDOW) emit();
    plat_mutex_unlock(&lk);
}
