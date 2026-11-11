// TOTK_PACE_LOG=1: frame pacing measurements, one summary line every 60 frames (pace_log.c). What the game asks for
// (present interval, the pacer's period and slot wait), when the host present returns, and, with VK_KHR_present_id and
// present_wait (enabled only for this switch), when each frame really reaches the display. Intervals are summarised as a
// histogram, p50/p95/p99 and the count of uneven frames (an interval more than 4 ms from the previous one).
#pragma once
#include <stdint.h>

int pace_log_on(void);   // TOTK_PACE_LOG=1
void pace_log_request(int interval, uint64_t period_ns, uint64_t wait_ns);   // nvnQueuePresentTexture, the game's thread
void pace_log_host_present(uint64_t t_ns);                                   // the host vkQueuePresentKHR returned
void pace_log_hold(uint64_t held_ns);                                         // the flip thread held a present this long for its slot
void pace_log_display(uint64_t t_ns);                                        // vkWaitForPresentKHR returned: the image is on the display

// The summary of a series of intervals (ms). hist: <12, 12-20, 20-28, 28-37, 37-50, >50.
typedef struct { int n, hist[6], uneven; double p50, p95, p99, mean; } PaceStats;
void pace_stats(const double *ms, int n, PaceStats *out);
