// Opt-in JSONL diagnostics. Open once after guest-address reservation, before starting worker threads.
#ifndef DKB_EVENT_LOG_H
#define DKB_EVENT_LOG_H

#include <stdint.h>
#include "platform/platform.h"

int rt_event_open(const char *path);  // create a new file; never overwrite an existing one
void rt_event_fatal(const char *message);
void rt_event_crash(const PlatCrash *crash);
void rt_event_device_lost(int32_t result);
void rt_event_shader_failure(const char *reason, const char *shader);

#endif
