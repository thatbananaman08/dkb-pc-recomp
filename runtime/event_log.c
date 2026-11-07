// Structured events for selected failures. Each record is assembled on the stack and appended in one write; no
// event lock, heap allocation, or file open is needed on the crash path. The existing crash report remains primary.
#include "event_log.h"

#include <stddef.h>

static PlatAppend *event_file;

int rt_event_open(const char *path) {
    if (!path || !*path || __atomic_load_n(&event_file, __ATOMIC_ACQUIRE)) return 0;
    PlatAppend *f = plat_append_create_new(path);
    if (!f) return 0;
    __atomic_store_n(&event_file, f, __ATOMIC_RELEASE);
    return 1;
}

typedef struct { char bytes[4096]; size_t n; int failed, truncated; } Event;

static void byte(Event *e, char c) {
    if (e->n == sizeof e->bytes) e->failed = 1;
    else e->bytes[e->n++] = c;
}

static void text(Event *e, const char *s) { while (*s) byte(e, *s++); }

static unsigned utf8_width(const unsigned char *s, size_t remaining) {
    unsigned char c = s[0];
    unsigned n = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
    if (!n || remaining < n) return 0;
    for (unsigned i = 1; i < n; i++) if (!s[i] || (s[i] & 0xc0) != 0x80) return 0;
    if ((c == 0xe0 && s[1] < 0xa0) || (c == 0xed && s[1] >= 0xa0) ||
        (c == 0xf0 && s[1] < 0x90) || (c == 0xf4 && s[1] >= 0x90)) return 0;
    return n;
}

// Preserve valid UTF-8. Escape control characters and invalid bytes so arbitrary diagnostic strings remain JSON.
static void string(Event *e, const char *s, size_t limit) {
    static const char hex[] = "0123456789abcdef";
    if (!s) { text(e, "null"); return; }
    byte(e, '"');
    size_t i = 0;
    while (i < limit && s[i]) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 0x80) {
            unsigned n = utf8_width((const unsigned char *)s + i, limit - i);
            if (n) { for (unsigned k = 0; k < n; k++) byte(e, s[i++]); continue; }
        }
        if (c == '"' || c == '\\') byte(e, '\\'), byte(e, (char)c);
        else if (c < 0x20 || c >= 0x7f) {
            text(e, "\\u00"); byte(e, hex[c >> 4]); byte(e, hex[c & 15]);
        } else byte(e, (char)c);
        i++;
    }
    if (i == limit && s[i]) e->truncated = 1;
    byte(e, '"');
}

static void number(Event *e, uint64_t n) {
    char digits[20];
    unsigned k = 0;
    do { digits[k++] = (char)('0' + n % 10); n /= 10; } while (n);
    while (k) byte(e, digits[--k]);
}

static void signed_number(Event *e, int64_t n) {
    if (n < 0) { byte(e, '-'); number(e, (uint64_t)(-(n + 1)) + 1); }
    else number(e, (uint64_t)n);
}

static void address(Event *e, uint64_t n) {
    static const char hex[] = "0123456789abcdef";
    text(e, "\"0x");
    for (int shift = 60; shift >= 0; shift -= 4) byte(e, hex[(n >> shift) & 15]);
    byte(e, '"');
}

static void begin(Event *e, const char *category, const char *kind) {
    *e = (Event){0};
    text(e, "{\"version\":1,\"category\":"); string(e, category, 32);
    text(e, ",\"level\":\"error\",\"kind\":"); string(e, kind, 32);
    text(e, ",\"time_unix_ms\":"); number(e, plat_unix_time_ns() / 1000000);
}

static void finish(Event *e) {
    PlatAppend *f = __atomic_load_n(&event_file, __ATOMIC_ACQUIRE);
    if (!f) return;
    if (e->truncated) text(e, ",\"truncated\":true");
    text(e, "}\n");
    if (!e->failed) (void)plat_append_write(f, e->bytes, e->n);
}

void rt_event_fatal(const char *message) {
    if (!__atomic_load_n(&event_file, __ATOMIC_ACQUIRE)) return;
    Event e; begin(&e, "crash", "fatal");
    text(&e, ",\"message\":"); string(&e, message, 384);
    finish(&e);
}

void rt_event_crash(const PlatCrash *x) {
    if (!__atomic_load_n(&event_file, __ATOMIC_ACQUIRE)) return;
    static _Thread_local Event e; // a stack-overflow report should not require a 4 KiB stack buffer
    begin(&e, "crash", "reported");
    text(&e, ",\"message\":"); string(&e, x->what, 128);
    text(&e, ",\"code\":"); number(&e, x->code);
    text(&e, ",\"host_address\":"); address(&e, x->rip);
    text(&e, ",\"module\":"); string(&e, x->module ? x->module : "executable", 128);
    text(&e, ",\"module_offset\":"); address(&e, x->offset);
    text(&e, ",\"access\":"); signed_number(&e, x->access);
    text(&e, ",\"fault_address\":"); if (x->access < 0) text(&e, "null"); else address(&e, x->addr);
    text(&e, ",\"minidump\":"); string(&e, x->dump, 256);
    finish(&e);
}

void rt_event_device_lost(int32_t result) {
    if (!__atomic_load_n(&event_file, __ATOMIC_ACQUIRE)) return;
    Event e; begin(&e, "device_lost", "vulkan");
    text(&e, ",\"result\":"); signed_number(&e, result);
    finish(&e);
}

void rt_event_shader_failure(const char *reason, const char *shader) {
    if (!__atomic_load_n(&event_file, __ATOMIC_ACQUIRE)) return;
    Event e; begin(&e, "shader_translation", "failure");
    text(&e, ",\"reason\":"); string(&e, reason, 384);
    text(&e, ",\"shader\":"); string(&e, shader, 128);
    finish(&e);
}
