/* Game-free AArch64 programs for the whole-program differential test. No headers or runtime are needed. */
typedef unsigned long long u64;

__attribute__((noinline))
static u64 mix(u64 x) {
    return (x ^ (x >> 7)) * 13 + 9;
}

__attribute__((noinline))
static u64 combine(u64 x, u64 y) {
    return mix(x) + mix(y ^ 17);
}

__attribute__((noinline))
u64 entry_tail(u64 *out, u64 n) {
    out[0] = n;
    return combine(n, n + 1);
}

__attribute__((noinline))
u64 entry_calls(u64 *out, u64 n) {
    u64 value = combine(n, n + 3);
    out[0] = value;
    out[1] = mix(value);
    return out[1] ^ value;
}

__attribute__((noinline))
u64 entry_loop(u64 *out, u64 n) {
    u64 acc = 1;
    for (u64 i = 0; i < (n & 15); i++)
        acc = (acc * 3) ^ (i + n);
    out[0] = acc;
    return acc + n;
}

__attribute__((noinline))
u64 entry_switch(u64 *out, u64 n) {
    u64 value;
    switch (n & 3) {
    case 0: value = n + 31; break;
    case 1: value = n ^ 0x55; break;
    case 2: value = n * 7; break;
    default: value = n - 17; break;
    }
    out[0] = value;
    return value;
}
