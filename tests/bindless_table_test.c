// The slot allocator of the global texture table (runtime/bindless_table.h): classes, lookup, release by view and sampler, a full
// class, and a long random run compared with a plain array.
#include <stdio.h>
#include <stdlib.h>

#include "bindless_table.h"

static int fails;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

static int released_cb_n;
static void cb(void *u, uint32_t cls, uint32_t slot) { (void)u, (void)cls, (void)slot; released_cb_n++; }

static uint64_t rng = 88172645463325252ull;
static uint32_t rnd(uint32_t n) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng % n); }

int main(void) {
    // the class map: 2D is class 1, 2D array 5, multisample 2D 9, shadow 2D 17, cube array shadow 31; buffer and unknown are -1
    CHECK(bt_class_of_type(1) == 0 && bt_class_of_type(3) == 1 && bt_class_of_type(4) == 2 && bt_class_of_type(5) == 3);
    CHECK(bt_class_of_type(3 | 0x100) == 5 && bt_class_of_type(3 | 0x200) == 9 && bt_class_of_type(3 | 0x400) == 17);
    CHECK(bt_class_of_type(5 | 0x100 | 0x200 | 0x400) == 31);
    CHECK(bt_class_of_type(2) == -1 && bt_class_of_type(0) == -1 && bt_class_of_type(9) == -1);

    uint32_t cap[BT_CLASSES] = {0};
    cap[1] = 8, cap[5] = 4;
    BtTable t;
    CHECK(bt_init(&t, cap));
    int fresh;
    BtKey a = {0x10, 0x20, 1}, b = {0x11, 0x20, 1}, c = {0x10, 0x21, 1}, d = {0x10, 0x20, 2};
    uint32_t sa = bt_acquire(&t, 1, &a, &fresh);
    CHECK(sa >= 1 && fresh);
    CHECK(bt_acquire(&t, 1, &a, &fresh) == sa && !fresh);
    uint32_t sb = bt_acquire(&t, 1, &b, &fresh), sc = bt_acquire(&t, 1, &c, &fresh), sd = bt_acquire(&t, 1, &d, &fresh);
    CHECK(sb != sa && sc != sa && sc != sb && sd != sa && sd != sc && sd != sb);   // view, sampler and layout all separate
    CHECK(bt_acquire(&t, 5, &a, &fresh) >= 1 && fresh);                            // another class: its own slot space
    CHECK(bt_find(&t, 1, &a) == sa && bt_find(&t, 5, &b) == 0 && bt_find(&t, 2, &a) == 0);
    CHECK(bt_key(&t, 1, sb) && bt_key(&t, 1, sb)->view == 0x11 && !bt_key(&t, 1, 0));
    CHECK(bt_acquire(&t, 2, &a, &fresh) == 0 && !fresh);   // an absent class gives the dummy slot

    // release by view frees the slots of view 0x10 in every class; b stays
    uint64_t gen = t.gen;
    released_cb_n = 0;
    CHECK(bt_release_view(&t, 0x10, cb, NULL) == 4 && released_cb_n == 4 && t.gen == gen + 4);   // a, c, d and class 5 of a
    CHECK(bt_find(&t, 1, &a) == 0 && bt_find(&t, 1, &c) == 0 && bt_find(&t, 1, &d) == 0 && bt_find(&t, 1, &b) == sb);
    // freed slots come back for new keys and nothing else changes
    BtKey e = {0x30, 0x20, 1};
    uint32_t se = bt_acquire(&t, 1, &e, &fresh);
    CHECK(fresh && (se == sa || se == sc || se == sd));
    CHECK(bt_release_sampler(&t, 0x20, cb, NULL) == 2);   // b and e carry sampler 0x20
    CHECK(bt_find(&t, 1, &b) == 0 && bt_find(&t, 1, &e) == 0 && bt_live(&t, 1) == 0);

    // a full class (cap 4 = three usable slots) gives 0, counts it, and recovers after a release
    BtKey k[5];
    for (int i = 0; i < 5; i++) k[i] = (BtKey){0x100 + (uint64_t)i, 1, 1};
    uint32_t s5[5];
    for (int i = 0; i < 5; i++) s5[i] = bt_acquire(&t, 5, &k[i], &fresh);
    CHECK(s5[0] && s5[1] && s5[2] && s5[3] == 0 && s5[4] == 0 && t.full >= 2);
    CHECK(bt_release_view(&t, 0x101, NULL, NULL) == 1);
    CHECK(bt_acquire(&t, 5, &k[3], &fresh) == s5[1] && fresh);

    // random run against a reference: 64 views x 8 samplers x 2 layouts in a class of 4096
    bt_free(&t);
    cap[1] = 4096, cap[5] = 0;
    CHECK(bt_init(&t, cap));
    static uint32_t ref[64][8][2];   // slot or 0
    for (int step = 0; step < 200000; step++) {
        uint32_t op = rnd(100), v = rnd(64), s = rnd(8), l = rnd(2);
        BtKey kk = {v + 1, s + 1, l};
        if (op < 80) {
            uint32_t got = bt_acquire(&t, 1, &kk, &fresh);
            if (ref[v][s][l]) {
                CHECK(got == ref[v][s][l] && !fresh);
            } else {
                CHECK(got && fresh);
                ref[v][s][l] = got;
            }
        } else if (op < 90) {
            bt_release_view(&t, v + 1, NULL, NULL);
            for (int i = 0; i < 8; i++) ref[v][i][0] = ref[v][i][1] = 0;
        } else if (op < 95) {
            bt_release_sampler(&t, s + 1, NULL, NULL);
            for (int i = 0; i < 64; i++) ref[i][s][0] = ref[i][s][1] = 0;
        } else {
            CHECK(bt_find(&t, 1, &kk) == ref[v][s][l]);
        }
        if (fails > 8) break;
    }
    uint32_t live = 0;
    for (int v = 0; v < 64; v++)
        for (int s = 0; s < 8; s++)
            for (int l = 0; l < 2; l++) live += ref[v][s][l] != 0;
    CHECK(bt_live(&t, 1) == live);
    bt_free(&t);
    printf(fails ? "bindless-table: %d checks failed\n" : "bindless-table: passed\n", fails);
    return fails != 0;
}
