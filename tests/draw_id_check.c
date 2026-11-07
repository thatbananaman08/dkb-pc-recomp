// Game-free checks of the draw identifier table used by the ordered dump.
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "draw_id.h"

int main(void) {
    DrawIdSlot first[32] = {0}, second[32] = {0}, full[1] = {0}, collisions[2] = {0};
    assert(draw_id_next(first, 32, 1, 2, 3, 4) == 1);
    assert(draw_id_next(first, 32, 9, 2, 3, 4) == 1);
    assert(draw_id_next(first, 32, 1, 2, 3, 4) == 2);
    // Interleaving another shader group does not change the repeated group's occurrence numbers.
    assert(draw_id_next(second, 32, 1, 2, 3, 4) == 1);
    assert(draw_id_next(second, 32, 1, 2, 3, 4) == 2);
    assert(draw_id_next(second, 32, 9, 2, 3, 4) == 1);
    assert(draw_id_next(second, 32, 1, 8, 3, 4) == 1); // geometry stage is part of the key
    assert(draw_id_next(second, 32, 1, 2, 3, 5) == 1); // replay operation is part of the key
    assert(draw_id_next(full, 1, 0, 0, 0, 0) == 1);    // zero hashes are valid keys
    assert(draw_id_next(full, 1, 1, 0, 0, 0) == 0);    // no silent merge on a collision or full table
    assert(draw_id_next(full, 1, 0, 0, 0, 0) == 2);
    // These keys have the same hash modulo two; both remain independently addressable.
    assert(draw_id_next(collisions, 2, 1, 2, 3, 4) == 1);
    assert(draw_id_next(collisions, 2, 3, 2, 3, 4) == 1);
    assert(draw_id_next(collisions, 2, 1, 2, 3, 4) == 2);
    assert(draw_id_next(collisions, 2, 3, 2, 3, 4) == 2);
    memset(second, 0, sizeof second);
    assert(draw_id_next(second, 32, 1, 2, 3, 4) == 1); // a new capture starts at one
    assert(draw_id_next(second, 0, 1, 2, 3, 4) == 0);
    puts("draw IDs: grouping, interleaving, collisions and reset passed");
    return 0;
}
