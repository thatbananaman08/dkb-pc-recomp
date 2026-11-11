// The DKB_CENSUS hooks of runtime/census.c that runtime/nvn_api_cmd.c calls, as no-ops for the CPU tests that link the recorder alone.
#include <stdint.h>

int census_flag = 0, split_flag = 0, split_sampling = 0;
int census_read(void) { return 0; }
int split_read(void) { return 0; }
void r0_begin(uint64_t tpidr) { (void)tpidr; }
void r0_op(int draw) { (void)draw; }
void r0_end(void) {}
void r0_inline(uint64_t handle, const void *body, uint64_t bytes) { (void)handle, (void)body, (void)bytes; }
