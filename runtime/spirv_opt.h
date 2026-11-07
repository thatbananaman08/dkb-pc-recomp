// SPIR-V clean-up that every translated shader goes through (spirv_opt.cpp).
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Runs SPIRV-Tools' --eliminate-local-single-block pass (CreateLocalSingleBlockLoadStoreElimPass) over a module: inside
// one basic block, a load of a function-local variable that follows a store to it is replaced by the stored value.
// Returns a malloc'ed copy of the optimized module (*out_words its length), or NULL on failure with the reason in err
// (cap bytes); the input is never modified. The pass drops loads and keeps every arithmetic instruction and decoration
// (NoContraction included), so the math is unchanged.
uint32_t *spirv_opt_local_single_block(const uint32_t *words, uint64_t n, uint64_t *out_words, char *err, size_t cap);

#ifdef __cplusplus
}
#endif
