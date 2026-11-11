// The decision of vk_exec_draw.c light_quad_fault_skip, kept free of Vulkan so a test can run it (tests/lq_skip_test.py).
#pragma once

// enabled: TOTK_FIX_VS_FAULT is on and the platform is Bananza on NVIDIA. match: light_quad_match (0 none, 1 a known faulting
// stage, 2 the family). folded: the draw's vertex module was changed by the light-quad fold (gfx_shader_folded). A folded
// module is drawn; every other match is skipped, as before the fold existed.
static inline int lq_skip_decision(int enabled, int match, int folded) {
    if (!enabled || !match) return 0;
    return !folded;
}

// The fold's own fail-safe (vk_exec_draw.c lq_fold_missing): rule: the draw's vertex stage is named in the fold rules (an origin
// with a rule, the fold enabled on Bananza + NVIDIA); folded: its module was folded (gfx_shader_folded). A module that should
// have been folded and was not (the pattern was not there, or the module came in by a path that did not know the stage) is not
// drawn, whether or not the stage is a light quad: its pipeline is what loses the device or crashes the driver's compiler.
static inline int lq_fold_missing_decision(int platform, int rule, int folded) {
    return platform && rule && !folded;
}

// gfx_shader_fold_fix: a module that a request for a stage with a fold rule gets from the cache is folded then when it was loaded
// without the stage (the boot prewarm, a stored input) and the fold was not tried on it yet. rule: the request's stage has a rule.
static inline int lq_refold_decision(int rule, int folded, int tried) {
    return rule && !folded && !tried;
}
