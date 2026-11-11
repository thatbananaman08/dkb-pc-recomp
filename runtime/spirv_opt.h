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

// Diagnostic patches for the deferred-light fragment family that faults the NVIDIA driver (DKB_LQ_PATCH; gfx_shader.c
// lq_patch). The family: a fragment module that reads element 83 of the uniform block named fp_c0 (or, with no names,
// of any uniform block). kinds, bit 0: drop capabilities (with their extensions) no instruction uses (interlock, Float64,
// DrawParameters, unused GroupNonUniform*); bit 1: the entry function's OpKill becomes OpDemoteToHelperInvocation +
// OpReturn; bit 2: implicit-LOD samples inside loops become explicit-LOD with Lod 0.0 (Bias and MinLod dropped);
// bit 3 (nullfs, breaks rendering): in a family module the entry function becomes one block that stores OpConstantNull into
// every Output of the entry interface and returns, then dead functions, code, constants and unused interface variables are
// removed; bit 4 (cullvs, breaks rendering): a vertex module with an Input variable decorated Location 9 stores
// vec4(0, 0, 2, 1) into its BuiltIn Position (a variable, or a member of a gl_PerVertex-style block) before every
// OpReturn of the entry function, so the primitive is clipped away; everything else stays; bit 5 (nullvs, breaks
// rendering): the same vertex modules get a new entry function that stores that Position, a null into every other Output
// and returns, then the dead-code passes of nullfs run (the inputs leave the interface); bit 6 (passvs): like nullvs, but
// it loads every Input with a Location (in Location order) and stores the k-th into the Output with the k-th smallest
// Location when their types match (null otherwise), so those inputs stay in the interface.
// caps (bit 0) also applies to those vertex modules. Bit 7 (vsmask) with the mask in bits 8..24 (bit k, 0..15: keep the
// Output at Location k; bit 16: keep Position): in a matched vertex module the original stores to every other Location
// output (and to Position when bit 16 is clear) are dropped, and OpConstantNull (vec4(0,0,2,1) for Position) is stored
// before every OpReturn of the entry function; DCE then removes the code that only fed them (the interface stays, so
// the inputs stay declared). On success with bit 7, err holds a line with the entry function's instruction counts.
// Bits 28..30, for the same vertex modules and independent of the mask: flatten (locals to SSA values, if-conversion, dead
// branch elimination, block merge, simplification, then SelectionControl Flatten on every remaining OpSelectionMerge),
// notrig (every GLSL.std.450 Sin / Cos result becomes OpCopyObject of a null), noflags (every OpBitwiseAnd result
// becomes one). Bit 31 (offline experiments only) makes flatten also combine access chains and run twice with DCE.
// The environment variable DKB_LQ_NOFLAGS_ONLY=<comma list of hex constants and/or nonconst>, read once by the patcher, limits noflags
// to the OpBitwiseAnd with an integer OpConstant operand (either side) in the list, or with no constant operand (nonconst).
// Bit 31 (ubocomp), for every module of any stage: a load through an OpAccessChain / OpInBoundsAccessChain into a Uniform or
// StorageBuffer variable whose last index selects a vector component and is not a constant becomes an OpLoad of the
// whole vector (through the chain without its last index), one OpCompositeExtract per component and
// (i==1)?y : (i==2)?z : (i==3)?w : x as OpIEqual / OpSelect on the index (an out-of-range index, undefined before, gives x).
// Chains also used by a store or atomic stay as they are (their loads are rewritten). The flatten experiment variant is
// the environment variable DKB_LQ_FLATTEN_EXTRA.
// Bit 8 while bit 7 is clear (ldcfix, any module; a SPIR-V post-processing, not a change of the translator): a triangle
// `if (c) { arm }` whose c compares `x & 3` and whose arm only loads, computes and stores function-local variables becomes
// the arm run unconditionally with each store turned into a select of the new and the old value; no data-dependent branch
// is left for those.
// Bit 9 while bit 7 is clear (and3exact, any module): every test-class `x & 3` (see DKB_LQ_AND3_CLASS) becomes
// x - ((x >> 2) << 2), exactly the same value, and each compare of one with a constant becomes an OpVectorExtractDynamic from a
// constant bool vector holding the compare for 0..3: no OpBitwiseAnd and no compare is left at those sites, and the maths is unchanged.
// Bit 10 while bit 7 is clear (rotexact, any module): the light shaders' one-hot weighted word pick -- four words each multiplied
// (through a 32-bit multiply built from 16-bit halves) by a weight local that is a function of r = x & 3 alone and 1 for exactly one
// r -- becomes an OpVectorExtractDynamic of the words' bit patterns at r, bit-identical; the weights' code is then dead and removed.
// Bit 11 while bit 7 is clear (loadexact, any module): rotexact's sums, but the four words' buffer loads are replaced by one load of the
// selected word: the index is chosen among the four original index values by r (a select chain) and the row / component are taken from it
// as the original did; two sums that are halves of one pick share that load. Bit-identical.
// Bit 12 while bit 7 is clear (fold, vertex modules that read Location 9 only): noflags limited to the AND-with-3 results that
// feed a compare or select (DKB_LQ_NOFLAGS_ONLY=3 and DKB_LQ_AND3_CLASS=test, ignoring those variables); it applies (applied bit 18)
// only if exactly two such ANDs are found, else returns 0 and the module is untouched. gfx_shader.c lq_fold runs it by default on
// two named stages.
// DKB_LQ_AND3_CLASS=<comp,round,test> (with noflags, normally with DKB_LQ_NOFLAGS_ONLY=3): only the OpBitwiseAnd with the constant 3
// of the listed classes are nulled. round: (x>>31)&3; comp: the result is the last index of an access chain into a Uniform /
// StorageBuffer variable; test: it feeds an integer compare, OpSelect or OpSwitch (priority in that order; the rest is left).
// Bits 25..27 (vscut): a cut selector 1..7 (tdepth, scale, rot, flags, pos, offset, center); in a matched vertex module that defines the
// ids of that cut with the expected opcodes (the captured Ground light-quad shader) each
// of those results becomes OpCopyObject of a null of its type, DCE removes what only fed them (bit 8 of *applied); any
// other matched vertex module gets mask 0 instead (bit 9 of *applied, not bit 7).
// Returns 1 and a malloc'ed module (*applied: the kinds that changed something), 0 when the module is not in the family
// or nothing applies, -1 on failure (err). The result is validated with SPIRV-Tools; the input is never modified.
int spirv_lq_patch(const uint32_t *words, uint64_t n, unsigned kinds, uint32_t **out, uint64_t *out_words,
                   unsigned *applied, char *err, size_t cap);
// The same with force_vertex: the vertex kinds (caps, cullvs, nullvs, passvs, vsmask, vscut, flatten, notrig, noflags, and vsblock)
// also apply to a vertex module without an Input at Location 9. gfx_shader.c passes it when DKB_LQ_PATCH_KEYS names the module.
// Bit 13 (vsblock, while bit 7 is clear): DKB_LQ_VSBLOCK=<a>-<b>[:t|:f] replaces the branch condition of the selection
// constructs a..b (OpSelectionMerge, counted from 0 in module order) by a constant and removes the dead path; applied bit 19.
// Bit 14 (selfix, while bit 7 is clear): DKB_LQ_SELFIX=<a>[-<b>][,...] names selection constructs (counted as for vsblock); each one that is a
// triangle whose arm is one block of loads, pure arithmetic and stores of function-local variables is replaced by the arm run
// unconditionally, every store becoming a select of the new and the old value (ldcfix's rewrite, without its condition test).
// Exact for every input. Applied bit 20.
// Fold site (bits 16 / 17 together with bit 12; DKB_LQ_FOLD_SITE=export|branch for the stages of DKB_LQ_FOLD_ORIGINS): null only one
// of the two test-class &3 ANDs, the one that reaches no branch condition (export, bit 16) or the one that does (branch, bit 17);
// the module is left alone unless exactly one AND is of each kind.
// Bit 18 with bit 12 (site 4, DKB_LQ_FOLD_SITE=construct with DKB_LQ_FOLD_CONSTRUCT=<n>, or the construct in bits 19..24 of kinds, force_vertex lets it reach a vertex module without a
// Location 9 input): null every test-class &3 AND that reaches the condition of selection construct n; fails closed when none does.
// Both bits (site 3, DKB_LQ_FOLD_SITE=exact): nothing is nulled; every selection construct whose condition the branch-reaching AND reaches
// must be a triangle with a pure arm, and each becomes ldcfix's branch-free select of the new and the old value (exact for every input).
// Bit 15 (mul2, while bit 7 is clear, any module): every test-class `&3` of the module must read select(cond, sum, -1) (directly or
// through a function-local variable stored once) with sum = shl(X, 16) + V; the shifted term has no bits below 16, so the &3 ANDs are
// pointed at a twin select(cond, V, -1), and when V = lo(P) * lo(Q) + C the product is reduced to (P & 3) * (Q & 3) + (C & 3). Exact for
// every input. The original select stays for its other users (the multiply chain goes when nothing else uses it). Otherwise the
// module is left alone. Applied bit 21.
int spirv_lq_patch_ex(const uint32_t *words, uint64_t n, unsigned kinds, uint32_t **out, uint64_t *out_words,
                      unsigned *applied, char *err, size_t cap, int force_vertex);

#ifdef __cplusplus
}
#endif
