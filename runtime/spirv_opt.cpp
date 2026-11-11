// The SPIR-V optimizer step of shader translation: one SPIRV-Tools pass, see spirv_opt.h.
//
// Why: the NVIDIA driver's shader compiler (nvgpucomp64.dll) crashes while building a pipeline from some generated
// vertex shaders. The translator stores intermediate values in function-local variables and loads them back within
// the same block; removing those loads avoids the crash and leaves the arithmetic and its NoContraction decorations
// as they were.
#include "spirv_opt.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <cstdio>
#include <map>
#include <set>
#include <cstdlib>
#include <exception>
#include <cstring>
#include <string>
#include <vector>

extern "C" uint32_t *spirv_opt_local_single_block(const uint32_t *words, uint64_t n, uint64_t *out_words, char *err,
                                                  size_t cap) {
    auto fail = [&](const std::string &why) -> uint32_t * {
        if (err && cap) std::snprintf(err, cap, "%s", why.c_str());
        return nullptr;
    };
    if (!words || n < 5) return fail("module shorter than its header");
    try {
        spvtools::Optimizer opt(SPV_ENV_VULKAN_1_3);
        std::string first;
        opt.SetMessageConsumer([&](spv_message_level_t level, const char *, const spv_position_t &, const char *msg) {
            if (level <= SPV_MSG_ERROR && first.empty()) first = msg ? msg : "";
        });
        opt.RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass());
        // The translator's modules are validated by its own tests (tests/manifest.toml); validating each one again on
        // every load would cost as much as the pass.
        spvtools::OptimizerOptions options;
        options.set_run_validator(false);
        std::vector<uint32_t> out;
        if (!opt.Run(words, static_cast<size_t>(n), &out, options)) return fail(first.empty() ? "optimizer failed" : first);
        if (out.size() < 5 || out[0] != 0x07230203u) return fail("optimizer returned a malformed module");
        auto *copy = static_cast<uint32_t *>(std::malloc(out.size() * 4));
        if (!copy) return fail("out of memory");
        std::memcpy(copy, out.data(), out.size() * 4);
        *out_words = out.size();
        return copy;
    } catch (const std::exception &e) {
        return fail(e.what());
    } catch (...) {
        return fail("exception in the optimizer");
    }
}

// ---- diagnostic patches for the NVIDIA deferred-light fragment family (DKB_LQ_PATCH, gfx_shader.c lq_patch) ----
namespace {

enum { OpName_ = 5, OpExtension_ = 10, OpEntryPoint_ = 15, OpExecutionMode_ = 16, OpCapability_ = 17, OpTypeFloat_ = 22,
       OpConstant_ = 43, OpFunction_ = 54, OpFunctionEnd_ = 56, OpVariable_ = 59, OpAccessChain_ = 65,
       OpInBoundsAccessChain_ = 66, OpDecorate_ = 71, OpMemberDecorate_ = 72, OpImageSampleImplicitLod_ = 87,
       OpImageSampleExplicitLod_ = 88, OpImageSampleDrefImplicitLod_ = 89, OpImageSampleDrefExplicitLod_ = 90,
       OpTypeInt_ = 21, OpTypeVector_ = 23, OpTypeStruct_ = 30, OpTypePointer_ = 32, OpConstantComposite_ = 44,
       OpConstantNull_ = 46, OpLoad_ = 61, OpStore_ = 62, OpCopyObject_ = 83, OpExecutionModeId_ = 331, OpExtInstImport_ = 11, OpExtInst_ = 12,
       OpBitwiseAnd_ = 199, OpSelectionMerge_ = 247, OpBranchConditional_ = 250,
       OpLoopMerge_ = 246, OpLabel_ = 248, OpKill_ = 252, OpReturn_ = 253, OpDemoteCore_ = 4415, OpDemoteExt_ = 5380,
       CapDemote_ = 5379 };

// Which capabilities the module's instructions need.
struct Uses {
    bool interlock = false, f64 = false, draw_params = false, basic = false, vote = false, ballot = false,
         shuffle = false, shuffle_rel = false, arith = false, quad = false, keep_group = false;
};

struct Loop { size_t from, to; };

// Opcodes whose first operand is the result type and second the result id (enough of them to find the type of an index).
bool type_first(unsigned op) {
    return op == 12 || (op >= 41 && op <= 46) || (op >= 48 && op <= 52) || op == 54 || op == 55 || op == 57 || op == 59 ||
           op == 60 || op == 61 || (op >= 65 && op <= 70) || (op >= 77 && op <= 84) || (op >= 86 && op <= 107) ||
           (op >= 109 && op <= 212) || (op >= 227 && op <= 242) || op == 245;
}

// ubocomp: one access chain into a Uniform / StorageBuffer variable whose last index picks a vector component dynamically.
struct UcChain {
    uint32_t type = 0, base = 0, last = 0, vec = 0, storage = 0, idx_type = 0, count = 0, short_ptr = 0, vptr = 0;
    uint32_t c[4] = {0, 0, 0, 0};   // the index type's constants 1..3 (c[1..3])
    std::vector<uint32_t> mid;      // the indices before the last one
    unsigned loads = 0, other = 0, stores = 0;
};   // word offsets: the loop header's OpLabel up to its merge block's OpLabel

bool str_eq(const uint32_t *w, size_t nwords, const char *s) {
    size_t l = std::strlen(s);
    if (l + 1 > nwords * 4) return false;
    return !std::memcmp(w, s, l + 1);
}

void put_str(std::vector<uint32_t> &o, uint16_t op, const char *s) {
    size_t l = std::strlen(s) + 1, nw = (l + 3) / 4;
    o.push_back(static_cast<uint32_t>((nw + 1) << 16 | op));
    size_t at = o.size();
    o.resize(at + nw, 0);
    std::memcpy(&o[at], s, l);
}

// DKB_LQ_VSCUT (spirv_opt.h): the instructions of the captured Ground light-quad vertex shader (the family's vertex
// module) whose results are replaced by a null of their type. Selector 1..7 = tdepth, scale, rot, flags, pos, offset, center (the names
// live in gfx_shader.c lq_patch). (result id, opcode) pairs; a module that does not define every one of them with that
// opcode in its entry function is not the captured shader and is left alone.
const std::vector<std::pair<uint32_t, uint16_t>> &vscut_table(unsigned sel) {
    static const std::vector<std::pair<uint32_t, uint16_t>> none,
        tdepth = {{793, 129}, {799, 133}},                                      // OpFAdd, OpFMul: the cascade depth parameter
        scale = {{1081, 133}},                                                  // OpFMul: the per-instance scalar
        rot = {{1696, 12}, {1768, 12}, {1819, 12}, {1822, 12}, {1825, 12}, {1834, 12}},   // OpExtInst Sin / Cos
        flags = {{1178, 61}, {1256, 61}, {1264, 61}, {1272, 61}, {1411, 61}, {1714, 61}, {1740, 61}, {2429, 61},
                 {2494, 61}},                                                   // OpLoad of vp_c9[7].x / .y (bit flags as float)
        pos = {{2651, 129}, {2666, 129}, {2681, 129}, {2696, 129}},             // OpFAdd: the view-space position
        offset = {{2048, 12}, {2057, 12}, {2066, 12}},                          // OpExtInst Fma: the rotated corner offset
        center = {{1294, 129}, {1345, 129}, {1357, 129}};                       // OpFAdd: the quad center
    switch (sel) {
    case 1: return tdepth;
    case 2: return scale;
    case 3: return rot;
    case 4: return flags;
    case 5: return pos;
    case 6: return offset;
    case 7: return center;
    default: return none;
    }
}

// DKB_LQ_NOFLAGS_ONLY=<comma list of hex constants and the token nonconst>: noflags then nulls only the OpBitwiseAnd whose
// OpConstant integer operand (either side) is in the list, or, with nonconst, that have no constant operand. Unset: all.
struct AndFilter {
    bool on = false, nonconst = false;
    std::vector<uint32_t> vals;
    unsigned classes = 0;   // DKB_LQ_AND3_CLASS: comp (1), round (2), test (4); with it only the &3 ANDs of those classes
    bool classes_on = false;
};

// ldcfix: one triangle `if (c) { pure arm }` whose arm only reads and writes function-local variables, and the branch it replaces.
struct LdcPlan {
    uint32_t merge = 0, cond = 0;
    bool arm_true = false;
    size_t arm_begin = 0, arm_term = 0;   // the arm's instructions: [arm_begin, arm_term)
    size_t sm_len = 0, bc_len = 0;
};

// rotexact: a store to a function-local variable inside a triangle arm: the condition that guards it (0: none) and its polarity.
struct LStore {
    uint32_t value, cond;
    bool when_true;
};

// rotexact: one sum of 32-bit products weight * word that picks a word by r, and what replaces it.
struct ChainAll {   // an access chain: its result type, base variable and indices
    uint32_t type = 0, base = 0;
    std::vector<uint32_t> idx;
};

// loadexact: where one picked word comes from: buffer[lead...][x >> 2][x & 3].
struct WordAddr {
    uint32_t x = 0, xtype = 0, row_type = 0, comp_type = 0, c_shift = 0, c_mask = 0, chain_type = 0, base = 0, float_type = 0;
    std::vector<uint32_t> lead;
};

struct RotPlan {
    bool lx_ok = false, shared = false;         // loadexact: the words' addresses are known; this sum's load also serves another sum
    uint32_t merge_into = 0;                    // loadexact: the root of the sum whose load this one reuses
    WordAddr addr[4];                           // loadexact: the address of the word for r = 0..3 (x = 0: none, or in the shared sum)
    uint32_t type = 0, r_id = 0;               // the int type, the `x & 3` result
    std::vector<uint32_t> extras;              // the addends of the sum that are not weighted words (kept as they are)
    uint32_t slot_key[4] = {0, 0, 0, 0};       // the word picked for r = 0..3 (0: the pick is 0): a float local (high bit set) or a float value id
};

// An OpBitwiseAnd with the constant 3: the class of its use. round: (x>>31)&3, the rounding fix-up of a signed divide by 4;
// comp: the result (through bitcasts) is the last index of an access chain into a Uniform / StorageBuffer variable; test:
// it feeds an integer compare, an OpSelect or an OpSwitch. In that priority; 0: none of them.
struct And3 {
    uint32_t other = 0, type = 0;
    bool comp = false, test = false;
    unsigned cls = 0;
};

const AndFilter &and_filter() {
    static const AndFilter f = [] {
        AndFilter r;
        const char *e = std::getenv("DKB_LQ_NOFLAGS_ONLY");
        const char *cl = std::getenv("DKB_LQ_AND3_CLASS");
        if (cl && *cl) {
            r.classes_on = true;
            std::string c = std::string(",") + cl + ",";
            if (c.find(",comp,") != std::string::npos) r.classes |= 1;
            if (c.find(",round,") != std::string::npos) r.classes |= 2;
            if (c.find(",test,") != std::string::npos) r.classes |= 4;
        }
        if (!e || !*e) return r;
        r.on = true;
        std::string list = std::string(e) + ",";
        for (size_t at = 0; at < list.size();) {
            size_t comma = list.find(',', at);
            std::string tok = list.substr(at, comma - at);
            at = comma + 1;
            if (tok == "nonconst") r.nonconst = true;
            else if (!tok.empty()) r.vals.push_back(static_cast<uint32_t>(std::strtoul(tok.c_str(), nullptr, 16)));
        }
        return r;
    }();
    return f;
}

}  // namespace

// DKB_LQ_PATCH (see spirv_opt.h). Returns 1 patched (*out_ptr, *out_words, *applied), 0 not in the family or nothing
// to change, -1 failure (reason in err).
extern "C" int spirv_lq_patch_ex(const uint32_t *w, uint64_t n64, unsigned kinds, uint32_t **out_ptr, uint64_t *out_words,
                                 unsigned *applied, char *err, size_t cap, int force_vertex) {
    auto fail = [&](const std::string &why) {
        if (err && cap) std::snprintf(err, cap, "%s", why.c_str());
        return -1;
    };
    *out_ptr = nullptr;
    *applied = 0;
    size_t n = static_cast<size_t>(n64);
    if (!w || n < 5 || w[0] != 0x07230203u) return 0;
    try {
        // ---- analysis: the family match, what the instructions need, the loops
        bool fragment = false, vertex = false, matched = false, has_demote_cap = false, has_demote_ext = false;
        bool have_ep = false;
        uint32_t ep_fn = 0, int32_ty = 0;
        const unsigned cut_want = (kinds >> 25) & 7u;
        std::map<uint32_t, uint16_t> cut_ids;   // result id -> opcode, the selected cut
        std::map<uint32_t, uint32_t> cut_type;  // result id -> result type, for the ones the entry function defines as expected
        for (auto &c : vscut_table(cut_want)) cut_ids[c.first] = c.second;
        std::set<uint32_t> int_types;
        std::map<uint32_t, uint32_t> iconst;     // integer OpConstant id -> its first literal word
        uint32_t ext_glsl = 0;                   // the id of the GLSL.std.450 import
        // ubocomp analysis: types, the value type of ids, and the candidate access chains.
        std::map<uint32_t, uint32_t> int_width, arr_elem, mat_col, vtype;
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> ptr_of;   // (storage, pointee) -> pointer type
        std::map<uint32_t, UcChain> uc;                               // chain result id -> chain
        uint32_t bool_ty = 0, const_true = 0, const_false = 0;
        std::map<uint32_t, uint32_t> sra_shift;   // OpShiftRightArithmetic result -> its shift amount id
        std::map<uint32_t, And3> and3;             // OpBitwiseAnd with the constant 3, by result id
        std::map<uint32_t, std::array<uint32_t, 3>> cond_def;   // OpLogicalNot / integer compare / bitcast: (opcode, operand 1, operand 2)
        // rotexact analysis: every in-function value that is plain integer / bool / load arithmetic, and the stores to locals.
        std::map<uint32_t, std::array<uint32_t, 6>> gdef;   // id -> (opcode, type, operands 1..4)
        std::map<uint32_t, std::vector<LStore>> lstores;    // function-local variable -> its stores in program order
        std::vector<uint32_t> store_vals;                    // the value of every store in a function
        std::map<uint32_t, ChainAll> chain_all;              // every access chain in a function
        std::map<uint32_t, std::pair<uint32_t, bool>> arm_map;   // triangle arm label -> (condition, runs when the condition is true)
        std::pair<uint32_t, bool> cur_arm = {0, false};
        uint32_t pend_merge = 0;
        std::set<uint32_t> trig_types, and_types;   // result types of Sin / Cos and of OpBitwiseAnd, for notrig / noflags
        size_t entry_ins = 0;   // instructions in the entry function's body (labels included)
        std::vector<uint32_t> ep_iface;
        std::map<uint32_t, std::pair<uint32_t, uint32_t>> vars, ptrs, vec_info;   // var: (ptr type, storage); ptr: (storage, pointee); vec: (component, count)
        std::map<uint32_t, uint32_t> loc_of, pos_member, out_ptr_to;   // Location by id; struct -> BuiltIn Position member; pointee -> Output pointer
        std::map<uint32_t, std::vector<uint32_t>> struct_members;
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> const_by;    // (type, 32-bit value) -> OpConstant id
        std::set<uint32_t> pos_vars;
        std::set<uint32_t> entry_fns, c0_named, other_named, uniform_vars;
        std::map<uint32_t, uint32_t> consts;
        uint32_t f32 = 0, f32_zero = 0;
        Uses u;
        std::vector<Loop> loops;
        size_t kills_in_entry = 0, implicit_total = 0, cur_label = 0;
        uint32_t cur_fn = 0;
        std::vector<std::pair<uint32_t, size_t>> labels;        // (id, offset) of the current function's blocks
        std::vector<std::pair<size_t, uint32_t>> loop_merges;   // (header label offset, merge block id)
        for (size_t i = 5; i < n;) {
            uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
            if (!len || i + len > n) return 0;
            const uint32_t *a = w + i + 1;
            if (op == OpExtInstImport_ && len >= 3 && str_eq(a + 1, len - 2, "GLSL.std.450")) ext_glsl = a[0];
            if (op == OpExtInst_ && len >= 5 && ext_glsl && a[2] == ext_glsl && (a[3] == 13 || a[3] == 14)) trig_types.insert(a[0]);
            if (op == OpBitwiseAnd_ && len >= 5) and_types.insert(a[0]);
            if (cur_fn && cur_fn == ep_fn && op != OpFunction_ && op != OpFunctionEnd_) {
                entry_ins++;
                if (!cut_ids.empty() && len >= 4) {
                    auto c = cut_ids.find(a[1]);
                    if (c != cut_ids.end() && c->second == op) cut_type[a[1]] = a[0];
                }
            }
            if (cur_fn && len >= 3 && type_first(op)) vtype[a[1]] = a[0];
            if (op == 41 && len >= 3) const_true = a[1];    // OpConstantTrue
            if (op == 42 && len >= 3) const_false = a[1];   // OpConstantFalse
            if (cur_fn) {   // rotexact: values, arms, stores to locals
                if ((op == OpAccessChain_ || op == OpInBoundsAccessChain_) && len >= 5)
                    chain_all[a[1]] = {a[0], a[2], std::vector<uint32_t>(a + 3, a + len - 1)};
                if (op == OpSelectionMerge_ && len >= 2) pend_merge = a[0];
                else if (op == OpBranchConditional_ && len >= 4) {
                    if (pend_merge) {
                        if (a[1] == pend_merge && a[2] != pend_merge) arm_map[a[2]] = {a[0], false};
                        else if (a[2] == pend_merge && a[1] != pend_merge) arm_map[a[1]] = {a[0], true};
                        else if (a[1] != pend_merge && a[2] != pend_merge) arm_map[a[1]] = {a[0], true}, arm_map[a[2]] = {a[0], false};
                    }
                    pend_merge = 0;
                } else if (op == OpLabel_) {
                    auto am = arm_map.find(a[0]);
                    cur_arm = am == arm_map.end() ? std::make_pair(0u, false) : am->second;
                } else if (op == OpStore_ && len >= 3) {
                    store_vals.push_back(a[1]);
                    auto sv = vars.find(a[0]);
                    if (sv != vars.end() && sv->second.second == 7) lstores[a[0]].push_back({a[1], cur_arm.first, cur_arm.second});
                }
                if (len >= 4 && (op == 61 || op == 83 || op == 124 || (op >= 126 && op <= 133) || op == 168 || op == 169 ||
                                 (op >= 170 && op <= 179) || (op >= 194 && op <= 200) || op == OpExtInst_))
                    gdef[a[1]] = {op, a[0], a[2], len >= 5 ? a[3] : 0u, len >= 6 ? a[4] : 0u, len >= 7 ? a[5] : 0u};
            }
            if (op == 195 && len >= 5) sra_shift[a[1]] = a[3];   // OpShiftRightArithmetic: type, id, base, shift
            if (len >= 4 && (op == 168 || op == 124 || op == OpCopyObject_ || (op >= 170 && op <= 179)))
                cond_def[a[1]] = {op, a[2], len >= 5 ? a[3] : 0u};
            if (op == OpBitwiseAnd_ && len >= 5) {
                auto c2 = iconst.find(a[2]), c3 = iconst.find(a[3]);
                if (c3 != iconst.end() && c3->second == 3) and3[a[1]].other = a[2], and3[a[1]].type = a[0];
                else if (c2 != iconst.end() && c2->second == 3) and3[a[1]].other = a[3], and3[a[1]].type = a[0];
            }
            if (cur_fn && (op == OpAccessChain_ || op == OpInBoundsAccessChain_) && len >= 5) {
                auto bv = vars.find(a[2]);
                auto bp = bv == vars.end() ? ptrs.end() : ptrs.find(bv->second.first);
                if (bp != ptrs.end() && (bv->second.second == 2 || bv->second.second == 12)) {
                    uint32_t t = bp->second.second;
                    bool ok = true;
                    for (size_t k = 3; k + 1 < len - 1u && ok; k++) {   // the indices before the last one
                        auto sm = struct_members.find(t);
                        auto ae = arr_elem.find(t);
                        auto mc = mat_col.find(t);
                        auto ci = consts.find(a[k]);
                        if (sm != struct_members.end() && ci != consts.end() && ci->second < sm->second.size()) t = sm->second[ci->second];
                        else if (ae != arr_elem.end()) t = ae->second;
                        else if (mc != mat_col.end()) t = mc->second;
                        else ok = false;
                    }
                    const uint32_t last = a[len - 2];
                    auto vi = vec_info.find(t);
                    auto vt = vtype.find(last);
                    auto wd = vt == vtype.end() ? int_width.end() : int_width.find(vt->second);
                    if (ok && vi != vec_info.end() && vi->second.second >= 2 && vi->second.second <= 4 && !consts.count(last) &&
                        wd != int_width.end() && wd->second == 32) {
                        UcChain &c = uc[a[1]];
                        c.type = a[0], c.base = a[2], c.last = last, c.vec = t, c.storage = bv->second.second, c.idx_type = vt->second;
                        c.count = vi->second.second;
                        c.mid.assign(a + 3, a + len - 2);
                    }
                }
            }
            switch (op) {
            case 20: if (len >= 2) bool_ty = a[0]; break;   // OpTypeBool
            case 24: if (len >= 4) mat_col[a[0]] = a[1]; break;   // OpTypeMatrix
            case 28: case 29: if (len >= 4) arr_elem[a[0]] = a[1]; break;   // OpTypeArray, OpTypeRuntimeArray
            case OpEntryPoint_:
                if (len >= 3) {
                    if (a[0] == 4) fragment = true;
                    if (a[0] == 0) vertex = true;
                    entry_fns.insert(a[1]);
                    if (!have_ep && len >= 4) {
                        have_ep = true;
                        ep_fn = a[1];
                        size_t k = 2;   // the name's words end at the first one holding a zero byte
                        while (k < len - 1u) {
                            uint32_t x = a[k++];
                            if (!(x & 0xff) || !(x & 0xff00) || !(x & 0xff0000) || !(x & 0xff000000u)) break;
                        }
                        ep_iface.assign(a + k, a + len - 1);
                    }
                }
                break;
            case OpTypeInt_:
                if (len >= 4) int_types.insert(a[0]), int_width[a[0]] = a[1];
                if (len >= 4 && a[1] == 32 && !int32_ty) int32_ty = a[0];
                break;
            case OpTypeVector_: if (len >= 4) vec_info[a[0]] = {a[1], a[2]}; break;
            case OpTypeStruct_: if (len >= 2) struct_members[a[0]].assign(a + 1, a + len - 1); break;
            case OpTypePointer_:
                if (len >= 4) {
                    ptrs[a[0]] = {a[1], a[2]};
                    ptr_of.emplace(std::make_pair(a[1], a[2]), a[0]);
                    if (a[1] == 3 && !out_ptr_to.count(a[2])) out_ptr_to[a[2]] = a[0];
                }
                break;
            case OpName_: if (len >= 3) (str_eq(a + 1, len - 2, "fp_c0") ? c0_named : other_named).insert(a[0]); break;
            case OpExecutionMode_: if (len >= 3 && a[1] >= 5366 && a[1] <= 5371) u.interlock = true; break;
            case OpCapability_: if (a[0] == CapDemote_) has_demote_cap = true; break;
            case OpExtension_: if (str_eq(a, len - 1, "SPV_EXT_demote_to_helper_invocation")) has_demote_ext = true; break;
            case OpTypeFloat_:
                if (len >= 3) { if (a[1] == 64) u.f64 = true; if (a[1] == 32 && !f32) f32 = a[0]; }
                break;
            case OpConstant_:
                if (len >= 4) {
                    consts[a[1]] = a[2];
                    const_by.emplace(std::make_pair(a[0], a[2]), a[1]);
                    if (int_types.count(a[0])) iconst[a[1]] = a[2];
                    if (f32 && a[0] == f32 && a[2] == 0 && !f32_zero) f32_zero = a[1];
                }
                break;
            case OpVariable_:
                if (len >= 4) { vars[a[1]] = {a[0], a[2]}; if (a[2] == 2) uniform_vars.insert(a[1]); }
                break;
            case OpFunction_: cur_fn = a[1]; labels.clear(); loop_merges.clear(); break;
            case OpLabel_: cur_label = i; labels.push_back({a[0], i}); break;
            case OpLoopMerge_: loop_merges.push_back({cur_label, a[0]}); break;
            case OpFunctionEnd_:
                for (auto &lm : loop_merges)
                    for (auto &lb : labels)
                        if (lb.first == lm.second) loops.push_back({lm.first, lb.second});
                cur_fn = 0;
                break;
            case OpAccessChain_: case OpInBoundsAccessChain_:
                // fp_c0[0][83]: base, member 0, element 83. A module without names accepts any uniform variable.
                if (len >= 6 && uniform_vars.count(a[2]) && !other_named.count(a[2]) &&
                    (c0_named.count(a[2]) || c0_named.empty())) {
                    auto it = consts.find(a[4]);
                    if (it != consts.end() && it->second == 83) matched = true;
                }
                break;
            case OpKill_: if (entry_fns.count(cur_fn)) kills_in_entry++; break;
            case OpImageSampleImplicitLod_: case OpImageSampleDrefImplicitLod_: implicit_total++; break;
            case 5364: case 5365: u.interlock = true; break;
            case OpDecorate_: case OpMemberDecorate_: {
                size_t k = op == OpDecorate_ ? 1 : 2;
                if (op == OpDecorate_ && len >= 4 && a[1] == 30) loc_of[a[0]] = a[2];
                if (len >= 4 + (op == OpMemberDecorate_) && a[k] == 11 && a[k + 1] == 0) {
                    if (op == OpDecorate_) pos_vars.insert(a[0]); else pos_member[a[0]] = a[1];
                }
                if (len >= k + 3 && a[k] == 11) {
                    uint32_t b = a[k + 1];
                    if (b == 4424 || b == 4425 || b == 4426) u.draw_params = true;
                    if (b == 36 || b == 38 || b == 40 || b == 41) u.basic = true;
                    if (b >= 4416 && b <= 4421) u.basic = u.ballot = true;
                }
                break;
            }
            default:
                if (op >= 333 && op <= 362) {
                    u.basic = true;
                    if (op >= 334 && op <= 336) u.vote = true;
                    else if (op >= 337 && op <= 344) u.ballot = true;
                    else if (op == 345 || op == 346) u.shuffle = true;
                    else if (op == 347 || op == 348) u.shuffle_rel = true;
                    else if (op >= 349 && op <= 360) u.arith = true;
                    else if (op == 361 || op == 362) u.quad = true;
                } else if (op == 4431 || (op >= 5296 && op <= 5301)) {
                    u.keep_group = true;
                }
            }
            i += len;
        }
        bool loc9 = false;   // an Input variable decorated Location 9
        for (auto &v : vars) {
            auto l = loc_of.find(v.first);
            if (v.second.second == 1 && l != loc_of.end() && l->second == 9) loc9 = true;
        }
        const bool family = fragment && matched;
        unsigned eff = 0;
        if (family) eff |= kinds & 15u;
        eff |= kinds & (1u << 31);   // ubocomp: any module
        // ldcfix reuses bit 8, which is bit 0 of the vsmask mask: it is only read when bit 7 (vsmask) is clear. Any module.
        // selfix: bit 14 (likewise a mask bit of vsmask): ldcfix limited to the selection constructs named by DKB_LQ_SELFIX=<a>[-<b>][,...]
        // (module order from 0, as DKB_LQ_VSBLOCK counts them), with no requirement on the condition: the named construct only has to
        // be a triangle whose arm is one block of loads, pure arithmetic and stores of function-local variables. Exact for every input.
        const bool sf_req = !(kinds & 128u) && (kinds & 16384u);
        std::vector<std::pair<unsigned, unsigned>> sf_ranges;
        if (sf_req) {
            const char *e = std::getenv("DKB_LQ_SELFIX");
            if (!e || !*e) return fail("selfix: DKB_LQ_SELFIX is not set");
            while (*e) {
                char *end = nullptr;
                unsigned long lo = std::strtoul(e, &end, 10), hi = lo;
                if (end == e) return fail("selfix: DKB_LQ_SELFIX must be a comma list of <a> or <a>-<b>");
                if (*end == '-') {
                    const char *q = end + 1;
                    hi = std::strtoul(q, &end, 10);
                    if (end == q || hi < lo) return fail("selfix: DKB_LQ_SELFIX must be a comma list of <a> or <a>-<b>");
                }
                sf_ranges.emplace_back(static_cast<unsigned>(lo), static_cast<unsigned>(hi));
                if (*end == ',') e = end + 1;
                else if (*end) return fail("selfix: DKB_LQ_SELFIX must be a comma list of <a> or <a>-<b>");
                else e = end;
            }
        }
        // mul2: bit 15 (likewise a mask bit of vsmask). The selector r of the family's computed indices is
        // `cond ? ((P * Q + C) & 3) : 3` with P * Q built from 16-bit halves; only the low two bits matter, so the sum becomes
        // (P & 3) * (Q & 3) + (C & 3). Exact for every input. Any module; gfx_shader.c lq_mul2 names the stages. Fails closed: every
        // test-class &3 must come from such a select whose only users are &3 ANDs, else nothing changes.
        const bool do_mul2 = !(kinds & 128u) && (kinds & 32768u);
        if (do_mul2) eff |= 32768u;
        const bool exact_req = !(kinds & 128u) && (kinds & 4096u) && ((kinds >> 16) & 3u) == 3u && vertex && !fragment && loc9;
        const bool ldc_req = (!(kinds & 128u) && (kinds & 256u)) || sf_req || exact_req;
        if (ldc_req) eff |= 256u;
        // and3exact reuses bit 9 the same way (bit 8 is ldcfix): only read while vsmask (bit 7) is off. Any module.
        const bool ex3_req = !(kinds & 128u) && (kinds & 512u);
        if (ex3_req) eff |= 512u;
        // rotexact: bit 10, likewise.
        const bool rx_req = !(kinds & 128u) && (kinds & 1024u);
        if (rx_req) eff |= 1024u;
        // loadexact: bit 11, likewise; it does what rotexact does but loads only the picked word.
        const bool lx_req = !(kinds & 128u) && (kinds & 2048u);
        if (lx_req) eff |= 2048u;
        // force_vertex (DKB_LQ_PATCH_KEYS names the module): the vertex kinds apply without a Location 9 input.
        const bool vtx_ok = vertex && !fragment && (loc9 || force_vertex);
        if (vtx_ok) eff |= kinds & (1u | 16u | 32u | 64u | 128u | (7u << 25) | (15u << 28));
        // vsblock: bit 13 (a mask bit of vsmask, read only while bit 7 is clear); the range comes from DKB_LQ_VSBLOCK=<a>-<b>[:t|:f].
        // The selection constructs (OpSelectionMerge, in module order from 0) a..b have their branch condition replaced by a constant,
        // so they take one path only; dead-branch elimination then removes the other path. Without :t / :f a construct that
        // has an arm straight to its merge block (an `if` without else) takes that way, and a diamond takes its false target.
        unsigned vb_lo = 0, vb_hi = 0;
        int vb_mode = 0;   // 0 automatic, 1 true target, 2 false target
        bool do_vsblock = false;
        if (!(kinds & 128u) && (kinds & 8192u) && vtx_ok) {
            const char *e = std::getenv("DKB_LQ_VSBLOCK");
            char *end = nullptr;
            if (!e || !*e) return fail("vsblock: DKB_LQ_VSBLOCK is not set");
            unsigned long lo = std::strtoul(e, &end, 10), hi = 0;
            if (end == e || *end != '-') return fail("vsblock: DKB_LQ_VSBLOCK must be <a>-<b>[:t|:f]");
            const char *p = end + 1;
            hi = std::strtoul(p, &end, 10);
            if (end == p || lo > hi) return fail("vsblock: DKB_LQ_VSBLOCK must be <a>-<b>[:t|:f] with a <= b");
            if (*end == ':' && (end[1] == 't' || end[1] == 'f') && !end[2]) vb_mode = end[1] == 't' ? 1 : 2;
            else if (*end) return fail("vsblock: DKB_LQ_VSBLOCK must be <a>-<b>[:t|:f]");
            vb_lo = static_cast<unsigned>(lo), vb_hi = static_cast<unsigned>(hi);
            do_vsblock = true;
            eff |= 8192u;
        }
        // fold: bit 12, likewise (the production fix for the NVIDIA fault in two Ground light vertex stages; spirv_opt.h).
        // never forced, except by site `construct` (bit 18), whose pattern check is the named construct
        const bool do_fold = !(kinds & 128u) && (kinds & 4096u) && vertex && !fragment && (loc9 || (force_vertex && (kinds & 262144u)));
        if (do_fold) eff |= 1u << 30;
        // fold site (bits 16 / 17 with the fold; mask bits of vsmask, which the fold excludes): 1 = export, 2 = branch. The module has two
        // test-class &3 ANDs; exactly one reaches the condition of a branch (data flow through arithmetic, selects and locals), the other
        // reaches outputs only. Branch nulls the first, export nulls the second; anything else fails closed. Diagnostic (DKB_LQ_FOLD_SITE).
        // Bit 18 with the fold, site 4 (DKB_LQ_FOLD_SITE=construct, DKB_LQ_FOLD_CONSTRUCT=<n>): null every test-class &3 AND that reaches the
        // condition of selection construct n (counted as for vsblock), for modules with any number of them; fails closed if none does.
        const unsigned fold_site = do_fold ? ((kinds & 262144u) ? 4u : (kinds >> 16) & 3u) : 0;
        unsigned fold_construct = 0;
        if (fold_site == 4) {
            const char *e = std::getenv("DKB_LQ_FOLD_CONSTRUCT");   // overrides the number in bits 19..24 of kinds
            char *end = nullptr;
            const unsigned long v = e && *e ? std::strtoul(e, &end, 10) : 0;
            if (e && *e && *end) return fail("fold site construct: DKB_LQ_FOLD_CONSTRUCT must be a construct number");
            fold_construct = e && *e ? static_cast<unsigned>(v) : (kinds >> 19) & 63u;
        }
        if (!eff || !have_ep) return 0;
        const bool v16 = w[1] >= 0x10600;
        const bool do_caps = eff & 1, do_demote = (eff & 2) && kills_in_entry > 0,
                   do_lod = (eff & 4) && implicit_total > 0, do_null = eff & 8, do_cull = eff & 16, do_nullvs = eff & 32,
                   do_passvs = eff & 64;
        // A cut applies only to the captured shader. Any other matched vertex module gets mask 0 instead (every Location
        // output null, Position culled), so a cut run does not leave the other modules of the family running their slice.
        const bool cut_on = (eff >> 25) & 7u;
        const bool do_cut = cut_on && cut_type.size() == cut_ids.size();
        const bool cut_fallback = cut_on && !do_cut;
        const bool do_vsmask = (eff & 128) || cut_fallback;
        const bool do_flatten = eff & (1u << 28), do_notrig = eff & (1u << 29), do_noflags = eff & (1u << 30);
        const bool do_ubocomp = eff & (1u << 31);
        AndFilter fold_filter;   // the fold nulls the &3 ANDs of the test class and nothing else
        fold_filter.on = fold_filter.classes_on = true, fold_filter.vals = {3u}, fold_filter.classes = 4;
        const AndFilter &af_sel = do_fold ? fold_filter : and_filter();
        const bool do_and3exact = ex3_req, do_loadexact = lx_req, do_rotexact = rx_req || lx_req;
        const bool do_ldcfix = ldc_req;
        std::set<unsigned> exact_constructs, exact_done;   // fold site `exact`: the selection constructs whose condition reaches the selector
        unsigned and3_nulled[3] = {0, 0, 0}, and3_left = 0;
        if (((do_noflags && af_sel.classes_on) || do_and3exact || do_mul2) && !and3.empty()) {   // how each &3 result is used
            std::map<uint32_t, uint32_t> root_of;
            for (auto &kv : and3) root_of[kv.first] = kv.first;
            bool in_fn = false;
            for (size_t i = 5; i < n;) {
                const uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
                if (op == OpFunction_) in_fn = true;
                if (op == OpFunctionEnd_) in_fn = false;
                if (in_fn) {
                    if ((op == 124 || op == OpCopyObject_) && len >= 4) {   // OpBitcast / OpCopyObject carries the root
                        auto r = root_of.find(w[i + 3]);
                        if (r != root_of.end()) root_of[w[i + 2]] = r->second;
                    }
                    for (size_t k = 1; k < len; k++) {
                        auto r = root_of.find(w[i + k]);
                        if (r == root_of.end() || (k == 2 && type_first(op))) continue;
                        And3 &x = and3[r->second];
                        if ((op == OpAccessChain_ || op == OpInBoundsAccessChain_) && k == len - 1u) {
                            auto bv = vars.find(w[i + 3]);
                            if (bv != vars.end() && (bv->second.second == 2 || bv->second.second == 12)) x.comp = true;
                        } else if ((op >= 170 && op <= 179) || op == 169 || (op == 251 && k == 1)) {
                            x.test = true;
                        }
                    }
                }
                i += len;
            }
            for (auto &kv : and3) {
                And3 &x = kv.second;
                auto sh = sra_shift.find(x.other);
                auto sc = sh == sra_shift.end() ? iconst.end() : iconst.find(sh->second);
                x.cls = (sc != iconst.end() && sc->second == 31) ? 2 : x.comp ? 1 : x.test ? 4 : 0;
            }
        }
        std::set<uint32_t> fold_site_ids;   // AND ids the fold nulls when a site is chosen
        if (fold_site) {
            std::map<uint32_t, std::vector<uint32_t>> flow;
            std::map<uint32_t, uint32_t> chain_of;
            std::set<uint32_t> branch_cond, out_val;
            std::map<uint32_t, unsigned> branch_idx;   // branch condition id -> the selection construct it ends
            unsigned sel_counter = 0;
            bool sel_pending = false;
            unsigned sel_cur = 0;
            bool in_fn = false;
            for (size_t i = 5; i < n;) {
                const uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
                const uint32_t *a = w + i + 1;
                if (op == OpFunction_) in_fn = true;
                if (op == OpFunctionEnd_) in_fn = false;
                if (in_fn) {
                    if (op == OpSelectionMerge_) sel_cur = sel_counter++, sel_pending = true;
                    else if (op != OpBranchConditional_) sel_pending = false;
                    auto base_of = [&](uint32_t id) { auto c = chain_of.find(id); return c == chain_of.end() ? id : c->second; };
                    if (op == OpStore_ && len >= 3) {
                        const uint32_t base = base_of(a[0]);
                        auto v = vars.find(base);
                        if (v != vars.end() && v->second.second == 3) out_val.insert(a[1]);
                        else flow[a[1]].push_back(base);
                    } else if ((op == OpAccessChain_ || op == OpInBoundsAccessChain_) && len >= 5) {
                        chain_of[a[1]] = base_of(a[2]);
                        flow[a[2]].push_back(a[1]);
                        for (size_t k = 3; k + 1 < len; k++) flow[a[k]].push_back(a[1]);
                    } else if (op == OpBranchConditional_ || op == 251) {
                        branch_cond.insert(a[0]);
                        if (sel_pending) branch_idx[a[0]] = sel_cur;
                    } else if (type_first(op) && len >= 4) {
                        for (size_t k = 2; k + 1 < len; k++) flow[a[k]].push_back(a[1]);
                    }
                }
                i += len;
            }
            std::vector<uint32_t> to_branch, to_out;   // the test-class ANDs by what they reach
            std::set<uint32_t> reached_ids;           // everything the branch-reaching AND reaches
            std::map<uint32_t, std::set<unsigned>> and_constructs;   // test-class AND -> the selection constructs whose condition it reaches
            size_t tests = 0;
            for (auto &kv : and3) {
                if (kv.second.cls != 4) continue;
                tests++;
                std::set<uint32_t> seen;
                std::vector<uint32_t> work{kv.first};
                bool br = false, out = false;
                while (!work.empty()) {
                    const uint32_t v = work.back();
                    work.pop_back();
                    if (!seen.insert(v).second) continue;
                    br |= branch_cond.count(v) > 0, out |= out_val.count(v) > 0;
                    auto f = flow.find(v);
                    if (f != flow.end()) work.insert(work.end(), f->second.begin(), f->second.end());
                }
                for (uint32_t id : seen) { auto bi = branch_idx.find(id); if (bi != branch_idx.end()) and_constructs[kv.first].insert(bi->second); }
                if (br) to_branch.push_back(kv.first), reached_ids = seen;
                else if (out) to_out.push_back(kv.first);
            }
            if (fold_site == 4) {
                for (auto &kv : and_constructs)
                    if (kv.second.count(fold_construct)) fold_site_ids.insert(kv.first);
                if (fold_site_ids.empty()) return 0;   // fail closed: no test-class AND decides that construct
            } else if (tests != 2 || to_branch.size() != 1 || to_out.size() != 1) {
                return 0;   // fail closed: the two sites are not told apart
            }
            if (fold_site == 4) {
            } else if (fold_site == 3) {   // exact: no AND is nulled; the constructs the selector decides become selects (ldcfix's rewrite)
                for (uint32_t id : reached_ids) {
                    if (!branch_cond.count(id)) continue;
                    auto bi = branch_idx.find(id);
                    if (bi == branch_idx.end()) return 0;   // a branch that ends no selection construct (a loop): fail closed
                    exact_constructs.insert(bi->second);
                }
                if (exact_constructs.empty()) return 0;
            } else {
                fold_site_ids.insert(fold_site == 2 ? to_branch[0] : to_out[0]);
            }
        }
        std::map<size_t, LdcPlan> ldc_plan;      // offset of the header's OpSelectionMerge -> the plan
        std::map<size_t, size_t> ldc_skip;       // offset of the arm's OpLabel -> the offset after the arm block
        if (do_ldcfix) {
            // Find the triangles `OpSelectionMerge M; OpBranchConditional c ARM M` (or with the arms swapped) where c is a compare of
            // an `x & 3` (the low two bits of an index), ARM is one block of loads / pure ALU / stores of function-local variables
            // only, and nothing else reaches ARM or needs a phi in M. Such a branch is replaced by its arm run unconditionally with
            // each store turned into a select of the new and the old value: the same maths without the data-dependent branch.
            struct Range { size_t label, begin, term, end; };
            std::map<uint32_t, Range> labs;
            std::map<uint32_t, unsigned> refs;
            bool in_fn = false;
            uint32_t cur = 0;
            for (size_t i = 5; i < n;) {
                const uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
                if (op == OpFunction_) in_fn = true;
                if (op == OpFunctionEnd_) in_fn = false;
                if (in_fn) {
                    if (op == OpLabel_) cur = w[i + 1], labs[cur] = {i, i + len, 0, 0};
                    else if (op >= 249 && op <= 255 && cur) labs[cur].term = i, labs[cur].end = i + len;
                    for (size_t k = 1; k < len; k++) refs[w[i + k]]++;
                }
                i += len;
            }
            auto unwrap = [&](uint32_t id) {
                for (int d = 0; d < 4; d++) {
                    auto it = cond_def.find(id);
                    if (it == cond_def.end() || (it->second[0] != 124 && it->second[0] != OpCopyObject_)) break;
                    id = it->second[1];
                }
                return id;
            };
            auto cond_is_and3 = [&](uint32_t c) {
                for (int d = 0; d < 6; d++) {
                    auto it = cond_def.find(c);
                    if (it == cond_def.end()) return false;
                    if (it->second[0] == 168 || it->second[0] == 124 || it->second[0] == OpCopyObject_) { c = it->second[1]; continue; }
                    if (it->second[0] >= 170 && it->second[0] <= 179)
                        return and3.count(unwrap(it->second[1])) > 0 || and3.count(unwrap(it->second[2])) > 0;
                    return false;
                }
                return false;
            };
            auto arm_op_ok = [&](size_t k) {
                const uint32_t op = w[k] & 0xffff, len = w[k] >> 16;
                if (op == OpLoad_) {
                    auto v = vars.find(w[k + 3]);
                    return len == 4 && v != vars.end() && v->second.second == 7;
                }
                if (op == OpStore_) {
                    auto v = vars.find(w[k + 1]);
                    return len == 3 && v != vars.end() && v->second.second == 7 && ptrs.count(v->second.first);
                }
                return op == OpExtInst_ || op == OpCopyObject_ || op == 124 || (op >= 126 && op <= 133) || op == 168 || op == 169 ||
                       (op >= 170 && op <= 179) || (op >= 194 && op <= 200);
            };
            in_fn = false;
            unsigned sel_index = 0;
            for (size_t i = 5; i < n;) {
                const uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
                if (op == OpFunction_) in_fn = true;
                if (op == OpFunctionEnd_) in_fn = false;
                const size_t j = i + len;
                bool sf_named = !sf_req && !exact_req;
                if (in_fn && op == OpSelectionMerge_) {
                    for (auto &r : sf_ranges) sf_named |= sel_index >= r.first && sel_index <= r.second;
                    if (exact_req) sf_named |= exact_constructs.count(sel_index) > 0;
                    sel_index++;
                }
                if (in_fn && sf_named && op == OpSelectionMerge_ && j < n && (w[j] & 0xffff) == OpBranchConditional_ && (w[j] >> 16) == 4) {
                    const uint32_t merge = w[i + 1], cond = w[j + 1], t = w[j + 2], f = w[j + 3];
                    const uint32_t arm = t == merge ? f : f == merge ? t : 0;
                    auto ar = labs.find(arm), mr = labs.find(merge);
                    bool ok = arm && t != f && ar != labs.end() && mr != labs.end() && ar->second.term &&
                              (w[ar->second.term] & 0xffff) == 249 && w[ar->second.term + 1] == merge &&
                              refs[arm] == 2 && (w[mr->second.begin] & 0xffff) != 245 && (sf_req || exact_req || cond_is_and3(cond));
                    for (size_t k = ok ? ar->second.begin : 0; ok && k < ar->second.term; k += w[k] >> 16) ok = arm_op_ok(k);
                    if (ok) {
                        LdcPlan &p = ldc_plan[i];
                        p.merge = merge, p.cond = cond, p.arm_true = arm == t;
                        p.arm_begin = ar->second.begin, p.arm_term = ar->second.term, p.sm_len = len, p.bc_len = w[j] >> 16;
                        ldc_skip[ar->second.label] = ar->second.end;
                        if (exact_req) exact_done.insert(sel_index - 1);
                    }
                }
                i += len;
            }
            if (exact_req && exact_done != exact_constructs) return 0;   // a construct the selector decides could not be made branch-free
        }
        struct Mul2 { uint32_t sum = 0, v = 0, p = 0, q = 0, c = 0, type = 0, c3 = 0; };   // p == 0: V is used as it is
        std::map<uint32_t, uint32_t> mul2_and;   // &3 AND id -> the select it reads
        std::map<uint32_t, Mul2> mul2_plan;   // select id -> the sum it picks and the operands of its product
        if (do_mul2) {
            auto gd = [&](uint32_t id) -> const std::array<uint32_t, 6> * {
                auto it = gdef.find(id);
                return it == gdef.end() ? nullptr : &it->second;
            };
            auto cval = [&](uint32_t id, uint32_t want) { auto it = iconst.find(id); return it != iconst.end() && it->second == want; };
            // lo(x) = x & 0xffff, hi(x) = x >> 16 (logical): the source x, or 0
            auto lo_of = [&](uint32_t id) -> uint32_t {
                auto d = gd(id);
                if (!d || (*d)[0] != OpBitwiseAnd_) return 0;
                return cval((*d)[3], 65535) ? (*d)[2] : cval((*d)[2], 65535) ? (*d)[3] : 0;
            };
            // the select an AND reads, also through a load of a function-local variable stored once and unconditionally
            auto resolve = [&](uint32_t id) {
                for (int d = 0; d < 3; d++) {
                    auto it = gdef.find(id);
                    if (it == gdef.end() || it->second[0] != 61) break;
                    auto ls = lstores.find(it->second[2]);
                    if (ls == lstores.end() || ls->second.size() != 1 || ls->second[0].cond) break;
                    id = ls->second[0].value;
                }
                return id;
            };
            for (auto &kv : and3) {
                if (kv.second.cls != 4) continue;
                const uint32_t sel = resolve(kv.second.other);
                if (mul2_plan.count(sel)) continue;
                auto sd = gd(sel);
                if (!sd || (*sd)[0] != 169 || !cval((*sd)[4], 0xffffffffu)) continue;
                auto sum = gd((*sd)[3]);
                if (!sum || (*sum)[0] != 128) continue;
                // sum = shl(X, 16) + V in either operand order: the shifted term has no bits below 16, so the low two bits are V's. V is
                // usually lo(P) * lo(Q) + C (then the product is reduced to the low two bits of its operands) or any other value.
                uint32_t shl_id = 0, low_id = 0;
                for (int k = 0; k < 2; k++) {
                    const uint32_t u = (*sum)[2 + k], v = (*sum)[3 - k];
                    auto ud = gd(u);
                    if (ud && (*ud)[0] == 196 && cval((*ud)[3], 16)) shl_id = u, low_id = v;
                }
                if (!shl_id) continue;
                Mul2 m;
                m.sum = (*sd)[3], m.v = low_id, m.type = and3[kv.first].type;
                auto low = gd(low_id);
                if (low && (*low)[0] == 128) {
                    for (int k = 0; k < 2 && !m.p; k++) {
                        auto md = gd((*low)[2 + k]);
                        if (md && (*md)[0] == 132 && lo_of((*md)[2]) && lo_of((*md)[3])) m.p = lo_of((*md)[2]), m.q = lo_of((*md)[3]), m.c = (*low)[3 - k];
                    }
                }
                mul2_plan[sel] = m;
            }
            // the constant 3 of the ANDs' type (the new ANDs reuse it); the select itself stays for its other users
            std::map<uint32_t, uint32_t> c3_of_type;
            for (size_t i = 5; i < n;) {
                const uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
                if (op == OpBitwiseAnd_ && len == 5 && (cval(w[i + 3], 3) || cval(w[i + 4], 3))) c3_of_type[w[i + 1]] = cval(w[i + 3], 3) ? w[i + 3] : w[i + 4];
                i += len;
            }
            for (auto &kv : mul2_plan) kv.second.c3 = c3_of_type.count(kv.second.type) ? c3_of_type[kv.second.type] : 0;
            // every &3 AND that reads a planned select, directly or through such a local, reads the twin instead
            for (auto &kv : and3)
                if (mul2_plan.count(resolve(kv.second.other))) mul2_and[kv.first] = resolve(kv.second.other);
            size_t tests = 0, covered = 0;
            for (auto &kv : and3)
                if (kv.second.cls == 4) tests++, covered += mul2_plan.count(resolve(kv.second.other));
            for (auto &kv : mul2_plan) if (!kv.second.c3) covered = 0;
            if (!tests || covered != tests) mul2_plan.clear();   // fail closed
            if (mul2_plan.empty()) return 0;
        }
        unsigned uc_store_chains = 0, uc_loads = 0;
        if (do_ubocomp && !uc.empty()) {   // how each candidate chain is used: loads are rewritten, anything else stays
            bool in_fn = false;
            for (size_t i = 5; i < n;) {
                const uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
                if (op == OpFunction_) in_fn = true;
                if (op == OpFunctionEnd_) in_fn = false;
                if (in_fn)
                    for (size_t k = 1; k < len; k++) {
                        auto it = uc.find(w[i + k]);
                        if (it == uc.end()) continue;
                        if ((op == OpAccessChain_ || op == OpInBoundsAccessChain_) && k == 2) continue;   // its definition
                        if (op == OpLoad_ && k == 3) it->second.loads++;
                        else if (op == OpStore_ && k == 1) it->second.other++, it->second.stores++;
                        else if (op >= 227 && op <= 242 && k == 3) it->second.other++, it->second.stores++;
                        else if ((op == 63 || op == 64) && k <= 2) it->second.other++, it->second.stores++;
                        else it->second.other++;
                    }
                i += len;
            }
            for (auto &c : uc) uc_store_chains += c.second.stores > 0;
        }
        const uint32_t vmask = cut_fallback ? 0u : (kinds >> 8) & 0x1ffffu;   // vsmask: bit k keeps Output Location k, bit 16 keeps Position
        const bool mask_pos = do_vsmask && !(vmask & 0x10000u);

        auto cap_removed = [&](uint32_t c) {
            switch (c) {
            case 5363: case 5378: case 5372: return !u.interlock;
            case 10: return !u.f64;
            case 4427: return !u.draw_params;
            case 61: return !u.keep_group && !u.basic;
            case 62: return !u.keep_group && !u.vote;
            case 64: return !u.keep_group && !u.ballot;
            case 65: return !u.keep_group && !u.shuffle;
            case 66: return !u.keep_group && !u.shuffle_rel;
            case 63: case 67: return !u.keep_group && !u.arith;
            case 68: return !u.keep_group && !u.quad;
            default: return false;
            }
        };

        // ---- rewrite
        std::vector<uint32_t> o;
        o.reserve(n + 64);
        o.insert(o.end(), w, w + 5);
        bool caps_done = false, exts_done = false, interlock_cap_left = false, dp_cap_left = false, want_zero = false;
        unsigned done = 0;
        uint32_t bound = w[3];
        // New global declarations (inserted before the first function), and for nullfs / nullvs / passvs the body of the
        // new entry function (everything between its OpLabel and its OpReturn).
        std::vector<uint32_t> decl, body;
        uint32_t new_fn = 0, pos_var = 0, pos_ac_ptr = 0, pos_idx_const = 0, pos_value = 0, pos_idx = ~0u;
        const bool new_entry = do_null || do_nullvs || do_passvs;
        std::map<uint32_t, uint32_t> null_of;
        auto get_null = [&](uint32_t type) {
            uint32_t &nid = null_of[type];
            if (!nid) {
                nid = bound++;
                decl.insert(decl.end(), {3u << 16 | OpConstantNull_, type, nid});
            }
            return nid;
        };
        if (new_entry) new_fn = bound++;
        if (do_vsblock) {
            if (!bool_ty) { bool_ty = bound++; decl.insert(decl.end(), {2u << 16 | 20u, bool_ty}); }
            if (!const_true) { const_true = bound++; decl.insert(decl.end(), {3u << 16 | 41u, bool_ty, const_true}); }
            if (!const_false) { const_false = bound++; decl.insert(decl.end(), {3u << 16 | 42u, bool_ty, const_false}); }
        }
        if (do_cull || do_nullvs || do_passvs || mask_pos) {
            bool member = false;
            uint32_t idx = 0, vec4 = 0;
            for (auto &v : vars) {
                if (v.second.second != 3) continue;
                auto p = ptrs.find(v.second.first);
                if (p == ptrs.end()) continue;
                uint32_t pointee = p->second.second;
                if (pos_vars.count(v.first)) { pos_var = v.first; vec4 = pointee; break; }
                auto pm = pos_member.find(pointee);
                auto sm = struct_members.find(pointee);
                if (pm != pos_member.end() && sm != struct_members.end() && pm->second < sm->second.size()) {
                    pos_var = v.first; member = true; idx = pm->second; vec4 = sm->second[idx];
                    break;
                }
            }
            auto vi = vec_info.find(vec4);
            if (!pos_var || vi == vec_info.end() || vi->second.second != 4) return fail("no vec4 BuiltIn Position output");
            const uint32_t flt = vi->second.first;
            auto get_const = [&](uint32_t type, uint32_t value) {
                auto it = const_by.find({type, value});
                if (it != const_by.end()) return it->second;
                uint32_t id = bound++;
                decl.insert(decl.end(), {4u << 16 | OpConstant_, type, id, value});
                const_by[{type, value}] = id;
                return id;
            };
            if (member) {
                pos_idx = idx;
                if (!int32_ty) { int32_ty = bound++; decl.insert(decl.end(), {4u << 16 | OpTypeInt_, int32_ty, 32, 0}); }
                auto po = out_ptr_to.find(vec4);
                if (po != out_ptr_to.end()) pos_ac_ptr = po->second;
                else { pos_ac_ptr = bound++; decl.insert(decl.end(), {4u << 16 | OpTypePointer_, pos_ac_ptr, 3, vec4}); }
                pos_idx_const = get_const(int32_ty, idx);
            }
            uint32_t c0 = get_const(flt, 0), c1 = get_const(flt, 0x3f800000u), c2 = get_const(flt, 0x40000000u);
            pos_value = bound++;
            decl.insert(decl.end(), {7u << 16 | OpConstantComposite_, vec4, pos_value, c0, c0, c2, c1});
        }
        // Stores vec4(0, 0, 2, 1) into Position (through an access chain when Position is a block member) at the end of v.
        auto store_position = [&](std::vector<uint32_t> &v) {
            uint32_t target = pos_var;
            if (pos_ac_ptr) {
                target = bound++;
                v.insert(v.end(), {5u << 16 | OpAccessChain_, pos_ac_ptr, target, pos_var, pos_idx_const});
            }
            v.insert(v.end(), {3u << 16 | OpStore_, target, pos_value});
        };
        auto pointee_of = [&](uint32_t var) { return ptrs[vars[var].first].second; };
        // vsmask: the Location outputs to clear, and the pointers whose original stores are dropped (a store that stays would
        // keep the code feeding it alive through DCE).
        std::vector<std::pair<uint32_t, uint32_t>> mask_nulls;   // (Output variable, OpConstantNull id)
        std::set<uint32_t> dead_ptrs;
        std::map<uint32_t, uint32_t> cut_null;   // cut result id -> its OpConstantNull
        if (do_cut) for (auto &c : cut_type) cut_null[c.first] = get_null(c.second);
        std::map<uint32_t, uint32_t> trig_null, and_null;   // result type -> OpConstantNull, for notrig / noflags
        if (do_notrig) for (uint32_t t : trig_types) trig_null[t] = get_null(t);
        if (do_noflags) for (uint32_t t : and_types) and_null[t] = get_null(t);
        if (!ldc_plan.empty()) done |= sf_req ? 1u << 20 : 1u << 14;
        // and3exact: the test-class `x & 3` becomes x - ((x >> 2) << 2) (the same value for every 32-bit x), and a compare of such a
        // value with a constant becomes a OpVectorExtractDynamic from a constant bool vector listing the compare for r = 0..3 (r is
        // always in 0..3, so the extract is always in range): no OpBitwiseAnd and no compare is left at those sites.
        struct CmpPlan { uint32_t and_id, table; };
        std::map<uint32_t, CmpPlan> ex3_cmp;            // compare result id -> plan
        std::map<uint32_t, uint32_t> ex3_c2, ex3_table;   // type -> constant 2; 4-bit table -> OpConstantComposite id
        uint32_t ex3_bvec = 0;
        unsigned ex3_ands = 0;
        if (do_and3exact) {
            auto uconst = [&](uint32_t type, uint32_t value) {
                auto ci = const_by.find({type, value});
                if (ci != const_by.end()) return ci->second;
                const uint32_t id = bound++;
                decl.insert(decl.end(), {4u << 16 | OpConstant_, type, id, value});
                const_by[{type, value}] = id;
                return id;
            };
            for (auto &kv : and3)
                if (kv.second.cls == 4 && kv.second.type) {
                    if (!ex3_c2.count(kv.second.type)) ex3_c2[kv.second.type] = uconst(kv.second.type, 2);
                    ex3_ands++;
                }
            for (auto &cd : cond_def) {
                const unsigned cop = cd.second[0];
                if (cop < 170 || cop > 179) continue;
                const uint32_t x = cd.second[1], y = cd.second[2];
                auto unwrap_bc = [&](uint32_t id) {   // through bitcasts / copies
                    for (int d = 0; d < 4; d++) {
                        auto it = cond_def.find(id);
                        if (it == cond_def.end() || (it->second[0] != 124 && it->second[0] != OpCopyObject_)) break;
                        id = it->second[1];
                    }
                    return id;
                };
                const uint32_t ux = unwrap_bc(x), uy = unwrap_bc(y);
                auto ax = and3.find(ux), ay = and3.find(uy);
                const bool x_is = ax != and3.end() && ax->second.cls == 4, y_is = ay != and3.end() && ay->second.cls == 4;
                if (x_is == y_is) continue;   // neither, or both
                const uint32_t and_id = x_is ? ux : uy, other = x_is ? y : x;
                auto oc = iconst.find(other);
                if (oc == iconst.end()) continue;
                const int64_t cu = oc->second, cs = static_cast<int32_t>(oc->second);
                uint32_t table = 0;
                for (int64_t k = 0; k < 4; k++) {
                    const bool sgn = cop == 173 || cop == 175 || cop == 177 || cop == 179;
                    const int64_t c = sgn ? cs : cu, lhs = x_is ? k : c, rhs = x_is ? c : k;
                    bool v = false;
                    switch (cop) {
                    case 170: v = lhs == rhs; break;
                    case 171: v = lhs != rhs; break;
                    case 172: case 173: v = lhs > rhs; break;
                    case 174: case 175: v = lhs >= rhs; break;
                    case 176: case 177: v = lhs < rhs; break;
                    default: v = lhs <= rhs; break;
                    }
                    table |= v ? 1u << k : 0u;
                }
                ex3_cmp[cd.first] = {and_id, table};
            }
            if (!ex3_cmp.empty()) {
                if (!const_true) { const_true = bound++; decl.insert(decl.end(), {3u << 16 | 41u, bool_ty, const_true}); }
                if (!const_false) { const_false = bound++; decl.insert(decl.end(), {3u << 16 | 42u, bool_ty, const_false}); }
                for (auto &vi : vec_info)
                    if (vi.second.first == bool_ty && vi.second.second == 4) ex3_bvec = vi.first;
                if (!ex3_bvec) { ex3_bvec = bound++; decl.insert(decl.end(), {4u << 16 | OpTypeVector_, ex3_bvec, bool_ty, 4}); }
                for (auto &cp : ex3_cmp)
                    if (!ex3_table.count(cp.second.table)) {
                        const uint32_t id = bound++;
                        decl.insert(decl.end(), {7u << 16 | OpConstantComposite_, ex3_bvec, id});
                        for (unsigned k = 0; k < 4; k++) decl.push_back(cp.second.table >> k & 1 ? const_true : const_false);
                        ex3_table[cp.second.table] = id;
                    }
            }
            if (ex3_ands) done |= 1u << 15;
        }
        // rotexact. The translated light shaders pick one of four words by r = prod & 3 with one-hot weights: each word is multiplied by a
        // weight local that is 1 for exactly one r (0 otherwise) through a 32-bit multiply built from 16-bit halves, and the products are
        // added. This finds such sums (every weight a function of r alone, evaluated here for r = 0..3, each 0 or 1, at most one 1 per r) and
        // replaces each by the pick itself: OpVectorExtractDynamic of the four words' bit patterns at r (0 where no weight is 1), plus any
        // plain int addend. The integer arithmetic copies the picked word's bits, so the pick is bit-identical, NaN and Inf payloads included.
        std::map<uint32_t, RotPlan> rot_plan;   // root IAdd id -> plan
        uint32_t rot_ivec4 = 0, rot_zero = 0, lx_j[4] = {0, 0, 0, 0};
        std::map<uint32_t, uint32_t> lx_s;   // root of a shared sum -> the unmasked picked word's bits
        std::string rot_words;
        if (do_rotexact && !and3.empty()) {
            struct Ev { bool ok; uint32_t v; };
            const Ev bad{false, 0};
            uint32_t prod = 0, r_id = 0;   // the integer all the `& 3` in the weights test, and one of the AND ids
            std::function<Ev(uint32_t, uint32_t, int)> ev;
            auto eval_local = [&](uint32_t var, uint32_t r, int depth) -> Ev {
                auto it = lstores.find(var);
                if (it == lstores.end() || it->second.empty() || it->second.size() > 2 || it->second[0].cond) return bad;
                Ev cur = ev(it->second[0].value, r, depth + 1);
                if (!cur.ok) return bad;
                for (size_t k = 1; k < it->second.size(); k++) {
                    const LStore &st = it->second[k];
                    if (!st.cond) { cur = ev(st.value, r, depth + 1); if (!cur.ok) return bad; continue; }
                    Ev c = ev(st.cond, r, depth + 1), nv = ev(st.value, r, depth + 1);
                    if (!c.ok || !nv.ok) return bad;
                    if ((c.v != 0) == st.when_true) cur = nv;
                }
                return cur;
            };
            ev = [&](uint32_t id, uint32_t r, int depth) -> Ev {
                if (depth > 40) return bad;
                auto ic = iconst.find(id);
                if (ic != iconst.end()) return {true, ic->second};
                if (id == const_true && id) return {true, 1};
                if (id == const_false && id) return {true, 0};
                auto a3 = and3.find(id);
                if (a3 != and3.end()) {   // x & 3
                    if (prod && prod != a3->second.other) return bad;
                    prod = a3->second.other, r_id = id;
                    return {true, r};
                }
                auto gd = gdef.find(id);
                if (gd == gdef.end()) return bad;
                const auto &g = gd->second;
                auto sub = [&](uint32_t x) { return ev(x, r, depth + 1); };
                const unsigned op = g[0];
                if (op == 61) {
                    auto v = vars.find(g[2]);
                    return v != vars.end() && v->second.second == 7 ? eval_local(g[2], r, depth + 1) : bad;
                }
                if (op == 124 || op == OpCopyObject_) return sub(g[2]);
                if (op == 126) { Ev x = sub(g[2]); return x.ok ? Ev{true, 0u - x.v} : bad; }
                if (op == 168) { Ev x = sub(g[2]); return x.ok ? Ev{true, x.v ? 0u : 1u} : bad; }
                if (op == 169) {
                    Ev c = sub(g[2]), x = sub(g[3]), y = sub(g[4]);
                    return c.ok && x.ok && y.ok ? Ev{true, c.v ? x.v : y.v} : bad;
                }
                if (op == OpExtInst_) {   // GLSL.std.450 SMax / SMin of two ints
                    if (g[3] != 42 && g[3] != 39) return bad;
                    Ev x = sub(g[4]), y = sub(g[5]);
                    if (!x.ok || !y.ok) return bad;
                    const int32_t p = static_cast<int32_t>(x.v), q = static_cast<int32_t>(y.v);
                    return {true, static_cast<uint32_t>(g[3] == 42 ? std::max(p, q) : std::min(p, q))};
                }
                if (op == 128 || op == 130 || op == 132 || op == 197 || op == 198 || op == 199 || (op >= 170 && op <= 179)) {
                    Ev x = sub(g[2]), y = sub(g[3]);
                    if (!x.ok || !y.ok) return bad;
                    const int32_t sx = static_cast<int32_t>(x.v), sy = static_cast<int32_t>(y.v);
                    switch (op) {
                    case 128: return {true, x.v + y.v};
                    case 130: return {true, x.v - y.v};
                    case 132: return {true, x.v * y.v};
                    case 197: return {true, x.v | y.v};
                    case 198: return {true, x.v ^ y.v};
                    case 199: return {true, x.v & y.v};
                    case 170: return {true, x.v == y.v};
                    case 171: return {true, x.v != y.v};
                    case 172: return {true, x.v > y.v};
                    case 173: return {true, sx > sy};
                    case 174: return {true, x.v >= y.v};
                    case 175: return {true, sx >= sy};
                    case 176: return {true, x.v < y.v};
                    case 177: return {true, sx < sy};
                    case 178: return {true, x.v <= y.v};
                    default: return {true, sx <= sy};
                    }
                }
                return bad;
            };
            auto def_of = [&](uint32_t id) -> const std::array<uint32_t, 6> * {
                auto it = gdef.find(id);
                return it == gdef.end() ? nullptr : &it->second;
            };
            auto local_of_load = [&](uint32_t id) -> uint32_t {   // the function-local variable an OpLoad reads, else 0
                const auto *g = def_of(id);
                if (!g || (*g)[0] != 61) return 0;
                auto v = vars.find((*g)[2]);
                return v != vars.end() && v->second.second == 7 ? (*g)[2] : 0;
            };
            auto and16_of = [&](uint32_t id) -> uint32_t {   // x for `x & 0xffff`
                const auto *g = def_of(id);
                if (!g || (*g)[0] != 199) return 0;
                auto c0 = iconst.find((*g)[2]), c1 = iconst.find((*g)[3]);
                if (c1 != iconst.end() && c1->second == 0xffffu) return (*g)[2];
                if (c0 != iconst.end() && c0->second == 0xffffu) return (*g)[3];
                return 0;
            };
            auto srl16_of = [&](uint32_t id) -> uint32_t {   // x for `x >> 16`
                const auto *g = def_of(id);
                if (!g || (*g)[0] != 194) return 0;
                auto c = iconst.find((*g)[3]);
                return c != iconst.end() && c->second == 16 ? (*g)[2] : 0;
            };
            // Keys: a function-local variable (high bit set) when the value is a load of one, else the value id itself (the local-single-block
            // pass forwards a stored value to the loads in its block, so a weight or a word is a local or a plain value).
            auto key_of = [&](uint32_t id) -> uint32_t { const uint32_t v = local_of_load(id); return v ? (v | 0x80000000u) : id; };
            auto word_key = [&](uint32_t id) -> uint32_t {   // the word under Bitcast(float word)
                const auto *g = def_of(id);
                return g && (*g)[0] == 124 ? key_of((*g)[2]) : 0;
            };
            // (W key, F key) of IMul(a, b) where a, b are the given half selectors of a weight and a word bitcast, either order
            auto mul_pair = [&](uint32_t id, bool w_lo, bool f_lo) -> std::pair<uint32_t, uint32_t> {
                const auto *g = def_of(id);
                if (!g || (*g)[0] != 132) return {0, 0};
                for (int swap = 0; swap < 2; swap++) {
                    const uint32_t a = (*g)[swap ? 3 : 2], b = (*g)[swap ? 2 : 3];
                    const uint32_t wx = w_lo ? and16_of(a) : srl16_of(a), fx = f_lo ? and16_of(b) : srl16_of(b);
                    const uint32_t fk = fx ? word_key(fx) : 0;
                    if (wx && fk) return {key_of(wx), fk};
                }
                return {0, 0};
            };
            std::function<void(uint32_t, std::vector<uint32_t> &)> flatten = [&](uint32_t id, std::vector<uint32_t> &out) {
                const auto *g = def_of(id);
                if (g && (*g)[0] == 128 && out.size() < 64) { flatten((*g)[2], out); flatten((*g)[3], out); }
                else out.push_back(id);
            };
            std::set<uint32_t> tried;
            for (uint32_t stored : store_vals) {
                struct { uint32_t value; } st = {stored};
                {
                    const auto *top = def_of(st.value);
                    if (!top || (*top)[0] != 128 || !tried.insert(st.value).second) continue;
                    std::vector<uint32_t> leaves;
                    flatten(st.value, leaves);
                    std::map<std::pair<uint32_t, uint32_t>, int> seen;   // (W, F) -> bit 0 low product, bit 1 cross term
                    std::vector<uint32_t> extras;
                    bool ok = leaves.size() >= 2;
                    for (uint32_t lf : leaves) {
                        const auto *g = def_of(lf);
                        std::pair<uint32_t, uint32_t> pr = mul_pair(lf, true, true);
                        if (pr.first) { seen[pr] |= 1; continue; }
                        if (g && (*g)[0] == 196) {   // (a + b) << 16, a = Wlo * Fhi, b = Whi * Flo
                            auto c = iconst.find((*g)[3]);
                            const auto *s = def_of((*g)[2]);
                            if (c != iconst.end() && c->second == 16 && s && (*s)[0] == 128) {
                                bool hit = false;
                                for (int swap = 0; swap < 2 && !hit; swap++) {
                                    auto p1 = mul_pair((*s)[swap ? 3 : 2], true, false), p2 = mul_pair((*s)[swap ? 2 : 3], false, true);
                                    if (p1.first && p1 == p2) { seen[p1] |= 2; hit = true; }
                                }
                                if (hit) continue;
                            }
                        }
                        extras.push_back(lf);   // anything else stays in the sum as it is
                    }
                    if (!ok || seen.empty()) continue;
                    for (auto &s : seen) if (s.second != 3) ok = false;
                    if (!ok) continue;
                    // the weights, for r = 0..3
                    RotPlan plan;
                    plan.type = (*top)[1];
                    plan.extras = extras;
                    prod = 0, r_id = 0;
                    for (auto &s : seen) {
                        for (uint32_t r = 0; r < 4 && ok; r++) {
                            Ev w = s.first.first & 0x80000000u ? eval_local(s.first.first & 0x7fffffffu, r, 0) : ev(s.first.first, r, 0);
                            if (!w.ok || w.v > 1) { ok = false; break; }
                            if (w.v) {
                                if (plan.slot_key[r]) { ok = false; break; }   // two words hot at once: not a pick
                                plan.slot_key[r] = s.first.second;
                            }
                        }
                        if (!ok) break;
                    }
                    if (!ok || !r_id) continue;
                    plan.r_id = r_id;
                    rot_plan[st.value] = plan;
                    for (auto &s : seen) rot_words += " %" + std::to_string(s.first.second & 0x7fffffffu) + (s.first.second & 0x80000000u ? "" : "(value)");
                }
            }
            if (do_loadexact && !rot_plan.empty()) {
                // The four words are loaded from buffer[lead][x >> 2][x & 3] with x one of four index values. Find x for every word, then
                // (below) load only the word whose index the weight selects.
                auto canon_idx = [&](uint32_t id) -> uint32_t {   // through bitcasts; a load of a local is that local
                    for (int d = 0; d < 6; d++) {
                        const auto *g = def_of(id);
                        if (g && ((*g)[0] == 124 || (*g)[0] == OpCopyObject_)) id = (*g)[2];
                        else break;
                    }
                    const uint32_t v = local_of_load(id);
                    return v ? (v | 0x80000000u) : id;
                };
                auto value_of = [&](uint32_t key) -> uint32_t {
                    if (!(key & 0x80000000u)) return key;
                    auto it = lstores.find(key & 0x7fffffffu);
                    return it == lstores.end() || it->second.size() != 1 || it->second[0].cond ? 0 : it->second[0].value;
                };
                auto word_addr = [&](uint32_t key, WordAddr &w) -> bool {
                    const uint32_t v = value_of(key);
                    const auto *g = v ? def_of(v) : nullptr;
                    if (!g || (*g)[0] != 61) return false;
                    auto ch = chain_all.find((*g)[2]);
                    if (ch == chain_all.end() || ch->second.idx.size() < 3) return false;
                    auto bv = vars.find(ch->second.base);
                    if (bv == vars.end() || (bv->second.second != 2 && bv->second.second != 12)) return false;
                    const size_t n = ch->second.idx.size();
                    const auto *gr = def_of(ch->second.idx[n - 2]), *gc = def_of(ch->second.idx[n - 1]);
                    if (!gr || (*gr)[0] != 194 || !gc || (*gc)[0] != 199) return false;
                    auto cs = iconst.find((*gr)[3]);
                    if (cs == iconst.end() || cs->second != 2) return false;
                    uint32_t z = 0, cm = 0;
                    auto c3 = iconst.find((*gc)[3]), c2 = iconst.find((*gc)[2]);
                    if (c3 != iconst.end() && c3->second == 3) cm = (*gc)[3], z = (*gc)[2];
                    else if (c2 != iconst.end() && c2->second == 3) cm = (*gc)[2], z = (*gc)[3];
                    else return false;
                    const uint32_t y = (*gr)[2];
                    const auto *gy = def_of(y);
                    if (canon_idx(y) != canon_idx(z) || !gy) return false;
                    w.x = y, w.xtype = (*gy)[1], w.row_type = (*gr)[1], w.comp_type = (*gc)[1], w.c_shift = (*gr)[3], w.c_mask = cm;
                    w.chain_type = ch->second.type, w.base = ch->second.base, w.float_type = (*g)[1];
                    w.lead.assign(ch->second.idx.begin(), ch->second.idx.end() - 2);
                    return true;
                };
                auto same_source = [&](const WordAddr &p, const WordAddr &q) {
                    return p.xtype == q.xtype && p.row_type == q.row_type && p.comp_type == q.comp_type && p.c_shift == q.c_shift && p.c_mask == q.c_mask &&
                           p.chain_type == q.chain_type && p.base == q.base && p.float_type == q.float_type && p.lead == q.lead;
                };
                // a sum B = (plain int local E) + pick, where E is the stored value of another sum A: both can use one load
                std::map<uint32_t, uint32_t> merged_b;   // root of B -> root of A
                for (auto &kv : rot_plan) {
                    RotPlan &b = kv.second;
                    if (b.extras.size() != 1) continue;
                    const uint32_t var = local_of_load(b.extras[0]);
                    auto ls = var ? lstores.find(var) : lstores.end();
                    if (ls == lstores.end() || ls->second.size() != 1 || ls->second[0].cond || !rot_plan.count(ls->second[0].value)) continue;
                    RotPlan &a = rot_plan[ls->second[0].value];
                    bool disjoint = true, full = true;
                    for (int j = 0; j < 4; j++) {
                        disjoint &= !(a.slot_key[j] && b.slot_key[j]);
                        full &= (a.slot_key[j] || b.slot_key[j]) != 0;
                    }
                    if (!disjoint || !full || a.shared) continue;
                    a.shared = true, b.merge_into = ls->second[0].value;
                    merged_b[kv.first] = ls->second[0].value;
                }
                for (auto it = rot_plan.begin(); it != rot_plan.end();) {
                    RotPlan &p = it->second;
                    bool ok = true;
                    if (!p.merge_into) {
                        const RotPlan *other = nullptr;
                        if (p.shared)
                            for (auto &mb : merged_b) if (mb.second == it->first) other = &rot_plan[mb.first];
                        bool have = false;
                        for (int j = 0; j < 4 && ok; j++) {
                            uint32_t key = p.slot_key[j] ? p.slot_key[j] : (other ? other->slot_key[j] : 0);
                            if (!key) continue;
                            ok = word_addr(key, p.addr[j]) && (!have || same_source(p.addr[j], p.addr[have ? 0 : 0]) || true);
                            have = true;
                        }
                        // all present words must agree on the buffer, the lead indices and the types
                        const WordAddr *first = nullptr;
                        for (int j = 0; j < 4 && ok; j++) {
                            if (!p.addr[j].x) continue;
                            if (!first) first = &p.addr[j];
                            else ok = same_source(*first, p.addr[j]);
                        }
                        ok = ok && first;
                        p.lx_ok = ok;
                    }
                    if (!ok) it = rot_plan.erase(it); else ++it;
                }
                // a merged B whose A was dropped is dropped too
                for (auto it = rot_plan.begin(); it != rot_plan.end();) {
                    if (it->second.merge_into && !rot_plan.count(it->second.merge_into)) it = rot_plan.erase(it); else ++it;
                }
                // constants for the comparisons with r
                if (!rot_plan.empty()) {
                    const uint32_t tr = and3[rot_plan.begin()->second.r_id].type;
                    for (unsigned j = 0; j < 4; j++) {
                        auto cj = const_by.find({tr, j});
                        if (cj != const_by.end()) lx_j[j] = cj->second;
                        else { lx_j[j] = bound++; decl.insert(decl.end(), {4u << 16 | OpConstant_, tr, lx_j[j], j}); const_by[{tr, j}] = lx_j[j]; }
                    }
                }
            }
            if (!rot_plan.empty()) {
                const uint32_t t = rot_plan.begin()->second.type;
                auto zc = const_by.find({t, 0});
                if (zc != const_by.end()) rot_zero = zc->second;
                else { rot_zero = bound++; decl.insert(decl.end(), {4u << 16 | OpConstant_, t, rot_zero, 0}); const_by[{t, 0}] = rot_zero; }
                for (auto &vi : vec_info) if (vi.second.first == t && vi.second.second == 4) rot_ivec4 = vi.first;
                if (!rot_ivec4) { rot_ivec4 = bound++; decl.insert(decl.end(), {4u << 16 | OpTypeVector_, rot_ivec4, t, 4}); }
                done |= do_loadexact ? 1u << 17 : 1u << 16;
            }
        }
        if (do_ubocomp) {   // the declarations the rewritten loads need: the vector pointer type, bool, and the index constants
            for (auto &kv : uc) {
                UcChain &c = kv.second;
                if (!c.loads) continue;
                auto pt = ptr_of.find({c.storage, c.vec});
                if (pt != ptr_of.end()) c.vptr = pt->second;
                else {
                    c.vptr = bound++;
                    decl.insert(decl.end(), {4u << 16 | OpTypePointer_, c.vptr, c.storage, c.vec});
                    ptr_of[{c.storage, c.vec}] = c.vptr;
                }
                if (!bool_ty) { bool_ty = bound++; decl.insert(decl.end(), {2u << 16 | 20u, bool_ty}); }
                for (unsigned v = 1; v < c.count; v++) {
                    auto ci = const_by.find({c.idx_type, v});
                    if (ci != const_by.end()) c.c[v] = ci->second;
                    else {
                        c.c[v] = bound++;
                        decl.insert(decl.end(), {4u << 16 | OpConstant_, c.idx_type, c.c[v], v});
                        const_by[{c.idx_type, v}] = c.c[v];
                    }
                }
                c.short_ptr = bound++;
            }
        }
        if (do_vsmask) {
            for (auto &v : vars) {
                if (v.second.second != 3) continue;
                auto l = loc_of.find(v.first);
                if (l == loc_of.end() || l->second >= 16 || ((vmask >> l->second) & 1)) continue;
                if (!ptrs.count(v.second.first)) return fail("vsmask: an Output variable's pointer type is unknown");
                mask_nulls.push_back({v.first, get_null(pointee_of(v.first))});
                dead_ptrs.insert(v.first);
            }
            if (mask_pos && !pos_ac_ptr) dead_ptrs.insert(pos_var);
        }
        if (do_null) {
            for (uint32_t id : ep_iface) {
                auto v = vars.find(id);
                if (v == vars.end() || v->second.second != 3) continue;
                if (!ptrs.count(v->second.first)) return fail("nullfs: an Output variable's pointer type is unknown");
                body.insert(body.end(), {3u << 16 | OpStore_, id, get_null(pointee_of(id))});
            }
        }
        if (do_nullvs || do_passvs) {
            for (uint32_t id : ep_iface) {
                auto v = vars.find(id);
                if (v != vars.end() && v->second.second != 1 && !ptrs.count(v->second.first))
                    return fail("a variable's pointer type is unknown");
            }
            store_position(body);
            // Outputs other than Position, in Location order (the ones without a Location come last).
            std::vector<std::pair<uint32_t, uint32_t>> outs, ins;   // (Location, variable)
            for (uint32_t id : ep_iface) {
                auto v = vars.find(id);
                if (v == vars.end() || id == pos_var) continue;
                auto l = loc_of.find(id);
                if (v->second.second == 3) outs.push_back({l == loc_of.end() ? 0xffffffffu : l->second, id});
                else if (v->second.second == 1 && l != loc_of.end()) ins.push_back({l->second, id});
            }
            std::sort(outs.begin(), outs.end());
            std::sort(ins.begin(), ins.end());
            std::vector<std::pair<uint32_t, uint32_t>> loaded;   // (type, value id)
            if (do_passvs)
                for (auto &in : ins) {
                    uint32_t type = pointee_of(in.second), ld = bound++;
                    body.insert(body.end(), {4u << 16 | 61u, type, ld, in.second});   // OpLoad
                    loaded.push_back({type, ld});
                }
            size_t k = 0;
            for (auto &out : outs) {
                uint32_t type = pointee_of(out.second), value = 0;
                if (do_passvs && out.first != 0xffffffffu) {
                    if (k < loaded.size() && loaded[k].first == type) value = loaded[k].second;
                    k++;
                }
                if (!value) value = get_null(type);
                body.insert(body.end(), {3u << 16 | OpStore_, out.second, value});
            }
        }
        bool decls_done = false;
        size_t ands_seen = 0, ands_nulled = 0;
        unsigned vb_seen = 0, vb_changed = 0, vb_merge = 0, mul2_done = 0;
        std::map<uint32_t, uint32_t> mul2_new;   // planned select id -> its low-two-bits twin
        bool vb_forced = false;
        if (do_flatten) done |= 1u << 10;   // flatten runs after the rewrite loop; the module counts as patched
        if (do_caps)   // the capabilities that stay decide whether their extension may go
            for (size_t i = 5; i < n && (w[i] & 0xffff) == OpCapability_; i += w[i] >> 16) {
                uint32_t c = w[i + 1];
                if (cap_removed(c)) continue;
                if (c == 5363 || c == 5378 || c == 5372) interlock_cap_left = true;
                if (c == 4427) dp_cap_left = true;
            }
        cur_fn = 0;
        for (size_t i = 5; i < n;) {
            uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
            const uint32_t *a = w + i + 1;
            if (!caps_done && op != OpCapability_) {
                caps_done = true;
                if (do_demote && !has_demote_cap) { o.push_back(2u << 16 | OpCapability_); o.push_back(CapDemote_); }
            }
            if (!exts_done && op != OpCapability_ && op != OpExtension_) {
                exts_done = true;
                if (do_demote && !v16 && !has_demote_ext) put_str(o, OpExtension_, "SPV_EXT_demote_to_helper_invocation");
            }
            if (do_caps) {
                bool drop = (op == OpCapability_ && cap_removed(a[0])) ||
                            (op == OpExtension_ && ((str_eq(a, len - 1, "SPV_EXT_fragment_shader_interlock") && !interlock_cap_left) ||
                                                    (str_eq(a, len - 1, "SPV_KHR_shader_draw_parameters") && !dp_cap_left)));
                if (drop) { done |= 1; i += len; continue; }
            }
            if (op == OpFunction_ && !decls_done) {
                decls_done = true;
                o.insert(o.end(), decl.begin(), decl.end());
            }
            if (new_entry && op == OpFunction_ && a[1] == ep_fn) {
                // The new entry function: the stores built above, then a return. The old one stays as a dead function
                // that EliminateDeadFunctions removes (and with it its names and decorations).
                o.insert(o.end(), {5u << 16 | OpFunction_, a[0], new_fn, a[2], a[3]});
                o.insert(o.end(), {2u << 16 | OpLabel_, bound});
                bound++;
                o.insert(o.end(), body.begin(), body.end());
                o.push_back(1u << 16 | OpReturn_);
                o.push_back(1u << 16 | OpFunctionEnd_);
                done |= (do_null ? 8u : 0u) | (do_nullvs ? 32u : 0u) | (do_passvs ? 64u : 0u);
            }
            if (new_entry && ((op == OpEntryPoint_ && len >= 3 && a[1] == ep_fn) ||
                            ((op == OpExecutionMode_ || op == OpExecutionModeId_) && len >= 2 && a[0] == ep_fn))) {
                o.insert(o.end(), w + i, w + i + len);
                o[o.size() - len + (op == OpEntryPoint_ ? 2 : 1)] = new_fn;
                i += len;
                continue;
            }
            if (op == OpFunction_) cur_fn = a[1];
            if (op == OpFunctionEnd_) cur_fn = 0;
            if (do_mul2 && op == 169 && len >= 6) {
                auto mp = mul2_plan.find(a[1]);
                if (mp != mul2_plan.end()) {
                    // the select stays; next to it, select(cond, V, -1) for the &3 ANDs that use it, V without the shifted term
                    const Mul2 &m = mp->second;
                    const uint32_t sel2 = bound++;
                    uint32_t ss = m.v;
                    o.insert(o.end(), w + i, w + i + len);
                    if (m.p) {   // V = lo(P) * lo(Q) + C: only the low two bits of P, Q and C matter
                        const uint32_t pl = bound++, ql = bound++, cl = bound++, mm = bound++;
                        ss = bound++;
                        o.insert(o.end(), {5u << 16 | OpBitwiseAnd_, m.type, pl, m.p, m.c3});
                        o.insert(o.end(), {5u << 16 | OpBitwiseAnd_, m.type, ql, m.q, m.c3});
                        o.insert(o.end(), {5u << 16 | OpBitwiseAnd_, m.type, cl, m.c, m.c3});
                        o.insert(o.end(), {5u << 16 | 132u, m.type, mm, pl, ql});
                        o.insert(o.end(), {5u << 16 | 128u, m.type, ss, mm, cl});
                    }
                    o.insert(o.end(), {6u << 16 | 169u, a[0], sel2, a[2], ss, a[4]});
                    mul2_new[a[1]] = sel2;
                    i += len;
                    continue;
                }
            }
            if (do_mul2 && op == OpBitwiseAnd_ && len == 5 && mul2_and.count(a[1])) {
                auto nw = mul2_new.find(mul2_and[a[1]]);
                if (nw != mul2_new.end()) {   // the operand that is not the constant 3
                    o.insert(o.end(), w + i, w + i + len);
                    o[o.size() - len + 3 + (iconst.count(a[2]) && iconst[a[2]] == 3 ? 1 : 0)] = nw->second;
                    done |= 1u << 21;
                    mul2_done++;
                    i += len;
                    continue;
                }
            }
            if (do_vsblock) {
                if (op == OpSelectionMerge_ && len >= 2) {
                    vb_forced = vb_seen >= vb_lo && vb_seen <= vb_hi;
                    vb_merge = a[0];
                    vb_seen++;
                } else if (op == OpBranchConditional_ && len >= 4 && vb_forced) {
                    const bool take_true = vb_mode == 1 ? true : vb_mode == 2 ? false : a[1] == vb_merge;
                    o.insert(o.end(), w + i, w + i + len);
                    o[o.size() - len + 1] = take_true ? const_true : const_false;
                    vb_forced = false;
                    vb_changed++;
                    done |= 1u << 19;
                    i += len;
                    continue;
                } else {
                    vb_forced = false;
                }
            }
            if (do_vsmask) {
                if ((op == OpAccessChain_ || op == OpInBoundsAccessChain_) && len >= 5) {
                    if (dead_ptrs.count(a[2])) {
                        dead_ptrs.insert(a[1]);
                    } else if (mask_pos && pos_ac_ptr && a[2] == pos_var) {   // the chain to the Position member
                        auto c = consts.find(a[3]);
                        if (c != consts.end() && c->second == pos_idx) dead_ptrs.insert(a[1]);
                    }
                }
                if (op == OpStore_ && len >= 3 && dead_ptrs.count(a[0])) { i += len; continue; }
            }
            if (op == OpReturn_ && entry_fns.count(cur_fn)) {
                if (do_vsmask) {
                    for (auto &st : mask_nulls) o.insert(o.end(), {3u << 16 | OpStore_, st.first, st.second});
                    if (mask_pos) store_position(o);
                    done |= cut_fallback ? 512 : 128;
                }
                if (do_cull) {
                    store_position(o);
                    done |= 16;
                }
            }
            if (!rot_plan.empty() && op == 128 && len >= 5) {
                auto rp = rot_plan.find(a[1]);
                if (rp != rot_plan.end()) {
                    const RotPlan &p = rp->second;
                    if (do_loadexact) {   // one load of the selected word
                        if (p.merge_into) {   // another sum's load already holds the word
                            o.insert(o.end(), {4u << 16 | OpCopyObject_, p.type, a[1], lx_s[p.merge_into]});
                            i += len;
                            continue;
                        }
                        const RotPlan *other = nullptr;
                        if (p.shared)
                            for (auto &q : rot_plan) if (q.second.merge_into == a[1]) other = &q.second;
                        WordAddr x[4];   // the address for each r (an empty slot takes the first known one: the result is zeroed below)
                        const WordAddr *known = nullptr;
                        for (int j = 0; j < 4; j++) if (p.addr[j].x && !known) known = &p.addr[j];
                        for (int j = 0; j < 4; j++) x[j] = p.addr[j].x ? p.addr[j] : *known;
                        const uint32_t tr = and3[p.r_id].type;
                        auto eq = [&](unsigned j) {
                            const uint32_t id = bound++;
                            o.insert(o.end(), {5u << 16 | 170u, bool_ty, id, p.r_id, lx_j[j]});   // OpIEqual r, j
                            return id;
                        };
                        (void)tr;
                        const uint32_t s2 = bound++, s1 = bound++, idx = bound++;
                        const uint32_t e2 = eq(2), e1 = eq(1), e0 = eq(0);
                        o.insert(o.end(), {6u << 16 | 169u, x[0].xtype, s2, e2, x[2].x, x[3].x});
                        o.insert(o.end(), {6u << 16 | 169u, x[0].xtype, s1, e1, x[1].x, s2});
                        o.insert(o.end(), {6u << 16 | 169u, x[0].xtype, idx, e0, x[0].x, s1});
                        const WordAddr &w = x[0];
                        const uint32_t row = bound++, col = bound++, chain = bound++, ld = bound++, bits = bound++;
                        o.insert(o.end(), {5u << 16 | 194u, w.row_type, row, idx, w.c_shift});      // x >> 2
                        o.insert(o.end(), {5u << 16 | 199u, w.comp_type, col, idx, w.c_mask});      // x & 3
                        o.push_back(static_cast<uint32_t>(4 + w.lead.size() + 2) << 16 | OpAccessChain_);
                        o.push_back(w.chain_type), o.push_back(chain), o.push_back(w.base);
                        o.insert(o.end(), w.lead.begin(), w.lead.end());
                        o.push_back(row), o.push_back(col);
                        o.insert(o.end(), {4u << 16 | OpLoad_, w.float_type, ld, chain});
                        o.insert(o.end(), {4u << 16 | 124u, p.type, bits, ld});                      // the word's bits
                        if (p.shared) lx_s[a[1]] = bits;
                        uint32_t acc = bits;
                        for (int j = 0; j < 4; j++) {   // r whose weight was hot for no word of this sum: the sum is 0 there
                            if (p.slot_key[j]) continue;
                            if (p.shared && other && other->slot_key[j]) continue;   // the other sum's word: kept in the shared value, zeroed here
                            const uint32_t e = eq(static_cast<unsigned>(j)), next = bound++;
                            o.insert(o.end(), {6u << 16 | 169u, p.type, next, e, rot_zero, acc});
                            acc = next;
                        }
                        if (p.shared && other)   // this sum holds only its own slots
                            for (int j = 0; j < 4; j++)
                                if (!p.slot_key[j] && other->slot_key[j] && j < 3) {}   // zeroed above through the empty-slot rule
                        for (size_t e = 0; e < p.extras.size(); e++) {
                            const uint32_t next = bound++;
                            o.insert(o.end(), {5u << 16 | 128u, p.type, next, p.extras[e], acc});
                            acc = next;
                        }
                        o.insert(o.end(), {4u << 16 | OpCopyObject_, p.type, a[1], acc});
                        i += len;
                        continue;
                    }
                    uint32_t comp[4];
                    for (int k = 0; k < 4; k++) {
                        if (!p.slot_key[k]) { comp[k] = rot_zero; continue; }
                        uint32_t src = p.slot_key[k];
                        if (src & 0x80000000u) {   // a float local: load it again
                            const uint32_t var = src & 0x7fffffffu, ft = ptrs[vars[var].first].second;
                            src = bound++;
                            o.insert(o.end(), {4u << 16 | OpLoad_, ft, src, var});
                        }
                        const uint32_t bc = bound++;
                        o.insert(o.end(), {4u << 16 | 124u, p.type, bc, src});   // OpBitcast to the int the original multiplied
                        comp[k] = bc;
                    }
                    const uint32_t vec = bound++;
                    uint32_t acc = p.extras.empty() ? a[1] : bound++;
                    o.insert(o.end(), {7u << 16 | 80u, rot_ivec4, vec, comp[0], comp[1], comp[2], comp[3]});   // OpCompositeConstruct
                    o.insert(o.end(), {5u << 16 | 77u, p.type, acc, vec, p.r_id});                            // OpVectorExtractDynamic
                    for (size_t e = 0; e < p.extras.size(); e++) {   // + the addends that are not weighted words
                        const uint32_t next = e + 1 == p.extras.size() ? a[1] : bound++;
                        o.insert(o.end(), {5u << 16 | 128u, p.type, next, p.extras[e], acc});
                        acc = next;
                    }
                    i += len;
                    continue;
                }
            }
            if (do_and3exact) {
                if (op == OpBitwiseAnd_ && len >= 5) {
                    auto it = and3.find(a[1]);
                    if (it != and3.end() && it->second.cls == 4 && it->second.type) {
                        const uint32_t t1 = bound++, t2 = bound++, c2 = ex3_c2[it->second.type], x = it->second.other;
                        o.insert(o.end(), {5u << 16 | 194u, a[0], t1, x, c2});     // OpShiftRightLogical x, 2
                        o.insert(o.end(), {5u << 16 | 196u, a[0], t2, t1, c2});    // OpShiftLeftLogical, 2
                        o.insert(o.end(), {5u << 16 | 130u, a[0], a[1], x, t2});   // OpISub x, that: x & 3 for every 32-bit x
                        i += len;
                        continue;
                    }
                }
                auto cp = ex3_cmp.find(a[1]);
                if (len >= 5 && cp != ex3_cmp.end() && op >= 170 && op <= 179) {
                    o.insert(o.end(), {5u << 16 | 77u, a[0], a[1], ex3_table[cp->second.table], cp->second.and_id});   // OpVectorExtractDynamic
                    i += len;
                    continue;
                }
            }
            if (do_ldcfix) {
                auto sk = ldc_skip.find(i);
                if (sk != ldc_skip.end()) { i = sk->second; continue; }   // the arm block, run at the header instead
                auto pl = ldc_plan.find(i);
                if (pl != ldc_plan.end()) {
                    const LdcPlan &p = pl->second;
                    for (size_t k = p.arm_begin; k < p.arm_term; k += w[k] >> 16) {
                        const uint32_t kop = w[k] & 0xffff, klen = w[k] >> 16;
                        if (kop == OpStore_) {   // store (cond ? new : old) instead of the guarded store
                            const uint32_t ptr = w[k + 1], val = w[k + 2], ty = ptrs[vars[ptr].first].second;
                            const uint32_t old = bound++, sel = bound++;
                            o.insert(o.end(), {4u << 16 | OpLoad_, ty, old, ptr});
                            o.insert(o.end(), {6u << 16 | 169u, ty, sel, p.cond, p.arm_true ? val : old, p.arm_true ? old : val});
                            o.insert(o.end(), {3u << 16 | OpStore_, ptr, sel});
                        } else {
                            o.insert(o.end(), w + k, w + k + klen);
                        }
                    }
                    o.insert(o.end(), {2u << 16 | 249u, p.merge});   // OpBranch to the merge block
                    i += p.sm_len + p.bc_len;
                    continue;
                }
            }
            if (do_ubocomp && len >= 5 && (op == OpAccessChain_ || op == OpInBoundsAccessChain_)) {
                auto it = uc.find(a[1]);
                if (it != uc.end() && it->second.loads) {   // the same chain without the last index, to the whole vector
                    const UcChain &c = it->second;
                    if (c.other) o.insert(o.end(), w + i, w + i + len);   // a store or atomic still needs the original
                    o.push_back(static_cast<uint32_t>(len - 1) << 16 | op);
                    o.push_back(c.vptr), o.push_back(c.short_ptr), o.push_back(c.base);
                    o.insert(o.end(), c.mid.begin(), c.mid.end());
                    i += len;
                    continue;
                }
            }
            if (do_ubocomp && op == OpLoad_ && len >= 4) {
                auto it = uc.find(a[2]);
                if (it != uc.end() && it->second.loads) {
                    // vector load, one extract per component, then (i==1)?y : (i==2)?z : (i==3)?w : x
                    const UcChain &c = it->second;
                    const uint32_t vec = bound++;
                    o.push_back(static_cast<uint32_t>(len) << 16 | OpLoad_);
                    o.push_back(c.vec), o.push_back(vec), o.push_back(c.short_ptr);
                    o.insert(o.end(), a + 3, a + len - 1);   // memory operands
                    uint32_t ex[4], eq[4];
                    for (uint32_t k = 0; k < c.count; k++) {
                        ex[k] = bound++;
                        o.insert(o.end(), {5u << 16 | 81u, a[0], ex[k], vec, k});   // OpCompositeExtract
                    }
                    for (uint32_t k = 1; k < c.count; k++) {
                        eq[k] = bound++;
                        o.insert(o.end(), {5u << 16 | 170u, bool_ty, eq[k], c.last, c.c[k]});   // OpIEqual
                    }
                    uint32_t rest = ex[0];
                    for (uint32_t k = c.count - 1; k >= 1; k--) {
                        const uint32_t res = k == 1 ? a[1] : bound++;
                        o.insert(o.end(), {6u << 16 | 169u, a[0], res, eq[k], ex[k], rest});   // OpSelect
                        rest = res;
                    }
                    done |= 1u << 13;
                    uc_loads++;
                    i += len;
                    continue;
                }
            }
            if (do_notrig && op == OpExtInst_ && len >= 5 && a[2] == ext_glsl && (a[3] == 13 || a[3] == 14)) {   // Sin / Cos
                o.insert(o.end(), {4u << 16 | OpCopyObject_, a[0], a[1], trig_null[a[0]]});
                done |= 1u << 11;
                i += len;
                continue;
            }
            if (do_noflags && op == OpBitwiseAnd_ && len >= 5) {
                const AndFilter &af = af_sel;
                bool hit = !af.on;
                unsigned cls = 0;
                if (af.on) {   // only the ANDs the list names
                    auto c0 = iconst.find(a[2]), c1 = iconst.find(a[3]);
                    if (c0 == iconst.end() && c1 == iconst.end()) hit = af.nonconst;
                    for (uint32_t v : af.vals) hit |= (c0 != iconst.end() && c0->second == v) || (c1 != iconst.end() && c1->second == v);
                }
                if (af.classes_on) {   // and only the &3 ANDs of the listed classes
                    auto x = and3.find(a[1]);
                    cls = x == and3.end() ? 0 : x->second.cls;
                    if (x != and3.end() && !cls) and3_left++;
                    hit = hit && (cls & af.classes) && (!fold_site || fold_site_ids.count(a[1]));
                }
                ands_seen++;
                if (hit) {
                    if (cls) and3_nulled[cls == 1 ? 0 : cls == 2 ? 1 : 2]++;
                    o.insert(o.end(), {4u << 16 | OpCopyObject_, a[0], a[1], and_null[a[0]]});
                    done |= 1u << 12;
                    ands_nulled++;
                    i += len;
                    continue;
                }
            }
            if (do_cut && len >= 4 && cur_fn == ep_fn) {   // the cut: the result becomes a null of its type
                auto c = cut_null.find(a[1]);
                if (c != cut_null.end() && cut_ids[a[1]] == op) {
                    o.insert(o.end(), {4u << 16 | OpCopyObject_, a[0], a[1], c->second});
                    done |= 256;
                    i += len;
                    continue;
                }
            }
            if (op == OpKill_ && do_demote && entry_fns.count(cur_fn)) {
                // OpKill ends its block and demote does not: the entry function returns right after it.
                o.push_back(1u << 16 | (v16 ? OpDemoteCore_ : OpDemoteExt_));
                o.push_back(1u << 16 | OpReturn_);
                done |= 2;
                i += len;
                continue;
            }
            if (do_lod && (op == OpImageSampleImplicitLod_ || op == OpImageSampleDrefImplicitLod_)) {
                bool in_loop = false;
                for (auto &l : loops) in_loop |= i > l.from && i < l.to;
                const size_t fixed = op == OpImageSampleImplicitLod_ ? 4 : 5;   // operand words before the image-operand mask
                const size_t have = len - 1;
                uint32_t mask = have > fixed ? a[fixed] : 0;
                bool ok = in_loop && have >= fixed && !(mask & ~0xffu) && !(mask & 6);
                std::vector<uint32_t> rest;
                if (ok) {
                    // Bias (0x1) and MinLod (0x80) drop out; the others keep their bit order after the new Lod operand.
                    size_t at = have > fixed ? fixed + 1 : fixed;
                    for (uint32_t bit = 1; bit <= 0x80 && ok; bit <<= 1) {
                        if (!(mask & bit)) continue;
                        if (at >= have) { ok = false; break; }
                        if (bit != 1 && bit != 0x80) rest.push_back(a[at]);
                        at++;
                    }
                    ok = ok && at == have;
                }
                if (ok) {
                    if (!f32_zero) { want_zero = true; f32_zero = bound++; }
                    o.push_back(static_cast<uint32_t>((fixed + 3 + rest.size()) << 16) |
                                (op == OpImageSampleImplicitLod_ ? OpImageSampleExplicitLod_ : OpImageSampleDrefExplicitLod_));
                    o.insert(o.end(), a, a + fixed);
                    o.push_back((mask & ~0x81u) | 2u);
                    o.push_back(f32_zero);
                    o.insert(o.end(), rest.begin(), rest.end());
                    done |= 4;
                    i += len;
                    continue;
                }
            }
            o.insert(o.end(), w + i, w + i + len);
            i += len;
        }
        if (!done) return 0;
        if (want_zero) {
            // OpConstant f32 0.0 goes right after the OpTypeFloat 32 it names, which precedes every use.
            for (size_t i = 5; i < o.size();) {
                uint32_t len = o[i] >> 16;
                if ((o[i] & 0xffff) == OpTypeFloat_ && o[i + 1] == f32) {
                    uint32_t k[4] = {4u << 16 | OpConstant_, f32, f32_zero, 0};
                    o.insert(o.begin() + static_cast<std::ptrdiff_t>(i + len), k, k + 4);
                    break;
                }
                i += len;
            }
        }
        o[3] = bound;
        std::string first;
        // Instructions in the entry function of a module (labels included).
        auto count_entry = [](const std::vector<uint32_t> &m) {
            uint32_t fn = 0;
            for (size_t i = 5; i < m.size() && !fn;) {
                if ((m[i] & 0xffff) == OpEntryPoint_) fn = m[i + 2];
                i += m[i] >> 16;
            }
            size_t cnt = 0;
            bool in = false;
            for (size_t i = 5; i < m.size(); i += m[i] >> 16) {
                uint32_t op = m[i] & 0xffff;
                if (op == OpFunction_) in = m[i + 2] == fn;
                else if (op == OpFunctionEnd_) in = false;
                else if (in) cnt++;
            }
            return cnt;
        };
        const size_t ins_pre = count_entry(o);
        if (new_entry || do_vsmask || do_cut || do_vsblock || do_mul2 || !rot_plan.empty()) {   // what the new entry function, the dropped stores, the cut, the forced branches or the picks left dead
            spvtools::Optimizer opt(SPV_ENV_VULKAN_1_3);
            opt.SetMessageConsumer([&](spv_message_level_t level, const char *, const spv_position_t &, const char *msg) {
                if (level <= SPV_MSG_ERROR && first.empty()) first = msg ? msg : "";
            });
            opt.RegisterPass(spvtools::CreateEliminateDeadFunctionsPass());
            if (do_vsblock) {
                opt.RegisterPass(spvtools::CreateDeadBranchElimPass());   // a constant condition: the other path goes
                opt.RegisterPass(spvtools::CreateBlockMergePass());
            }
            opt.RegisterPass(spvtools::CreateAggressiveDCEPass(!new_entry));   // vsmask alone keeps the interface
            opt.RegisterPass(spvtools::CreateEliminateDeadConstantPass());
            if (new_entry) opt.RegisterPass(spvtools::CreateRemoveUnusedInterfaceVariablesPass());
            spvtools::OptimizerOptions options;
            options.set_run_validator(false);
            std::vector<uint32_t> r;
            if (!opt.Run(o.data(), o.size(), &r, options)) return fail("optimizer failed after the entry function swap: " + first);
            o.swap(r);
        }
        const size_t ins_dce = count_entry(o);
        std::string flat_info;
        if (do_flatten) {
            // Remove data-dependent branches, keeping the maths: locals to SSA values, if-conversion turns the diamonds
            // into OpSelect, then the branches that are left are marked Flatten. Bit 31 of kinds (offline experiments
            // only) adds CombineAccessChains first and runs the sequence a second time before the final DCE.
            auto count_op = [](const std::vector<uint32_t> &m, unsigned op) {
                size_t cnt = 0;
                for (size_t i = 5; i < m.size(); i += m[i] >> 16) cnt += (m[i] & 0xffff) == op;
                return cnt;
            };
            const size_t br0 = count_op(o, OpBranchConditional_), sm0 = count_op(o, OpSelectionMerge_), in0 = count_entry(o);
            spvtools::Optimizer fo(SPV_ENV_VULKAN_1_3);
            fo.SetMessageConsumer([&](spv_message_level_t level, const char *, const spv_position_t &, const char *msg) {
                if (level <= SPV_MSG_ERROR && first.empty()) first = msg ? msg : "";
            });
            const bool extra = std::getenv("DKB_LQ_FLATTEN_EXTRA") != nullptr;
            if (extra) fo.RegisterPass(spvtools::CreateCombineAccessChainsPass());
            for (int round = 0; round < (extra ? 2 : 1); round++) {
                fo.RegisterPass(spvtools::CreateLocalAccessChainConvertPass());
                fo.RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass());
                fo.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
                fo.RegisterPass(spvtools::CreateSSARewritePass());
                // The promoted locals keep their stores until DCE; a store in an arm stops if-conversion from hoisting it.
                fo.RegisterPass(spvtools::CreateAggressiveDCEPass(true));
                fo.RegisterPass(spvtools::CreateIfConversionPass());
                fo.RegisterPass(spvtools::CreateDeadBranchElimPass());
                fo.RegisterPass(spvtools::CreateBlockMergePass());
                fo.RegisterPass(spvtools::CreateSimplificationPass());
            }
            // If-conversion turns a diamond's phis into OpSelect but leaves the branch; DCE removes the now empty arms.
            fo.RegisterPass(spvtools::CreateAggressiveDCEPass(true));
            fo.RegisterPass(spvtools::CreateBlockMergePass());
            spvtools::OptimizerOptions options;
            options.set_run_validator(false);
            std::vector<uint32_t> r;
            if (!fo.Run(o.data(), o.size(), &r, options)) return fail("flatten: optimizer failed: " + first);
            o.swap(r);
            for (size_t i = 5; i < o.size(); i += o[i] >> 16)
                if ((o[i] & 0xffff) == OpSelectionMerge_ && (o[i] >> 16) >= 3) o[i + 2] = 1;   // SelectionControl Flatten
            done |= 1u << 10;
            char buf[160];
            std::snprintf(buf, sizeof buf, "flatten: OpBranchConditional %zu -> %zu, OpSelectionMerge %zu -> %zu, entry function %zu -> %zu instructions",
                          br0, count_op(o, OpBranchConditional_), sm0, count_op(o, OpSelectionMerge_), in0, count_entry(o));
            flat_info = buf;
        }
        // The fold is exact on the captured draws only because the selector is always 0 there; it is applied to the pattern it was
        // found on or not at all: two test-class &3 ANDs and nothing else of that kind.
        if (do_fold && fold_site != 3 && and3_nulled[2] != (fold_site == 4 ? static_cast<unsigned>(fold_site_ids.size()) : fold_site ? 1u : 2u)) return 0;
        if (do_fold) done |= 1u << 18;
        // ---- validate
        spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
        tools.SetMessageConsumer([&](spv_message_level_t level, const char *, const spv_position_t &, const char *msg) {
            if (level <= SPV_MSG_ERROR && first.empty()) first = msg ? msg : "";
        });
        if (!tools.Validate(o.data(), o.size())) return fail("patched module fails validation: " + first);
        auto *copy = static_cast<uint32_t *>(std::malloc(o.size() * 4));
        if (!copy) return fail("out of memory");
        std::memcpy(copy, o.data(), o.size() * 4);
        *out_ptr = copy;
        *out_words = o.size();
        *applied = done;
        if ((do_vsmask || do_cut || do_flatten || do_noflags || do_ubocomp || do_ldcfix || do_and3exact || do_rotexact || do_vsblock || do_mul2) && err && cap) {   // the success text: what the mask, the cut and flatten did
            std::string text;
            if (do_vsmask || do_cut) {
                char what[48] = "", buf[200];
                if (do_cut) std::snprintf(what, sizeof what, " cut %u of %zu ids", cut_want, cut_ids.size());
                if (cut_fallback) std::snprintf(what, sizeof what, " cut %u not applicable, fallback", cut_want);
                std::snprintf(buf, sizeof buf, "mask %05x%s: entry function %zu instructions, %zu with the stores swapped, %zu after DCE",
                              do_vsmask ? vmask : 0x1ffffu, what, entry_ins, ins_pre, ins_dce);
                text = buf;
            }
            if (!flat_info.empty()) text += (text.empty() ? "" : "; ") + flat_info;
            if (do_fold && fold_site == 3) {
                char buf[160];
                std::snprintf(buf, sizeof buf, "fold site exact: %zu selection constructs decided by the selector are now selects", exact_done.size());
                text += (text.empty() ? "" : "; ") + std::string(buf);
            } else if (do_fold && fold_site == 4) {
                char buf[160];
                std::snprintf(buf, sizeof buf, "fold site construct %u: nulled the %zu test-class &3 ANDs that reach its condition", fold_construct, fold_site_ids.size());
                text += (text.empty() ? "" : "; ") + std::string(buf);
            } else if (do_fold && fold_site) {
                text += (text.empty() ? "" : "; ") + std::string(fold_site == 1 ? "fold site: nulled the &3 AND that reaches no branch condition (export)"
                                                                                : "fold site: nulled the &3 AND that reaches the r != 3 branch conditions (branch)");
            }
            if (do_mul2) {
                char buf[160];
                std::snprintf(buf, sizeof buf, "mul2: %u &3 ANDs now read the low two bits of P * Q + C", mul2_done);
                text += (text.empty() ? "" : "; ") + std::string(buf);
            }
            if (do_vsblock) {
                char buf[160];
                std::snprintf(buf, sizeof buf, "vsblock: %u selection constructs in the module, %u of %u..%u forced (%s)", vb_seen, vb_changed, vb_lo, vb_hi,
                              vb_mode == 1 ? "true target" : vb_mode == 2 ? "false target" : "the way to the merge block, else the false target");
                text += (text.empty() ? "" : "; ") + std::string(buf);
            }
            if (do_rotexact) {
                char buf[160];
                std::snprintf(buf, sizeof buf, "%s: %zu weighted sums replaced by a %s (words%s)", do_loadexact ? "loadexact" : "rotexact", rot_plan.size(),
                              do_loadexact ? "single load" : "pick", rot_words.c_str());
                text += (text.empty() ? "" : "; ") + std::string(buf);
            }
            if (do_and3exact) {
                char buf[128];
                std::snprintf(buf, sizeof buf, "and3exact: %u test-class ANDs rewritten, %zu compares replaced by a table lookup", ex3_ands, ex3_cmp.size());
                text += (text.empty() ? "" : "; ") + std::string(buf);
            }
            if (!ldc_plan.empty()) {
                char buf[96];
                std::snprintf(buf, sizeof buf, "%s: %zu data-dependent branches removed", sf_req ? "selfix" : "ldcfix", ldc_plan.size());
                text += (text.empty() ? "" : "; ") + std::string(buf);
            }
            if (do_ubocomp) {
                char buf[160];
                std::snprintf(buf, sizeof buf, "ubocomp: %u dynamic component loads rewritten%s", uc_loads,
                              uc_store_chains ? "; chains also used by a store or atomic are kept" : "");
                text += (text.empty() ? "" : "; ") + std::string(buf);
            }
            if (do_noflags) {
                char buf[96];
                std::snprintf(buf, sizeof buf, "noflags: %zu of %zu OpBitwiseAnd nulled", ands_nulled, ands_seen);
                text += (text.empty() ? "" : "; ") + std::string(buf);
                if (af_sel.classes_on) {
                    std::snprintf(buf, sizeof buf, " (&3 classes nulled: comp %u, round %u, test %u; unclassified left %u)",
                                  and3_nulled[0], and3_nulled[1], and3_nulled[2], and3_left);
                    text += buf;
                }
            }
            std::snprintf(err, cap, "%s", text.c_str());
        }
        return 1;
    } catch (const std::exception &e) {
        return fail(e.what());
    } catch (...) {
        return fail("exception in the patcher");
    }
}

extern "C" int spirv_lq_patch(const uint32_t *w, uint64_t n64, unsigned kinds, uint32_t **out_ptr, uint64_t *out_words,
                              unsigned *applied, char *err, size_t cap) {
    return spirv_lq_patch_ex(w, n64, kinds, out_ptr, out_words, applied, err, cap, 0);
}
