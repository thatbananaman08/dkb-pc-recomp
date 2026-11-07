// The SPIR-V optimizer step of shader translation: one SPIRV-Tools pass, see spirv_opt.h.
//
// Why: the NVIDIA driver's shader compiler (nvgpucomp64.dll) crashes while building a pipeline from some generated
// vertex shaders. The translator stores intermediate values in function-local variables and loads them back within
// the same block; removing those loads avoids the crash and leaves the arithmetic and its NoContraction decorations
// as they were.
#include "spirv_opt.h"

#include <spirv-tools/optimizer.hpp>

#include <cstdio>
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
