"""runtime/spirv_opt.cpp (the SPIR-V optimizer step of shader translation) on small hand-written modules, no game shaders.

The helper is compiled with the project's C++ compiler against SPIRV-Tools' static libraries (MSYS2 spirv-tools) in a
harness that reads a module, runs spirv_opt_local_single_block and writes the result. It checks that a function-local
store followed by a load in the same block loses the load, that a load in another block is kept (the pass is
single-block), that the multiplication and its NoContraction decoration survive, that spirv-val accepts the result, and
that a module the optimizer refuses is reported as an error. No Vulkan, GPU or game run.
"""
import argparse
from pathlib import Path
import struct
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
OP_LOAD, OP_STORE, OP_FMUL, OP_DECORATE, OP_VARIABLE = 61, 62, 133, 71, 59
NO_CONTRACTION = 42


def instr(op, *words):
    return [(len(words) + 1) << 16 | op, *words]


def module(split_block):
    """A vertex shader: v = 1.0 (function-local); out = vec4(v * v). With split_block the load is in a second block."""
    VOID, FN, FLOAT, VEC, PTR_OUT, PTR_FN, OUT, MAIN, ONE, ENTRY, VAR, LOAD, MUL, VEC4, SECOND = range(1, 16)
    w = [0x07230203, 0x00010300, 0, 16, 0]
    w += instr(17, 1)                                    # OpCapability Shader
    w += instr(14, 0, 1)                                 # OpMemoryModel Logical GLSL450
    w += instr(15, 0, MAIN, 0x6e69616d, 0, OUT)          # OpEntryPoint Vertex "main"
    w += instr(OP_DECORATE, OUT, 30, 0)                  # Location 0
    w += instr(OP_DECORATE, MUL, NO_CONTRACTION)
    w += instr(19, VOID) + instr(33, FN, VOID) + instr(22, FLOAT, 32) + instr(23, VEC, FLOAT, 4)
    w += instr(32, PTR_OUT, 3, VEC) + instr(32, PTR_FN, 7, FLOAT) + instr(OP_VARIABLE, PTR_OUT, OUT, 3)
    w += instr(43, FLOAT, ONE, 0x3f800000)
    w += instr(54, VOID, MAIN, 0, FN) + instr(248, ENTRY) + instr(OP_VARIABLE, PTR_FN, VAR, 7)
    w += instr(OP_STORE, VAR, ONE)
    if split_block:
        w += instr(249, SECOND) + instr(248, SECOND)     # OpBranch, OpLabel
    w += instr(OP_LOAD, FLOAT, LOAD, VAR) + instr(OP_FMUL, FLOAT, MUL, LOAD, LOAD)
    w += instr(80, VEC, VEC4, MUL, MUL, MUL, MUL) + instr(OP_STORE, OUT, VEC4)   # OpCompositeConstruct
    w += instr(253) + instr(56)                          # OpReturn, OpFunctionEnd
    return w


def ops(words):
    i, out = 5, []
    while i < len(words):
        n = words[i] >> 16
        out.append((words[i] & 0xffff, words[i + 1:i + n]))
        i += n
    return out


def count(words, opcode):
    return sum(op == opcode for op, _ in ops(words))


class SpirvOpt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args.out.mkdir(parents=True, exist_ok=False)
        harness = args.out / 'harness.cpp'
        harness.write_text(r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "spirv_opt.h"
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    static uint32_t w[1 << 16];
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    size_t n = fread(w, 4, 1 << 16, f);
    fclose(f);
    char err[256];
    uint64_t out_n = 0;
    uint32_t *o = spirv_opt_local_single_block(w, n, &out_n, err, sizeof err);
    if (!o) return fprintf(stderr, "%s\n", err), 3;
    f = fopen(argv[2], "wb");
    fwrite(o, 4, out_n, f);
    fclose(f);
    free(o);
    return 0;
}
''', encoding='utf-8')
        libs = args.cxx.parent.parent / 'lib'
        cls.exe = args.out / 'harness.exe'
        subprocess.run([str(args.cxx), '-std=c++20', '-O1', '-Wall', '-Wextra', '-Werror', f'-I{ROOT / "runtime"}',
                        str(harness), str(ROOT / 'runtime/spirv_opt.cpp'), f'-L{libs}', '-l:libSPIRV-Tools-opt.a',
                        '-l:libSPIRV-Tools.a', '-o', str(cls.exe)], check=True, capture_output=True, text=True, timeout=300)

    def run_helper(self, name, words):
        src, dst = args.out / (name + '.in.spv'), args.out / (name + '.out.spv')
        src.write_bytes(struct.pack(f'<{len(words)}I', *words))
        run = subprocess.run([str(self.exe), str(src), str(dst)], capture_output=True, text=True, timeout=60)
        if run.returncode == 3:
            return None
        self.assertEqual(run.returncode, 0, run.stderr)
        data = dst.read_bytes()
        self.validate(src)
        self.validate(dst)
        return list(struct.unpack(f'<{len(data) // 4}I', data))

    def validate(self, path):
        check = subprocess.run([str(args.spirv_val), '--target-env', 'vulkan1.3', str(path)], capture_output=True,
                               text=True, timeout=60)
        self.assertEqual(check.returncode, 0, f'{path.name}: {check.stdout}{check.stderr}')

    def test_same_block_load_of_a_local_is_removed(self):
        before = module(False)
        after = self.run_helper('same_block', before)
        self.assertEqual(count(before, OP_LOAD), 1)
        self.assertEqual(count(after, OP_LOAD), 0)

    def test_arithmetic_and_no_contraction_are_kept(self):
        after = self.run_helper('math', module(False))
        self.assertEqual(count(after, OP_FMUL), 1)
        fmul_ids = [a[1] for op, a in ops(after) if op == OP_FMUL]
        self.assertEqual([a[0] for op, a in ops(after) if op == OP_DECORATE and a[1] == NO_CONTRACTION], fmul_ids)

    def test_load_in_another_block_is_kept(self):
        after = self.run_helper('two_blocks', module(True))
        self.assertEqual(count(after, OP_LOAD), 1)

    def test_a_malformed_module_is_an_error(self):
        self.assertIsNone(self.run_helper('bad', [0x07230203, 0x00010300, 0, 1]))
        words = module(False)
        words[0] = 0
        src = args.out / 'bad_magic.spv'
        src.write_bytes(struct.pack(f'<{len(words)}I', *words))
        run = subprocess.run([str(self.exe), str(src), str(args.out / 'bad_magic.out')], capture_output=True, timeout=60)
        self.assertEqual(run.returncode, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', type=Path, required=True)
    parser.add_argument('--spirv-val', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args, rest = parser.parse_known_args()
    raise SystemExit(not unittest.main(argv=[__file__] + rest, exit=False).result.wasSuccessful())
