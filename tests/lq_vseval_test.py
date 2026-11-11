"""tools/lq_vseval.py (the CPU evaluator of a captured vertex shader) on a small hand-written module and a hand-written capture.

The shader reads a float per instance from a vertex stream, turns it into a row with a float-to-int conversion, and reads
ubo[row][comp] (a vec4 array of 8) at a dynamic row and component. The three instances use rows 1, 6 and 9 with a uniform block
bound to 64 bytes (4 rows): row 6 is inside the declared array but beyond the bound buffer, row 9 is beyond both. A fourth
output divides 0 by 0, converts the NaN to an integer and uses it as a component: the evaluator must report the non-finite
conversion. No Vulkan, GPU or game run.
"""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import lq_vseval  # noqa: E402

SHADER = r"""
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in0 %out0 %out1 %pos
               OpName %main "main"
               OpName %ubo "ubo"
               OpDecorate %in0 Location 0
               OpDecorate %out0 Location 0
               OpDecorate %out1 Location 1
               OpDecorate %pos BuiltIn Position
               OpDecorate %arr ArrayStride 16
               OpMemberDecorate %blk 0 Offset 0
               OpDecorate %blk Block
               OpDecorate %ubo DescriptorSet 0
               OpDecorate %ubo Binding 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
        %int = OpTypeInt 32 1
       %uint = OpTypeInt 32 0
    %v4float = OpTypeVector %float 4
      %int_0 = OpConstant %int 0
      %int_3 = OpConstant %int 3
     %uint_8 = OpConstant %uint 8
       %zero = OpConstant %float 0
        %arr = OpTypeArray %v4float %uint_8
        %blk = OpTypeStruct %arr
    %ptr_blk = OpTypePointer Uniform %blk
        %ubo = OpVariable %ptr_blk Uniform
     %ptr_uf = OpTypePointer Uniform %float
     %ptr_in = OpTypePointer Input %float
    %ptr_out = OpTypePointer Output %float
   %ptr_out4 = OpTypePointer Output %v4float
        %in0 = OpVariable %ptr_in Input
       %out0 = OpVariable %ptr_out Output
       %out1 = OpVariable %ptr_out Output
        %pos = OpVariable %ptr_out4 Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %a = OpLoad %float %in0
        %row = OpConvertFToS %int %a
        %nan = OpFDiv %float %zero %zero
      %comp0 = OpConvertFToS %int %nan
       %comp = OpBitwiseAnd %int %comp0 %int_3
          %p = OpAccessChain %ptr_uf %ubo %int_0 %row %int_3
          %v = OpLoad %float %p
          %q = OpAccessChain %ptr_uf %ubo %int_0 %int_0 %comp
          %w = OpLoad %float %q
               OpStore %out0 %v
               OpStore %out1 %w
         %pv = OpCompositeConstruct %v4float %v %w %zero %zero
               OpStore %pos %pv
               OpReturn
               OpFunctionEnd
"""


class VsEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args.out.mkdir(parents=True, exist_ok=False)
        asm = args.out / 'vs.spvasm'
        asm.write_text(SHADER, encoding='utf-8')
        subprocess.run([str(args.spirv_val.parent / 'spirv-as'), '--target-env', 'spv1.3', str(asm), '-o', str(args.out / 'vertex.spv')],
                       check=True, capture_output=True, text=True, timeout=60)
        subprocess.run([str(args.spirv_val), '--target-env', 'vulkan1.3', str(args.out / 'vertex.spv')], check=True,
                       capture_output=True, text=True, timeout=60)
        (args.out / 'support.bin').write_bytes(bytes(16))
        (args.out / 'stream-0.bin').write_bytes(struct.pack('<3f', 1.0, 6.0, 9.0))
        # ubo[r][c] = 10 * r + c, only 4 rows bound and captured
        (args.out / 'ubo-9.bin').write_bytes(b''.join(struct.pack('<f', 10.0 * r + c) for r in range(4) for c in range(4)))
        (args.out / 'draw.txt').write_text('\n'.join([
            'format=lq-vs-capture-v1', 'origin=test', 'cache_key=x', 'skipped=0', 'prim=5', 'kind=direct', 'count=1', 'first=0',
            'instances=3', 'base_instance=0', 'base_vertex=0', 'itype=-1', 'support_bytes=16',
            'attr location=0 nvn_format=10 offset=0 stream=0 type=0 comps=1',
            'stream index=0 stride=4 divisor=1 address=0 bound_size=12 captured=12 readable=1 file=stream-0.bin',
            'ubo binding=1 cbuf_slot=9 address=0 bound_size=64 captured=64 readable=1 file=ubo-9.bin']) + '\n', encoding='utf-8')
        cls.mod, cls.rep, cls.results, cls.notes = lq_vseval.evaluate(args.out)
        cls.text = lq_vseval.format_report(cls.mod, cls.rep, cls.results, cls.notes, lq_vseval.load_capture(args.out)['top'])

    def outputs(self, i):
        return {self.mod.names.get(v) or 'loc%s' % self.mod.deco_value(v, 30): val for v, val in self.results[i][1].items()}

    def test_values_inside_the_bound_buffer(self):
        self.assertEqual(self.outputs(0)['loc0'], 10.0 * 1 + 3)      # row 1, component 3 (the constant)

    def test_a_row_beyond_the_bound_buffer_reads_zero_and_is_reported(self):
        self.assertEqual(self.outputs(1)['loc0'], 0.0)               # row 6: declared, but only 4 rows captured
        self.assertEqual(self.outputs(2)['loc0'], 0.0)               # row 9: beyond the declared 8 as well
        self.assertIn('oob-read', self.text)

    def test_the_dynamic_index_is_logged_with_its_declared_length(self):
        ((cid, var, kind, length), c), = [kv for kv in self.rep.chains.items() if kv[0][3] == 8 and kv[1]['n'] == 3 and kv[1]['bad']]
        self.assertEqual((var, kind, length, c['n'], c['min'], c['max'], c['bad']), ('ubo', 'arr', 8, 3, 1, 9, 1))
        self.assertIn('<-- OUT OF RANGE', self.text)

    def test_a_non_finite_value_reaching_an_integer_conversion_is_reported(self):
        self.assertIn('nonfinite-convert', self.text)
        self.assertEqual(self.outputs(0)['loc1'], 0.0)               # NaN converts to 0, component 0 of row 0
        comp_chain = [c for k, c in self.rep.chains.items() if k[3] == 4]
        self.assertTrue(comp_chain and comp_chain[0]['taint'] == 3)    # the component index came from a NaN

    def test_the_run_is_flagged(self):
        self.assertGreater(self.rep.flagged, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spirv-val', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args, rest = parser.parse_known_args()
    raise SystemExit(not unittest.main(argv=[__file__] + rest, exit=False).result.wasSuccessful())
