"""runtime/spirv_opt.cpp (the SPIR-V optimizer step of shader translation) on small hand-written modules, no game shaders.

The helper is compiled with the project's C++ compiler against SPIRV-Tools' static libraries (MSYS2 spirv-tools) in a
harness that reads a module, runs spirv_opt_local_single_block and writes the result. It checks that a function-local
store followed by a load in the same block loses the load, that a load in another block is kept (the pass is
single-block), that the multiplication and its NoContraction decoration survive, that spirv-val accepts the result, and
that a module the optimizer refuses is reported as an error. It also covers the DKB_LQ_PATCH diagnostics
(spirv_lq_patch) on a small fragment shader of the deferred-light family (reads element 83 of fp_c0): unused capabilities
are dropped, OpKill becomes demote + return, and an implicit-LOD sample in a loop becomes explicit LOD 0.0; each result
must pass spirv-val, and a fragment shader outside the family must come back unpatched. The nullfs kind (the same shader
with a second output) must leave no sample or loop and a null store per output; the cullvs kind (a vertex shader with an
input at Location 9 and a gl_PerVertex block, or a plain Position variable) must store vec4(0, 0, 2, 1) into Position
right before OpReturn and keep every load. The nullvs kind replaces such a vertex entry function by one that only stores
Position and a null per other output (no loads, the inputs leave the interface); passvs also forwards the input at the
lowest Location to the output at the lowest Location. The vsmask bits (a mask over a vertex shader with three Location
outputs and Position) must drop the code that only fed the masked outputs and keep the inputs in the interface, and caps
must apply to a vertex module. The vscut cuts (fixed result ids of the captured Ground light-quad shader) are tested on a
small module that defines those ids, and a module without them gets mask 0. The generic vertex kinds flatten (a diamond
with pure arms loses its branch, one whose arm loads keeps it marked Flatten), notrig (Sin / Cos results become nulls)
and noflags (OpBitwiseAnd results become nulls) are tested on small modules with an input at Location 9. ubocomp (any
stage) is tested on uniform and storage buffers with a dynamic component index: the rewritten value is evaluated for every
index with a small interpreter and compared with what the original access picks, for vec2/3/4, signed and unsigned
indices, a pointer feeding two loads and a pointer also stored through. and3exact (every test-class x & 3 and the compares of it
replaced by x - ((x >> 2) << 2) and a constant bool table lookup) is checked with the same interpreter on the original and the
patched function for 32-bit inputs of all four residues. rotexact (the one-hot weighted pick of four words by r = x & 3) is built
in both forms the translator gives it (weights and words as function-local loads, or as forwarded values) with an extra int addend,
and interpreted on the original and the patched module for all four r with NaN, Inf and denormal words. No Vulkan, GPU or game run.
"""
import argparse
import os
import re
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


# A fragment shader of the deferred-light family: reads fp_c0[0][83].x, has an OpKill in the entry function, a loop with an
# implicit-LOD sample (Bias) inside it, and capabilities nothing uses.
LQ_FRAGMENT = r"""
               OpCapability Shader
               OpCapability Float64
               OpCapability FragmentShaderPixelInterlockEXT
               OpCapability GroupNonUniformBallot
               OpCapability DrawParameters
               OpExtension "SPV_EXT_fragment_shader_interlock"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %uv %color
               OpExecutionMode %main OriginUpperLeft
               OpName %main "main"
               OpName %fp_c0 "@C0NAME@"
               OpDecorate %uv Location 0
               OpDecorate %color Location 0
               OpDecorate %arr ArrayStride 16
               OpMemberDecorate %blk 0 Offset 0
               OpDecorate %blk Block
               OpDecorate %fp_c0 DescriptorSet 0
               OpDecorate %fp_c0 Binding 22
               OpDecorate %tex DescriptorSet 0
               OpDecorate %tex Binding 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
        %int = OpTypeInt 32 1
       %bool = OpTypeBool
    %v2float = OpTypeVector %float 2
    %v4float = OpTypeVector %float 4
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
      %int_3 = OpConstant %int 3
     %int_83 = OpConstant %int 83
       %uint = OpTypeInt 32 0
   %uint_128 = OpConstant %uint 128
        %arr = OpTypeArray %v4float %uint_128
        %blk = OpTypeStruct %arr
    %ptr_blk = OpTypePointer Uniform %blk
      %fp_c0 = OpVariable %ptr_blk Uniform
     %ptr_uf = OpTypePointer Uniform %float
    %ptr_in2 = OpTypePointer Input %v2float
   %ptr_out4 = OpTypePointer Output %v4float
         %uv = OpVariable %ptr_in2 Input
      %color = OpVariable %ptr_out4 Output
        %img = OpTypeImage %float 2D 0 0 0 1 Unknown
    %sampled = OpTypeSampledImage %img
    %ptr_tex = OpTypePointer UniformConstant %sampled
        %tex = OpVariable %ptr_tex UniformConstant
       %zero = OpConstant %float 0
      %vzero = OpConstantComposite %v4float %zero %zero %zero %zero
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %p = OpAccessChain %ptr_uf %fp_c0 %int_0 %int_83 %int_0
         %c0 = OpLoad %float %p
        %neg = OpFOrdLessThan %bool %c0 %zero
               OpSelectionMerge %cont None
               OpBranchConditional %neg %dead %cont
       %dead = OpLabel
               OpKill
       %cont = OpLabel
        %uvv = OpLoad %v2float %uv
               OpBranch %head
       %head = OpLabel
          %i = OpPhi %int %int_0 %cont %inext %latch
        %acc = OpPhi %v4float %vzero %cont %accn %latch
               OpLoopMerge %done %latch None
               OpBranch %test
       %test = OpLabel
         %lt = OpSLessThan %bool %i %int_3
               OpBranchConditional %lt %body %done
       %body = OpLabel
         %ti = OpLoad %sampled %tex
        %smp = OpImageSampleImplicitLod %v4float %ti %uvv Bias %c0
               OpBranch %latch
      %latch = OpLabel
       %accn = OpFAdd %v4float %acc %smp
      %inext = OpIAdd %int %i %int_1
               OpBranch %head
       %done = OpLabel
               OpStore %color %acc
               OpReturn
               OpFunctionEnd
"""


# The family fragment shader with a second output (Location 1), for the nullfs kind.
LQ_FRAGMENT_2OUT = (LQ_FRAGMENT
                    .replace('%main "main" %uv %color', '%main "main" %uv %color %color2')
                    .replace('OpDecorate %color Location 0', 'OpDecorate %color Location 0\n OpDecorate %color2 Location 1')
                    .replace('%color = OpVariable %ptr_out4 Output',
                             '%color = OpVariable %ptr_out4 Output\n %color2 = OpVariable %ptr_out4 Output')
                    .replace('OpStore %color %acc', 'OpStore %color %acc\n OpStore %color2 %vzero'))

# A vertex shader with an input at Location @LOC@ and a gl_PerVertex-style block holding Position (member 0) and PointSize.
VS_BLOCK = r"""
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in0 %in9 %out0 %pv
               OpName %main "main"
               OpDecorate %in0 Location 0
               OpDecorate %in9 Location @LOC@
               OpDecorate %out0 Location 0
               OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
               OpMemberDecorate %gl_PerVertex 1 BuiltIn PointSize
               OpDecorate %gl_PerVertex Block
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%gl_PerVertex = OpTypeStruct %v4float %float
     %ptr_pv = OpTypePointer Output %gl_PerVertex
         %pv = OpVariable %ptr_pv Output
     %ptr_in = OpTypePointer Input %v4float
    %ptr_out = OpTypePointer Output %v4float
        %in0 = OpVariable %ptr_in Input
        %in9 = OpVariable %ptr_in Input
       %out0 = OpVariable %ptr_out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %a = OpLoad %v4float %in0
          %b = OpLoad %v4float %in9
          %s = OpFAdd %v4float %a %b
               OpStore %out0 %s
               OpReturn
               OpFunctionEnd
"""

# The same with a plain BuiltIn Position variable.
VS_VAR = r"""
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in0 %in9 %pos
               OpName %main "main"
               OpDecorate %in0 Location 0
               OpDecorate %in9 Location 9
               OpDecorate %pos BuiltIn Position
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
     %ptr_in = OpTypePointer Input %v4float
    %ptr_out = OpTypePointer Output %v4float
        %in0 = OpVariable %ptr_in Input
        %in9 = OpVariable %ptr_in Input
        %pos = OpVariable %ptr_out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %a = OpLoad %v4float %in0
          %b = OpLoad %v4float %in9
          %s = OpFAdd %v4float %a %b
               OpStore %pos %s
               OpReturn
               OpFunctionEnd
"""


# A vertex shader whose four outputs each come from a different arithmetic instruction: out0 (Location 0) FAdd, out1
# (Location 1) FSub, out2 (Location 2) FDiv, Position (block member 0) FMul. Input at Location 9.
VS_MULTI = r"""
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in0 %in9 %out0 %out1 %out2 %pv
               OpName %main "main"
               OpDecorate %in0 Location 0
               OpDecorate %in9 Location 9
               OpDecorate %out0 Location 0
               OpDecorate %out1 Location 1
               OpDecorate %out2 Location 2
               OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
               OpDecorate %gl_PerVertex Block
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%gl_PerVertex = OpTypeStruct %v4float
     %ptr_pv = OpTypePointer Output %gl_PerVertex
         %pv = OpVariable %ptr_pv Output
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
     %ptr_in = OpTypePointer Input %v4float
    %ptr_out = OpTypePointer Output %v4float
        %in0 = OpVariable %ptr_in Input
        %in9 = OpVariable %ptr_in Input
       %out0 = OpVariable %ptr_out Output
       %out1 = OpVariable %ptr_out Output
       %out2 = OpVariable %ptr_out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %a = OpLoad %v4float %in0
          %b = OpLoad %v4float %in9
        %add = OpFAdd %v4float %a %b
        %sub = OpFSub %v4float %a %b
        %div = OpFDiv %v4float %a %b
        %mul = OpFMul %v4float %a %b
               OpStore %out0 %add
               OpStore %out1 %sub
               OpStore %out2 %div
         %pp = OpAccessChain %ptr_out %pv %int_0
               OpStore %pp %mul
               OpReturn
               OpFunctionEnd
"""


# A vertex shader that defines the ids of the vscut "pos" cut (2651, 2666, 2681, 2696: OpFAdd) with numeric ids:
# out0 = fadd(2651, 2666), out1 = fadd(2681, 2696); the input at Location 9 is read only by those four additions' operands.
VS_CUT = r"""
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %11 "main" %7 %8 %9 %10 %20
               OpDecorate %7 Location 0
               OpDecorate %8 Location 9
               OpDecorate %9 Location 0
               OpDecorate %10 Location 1
               OpDecorate %20 BuiltIn Position
          %1 = OpTypeVoid
          %2 = OpTypeFunction %1
          %3 = OpTypeFloat 32
          %4 = OpTypeVector %3 4
          %5 = OpTypePointer Input %3
          %6 = OpTypePointer Output %3
         %21 = OpTypePointer Output %4
          %7 = OpVariable %5 Input
          %8 = OpVariable %5 Input
          %9 = OpVariable %6 Output
         %10 = OpVariable %6 Output
         %20 = OpVariable %21 Output
         %11 = OpFunction %1 None %2
         %12 = OpLabel
         %13 = OpLoad %3 %7
         %14 = OpLoad %3 %8
       %2651 = OpFAdd %3 %13 %14
       %2666 = OpFAdd %3 %14 %14
       %2681 = OpFAdd %3 %14 %13
       %2696 = OpFAdd %3 %13 %13
         %15 = OpFAdd %3 %2651 %2666
         %16 = OpFAdd %3 %2681 %2696
               OpStore %9 %15
               OpStore %10 %16
         %17 = OpCompositeConstruct %4 %13 %13 %13 %13
               OpStore %20 %17
               OpReturn
               OpFunctionEnd
"""


# Vertex shaders with an input at Location 9 and an if (the shape the translator emits) that writes a function-local. ARM_PURE
# stores a value that exists above the branch: if-conversion turns the phi into OpSelect and DCE removes the branch. ARM_LOAD
# and ARM_MUL compute the stored value inside the arm: this SPIRV-Tools only converts a phi whose inputs are already
# defined above the branch, so the branch stays (marked Flatten).
VS_DIAMOND = r"""
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in0 %in9 %out0 %pos
               OpName %main "main"
               OpDecorate %in0 Location 0
               OpDecorate %in9 Location 9
               OpDecorate %out0 Location 0
               OpDecorate %pos BuiltIn Position
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
       %bool = OpTypeBool
    %v4float = OpTypeVector %float 4
     %ptr_in = OpTypePointer Input %float
    %ptr_out = OpTypePointer Output %float
     %ptr_fn = OpTypePointer Function %float
    %ptr_out4 = OpTypePointer Output %v4float
        %in0 = OpVariable %ptr_in Input
        %in9 = OpVariable %ptr_in Input
       %out0 = OpVariable %ptr_out Output
        %pos = OpVariable %ptr_out4 Output
       %zero = OpConstant %float 0
        %two = OpConstant %float 2
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %x = OpVariable %ptr_fn Function
          %a = OpLoad %float %in0
          %b = OpLoad %float %in9
          %c = OpFOrdGreaterThan %bool %a %zero
               OpStore %x %b
               OpSelectionMerge %merge None
               OpBranchConditional %c %then %merge
       %then = OpLabel
@THEN@
               OpBranch %merge
      %merge = OpLabel
          %r = OpLoad %float %x
               OpStore %out0 %r
         %pv = OpCompositeConstruct %v4float %a %a %a %a
               OpStore %pos %pv
               OpReturn
               OpFunctionEnd
"""
ARM_PURE = "               OpStore %x %two"
ARM_LOAD = "          %m = OpLoad %float %in0\n               OpStore %x %m"
ARM_MUL = "          %m = OpFMul %float %a %two\n               OpStore %x %m"

# Sin and Cos through GLSL.std.450 and an OpBitwiseAnd on a bitcast, for notrig and noflags.
VS_TRIG = r"""
               OpCapability Shader
               %glsl = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in0 %in9 %out0 %out1 %pos
               OpName %main "main"
               OpDecorate %in0 Location 0
               OpDecorate %in9 Location 9
               OpDecorate %out0 Location 0
               OpDecorate %out1 Location 1
               OpDecorate %pos BuiltIn Position
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
    %v4float = OpTypeVector %float 4
     %ptr_in = OpTypePointer Input %float
    %ptr_out = OpTypePointer Output %float
    %ptr_out4 = OpTypePointer Output %v4float
        %in0 = OpVariable %ptr_in Input
        %in9 = OpVariable %ptr_in Input
       %out0 = OpVariable %ptr_out Output
       %out1 = OpVariable %ptr_out Output
        %pos = OpVariable %ptr_out4 Output
     %uint_1 = OpConstant %uint 1
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %a = OpLoad %float %in0
          %b = OpLoad %float %in9
          %s = OpExtInst %float %glsl Sin %a
          %k = OpExtInst %float %glsl Cos %b
          %t = OpFAdd %float %s %k
               OpStore %out0 %t
         %ib = OpBitcast %uint %a
         %an = OpBitwiseAnd %uint %ib %uint_1
         %fb = OpBitcast %float %an
               OpStore %out1 %fb
         %pv = OpCompositeConstruct %v4float %a %a %a %a
               OpStore %pos %pv
               OpReturn
               OpFunctionEnd
"""


# Three OpBitwiseAnd on a bitcast: with the constant 1 (right), with 0x10 (left) and with a computed value.
VS_ANDS = r"""
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in0 %in9 %out0 %pos
               OpName %main "main"
               OpDecorate %in0 Location 0
               OpDecorate %in9 Location 9
               OpDecorate %out0 Location 0
               OpDecorate %pos BuiltIn Position
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
    %v4float = OpTypeVector %float 4
     %ptr_in = OpTypePointer Input %float
    %ptr_out = OpTypePointer Output %float
    %ptr_out4 = OpTypePointer Output %v4float
        %in0 = OpVariable %ptr_in Input
        %in9 = OpVariable %ptr_in Input
       %out0 = OpVariable %ptr_out Output
        %pos = OpVariable %ptr_out4 Output
     %uint_1 = OpConstant %uint 1
    %uint_16 = OpConstant %uint 16
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %a = OpLoad %float %in0
          %b = OpLoad %float %in9
         %ia = OpBitcast %uint %a
         %ib = OpBitcast %uint %b
        %an1 = OpBitwiseAnd %uint %ia %uint_1
        %an2 = OpBitwiseAnd %uint %uint_16 %ia
        %an3 = OpBitwiseAnd %uint %ia %ib
        %or1 = OpBitwiseOr %uint %an1 %an2
        %or2 = OpBitwiseOr %uint %or1 %an3
         %fb = OpBitcast %float %or2
               OpStore %out0 %fb
         %pv = OpCompositeConstruct %v4float %a %a %a %a
               OpStore %pos %pv
               OpReturn
               OpFunctionEnd
"""


# A buffer with a vec<N> array; the load takes the row 2 element's component at a dynamic index. @SC@ Uniform / StorageBuffer,
# @IT@ the index type (uint / int), @VT@ the vector type, @EXTRA@ more instructions in the entry block.
UBO_COMP = r"""
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %idxin %out
               OpExecutionMode %main OriginUpperLeft
               OpName %main "main"
               OpDecorate %idxin Location 0
               OpDecorate %idxin Flat
               OpDecorate %out Location 0
               OpDecorate %arr ArrayStride 16
               OpMemberDecorate %blk 0 Offset 0
               OpDecorate %blk Block
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
        %int = OpTypeInt 32 1
       %uint = OpTypeInt 32 0
    %vecty = OpTypeVector %float @N@
     %int_0 = OpConstant %int 0
     %int_2 = OpConstant %int 2
    %uint_8 = OpConstant %uint 8
        %arr = OpTypeArray %vecty %uint_8
        %blk = OpTypeStruct %arr
    %ptr_blk = OpTypePointer @SC@ %blk
        %buf = OpVariable %ptr_blk @SC@
    %ptr_flt = OpTypePointer @SC@ %float
    %ptr_in = OpTypePointer Input @IT@
    %ptr_out = OpTypePointer Output %float
      %idxin = OpVariable %ptr_in Input
        %out = OpVariable %ptr_out Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
        %idx = OpLoad @IT@ %idxin
          %p = OpAccessChain %ptr_flt %buf %int_0 %int_2 %idx
@EXTRA@
               OpReturn
               OpFunctionEnd
"""
UBO_LOAD1 = """          %v = OpLoad %float %p
               OpStore %out %v"""
UBO_LOAD2 = """          %v = OpLoad %float %p
         %v2 = OpLoad %float %p
          %s = OpFAdd %float %v %v2
               OpStore %out %s"""
UBO_STORE = """          %v = OpLoad %float %p
               OpStore %p %v
               OpStore %out %v"""


# Four OpBitwiseAnd with the constant 3, one per class of DKB_LQ_AND3_CLASS plus one that is none of them: comp (the last index of
# a uniform access chain), round ((x>>31)&3), test (feeds a compare and a select) and a plain one (feeds an add only).
VS_AND3 = r"""
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %in0 %in9 %out0 %pos
               OpName %main "main"
               OpDecorate %in0 Location 0
               OpDecorate %in9 Location 9
               OpDecorate %out0 Location 0
               OpDecorate %pos BuiltIn Position
               OpDecorate %arr ArrayStride 16
               OpMemberDecorate %blk 0 Offset 0
               OpDecorate %blk Block
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
       %bool = OpTypeBool
    %v4float = OpTypeVector %float 4
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_3 = OpConstant %uint 3
     %uint_8 = OpConstant %uint 8
    %uint_31 = OpConstant %uint 31
        %arr = OpTypeArray %v4float %uint_8
        %blk = OpTypeStruct %arr
    %ptr_blk = OpTypePointer Uniform %blk
        %buf = OpVariable %ptr_blk Uniform
     %ptr_uf = OpTypePointer Uniform %float
     %ptr_in = OpTypePointer Input %float
    %ptr_out = OpTypePointer Output %float
   %ptr_out4 = OpTypePointer Output %v4float
        %in0 = OpVariable %ptr_in Input
        %in9 = OpVariable %ptr_in Input
       %out0 = OpVariable %ptr_out Output
        %pos = OpVariable %ptr_out4 Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %a = OpLoad %float %in0
          %b = OpLoad %float %in9
         %ia = OpBitcast %uint %a
         %ib = OpBitcast %uint %b
        %an1 = OpBitwiseAnd %uint %ia %uint_3
          %p = OpAccessChain %ptr_uf %buf %uint_0 %uint_1 %an1
          %c = OpLoad %float %p
        %sra = OpShiftRightArithmetic %uint %ia %uint_31
        %an2 = OpBitwiseAnd %uint %sra %uint_3
         %s2 = OpIAdd %uint %an2 %uint_1
        %an3 = OpBitwiseAnd %uint %uint_3 %ib
          %e = OpIEqual %bool %an3 %uint_0
        %sel = OpSelect %uint %e %uint_1 %uint_0
        %an4 = OpBitwiseAnd %uint %ib %uint_3
        %tot = OpIAdd %uint %s2 %sel
       %tot2 = OpIAdd %uint %tot %an4
          %f = OpConvertUToF %float %tot2
          %r = OpFAdd %float %c %f
               OpStore %out0 %r
         %pv = OpCompositeConstruct %v4float %a %a %a %a
               OpStore %pos %pv
               OpReturn
               OpFunctionEnd
"""


# and3exact: r = x & 3 feeds a compare-and-select (r != 0) and a compare-and-branch (!(r > 1)); r itself is also added.
VS_EX3 = r"""
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %xin %out %pos
               OpName %main "main"
               OpDecorate %xin Location 0
               OpDecorate %out Location 0
               OpDecorate %pos BuiltIn Position
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
        %int = OpTypeInt 32 1
       %uint = OpTypeInt 32 0
       %bool = OpTypeBool
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
      %int_3 = OpConstant %int 3
     %uint_1 = OpConstant %uint 1
       %f0 = OpConstant %float 0
     %ptr_in = OpTypePointer Input %int
    %ptr_out = OpTypePointer Output %int
     %ptr_fn = OpTypePointer Function %int
   %ptr_out4 = OpTypePointer Output %v4float
        %xin = OpVariable %ptr_in Input
        %out = OpVariable %ptr_out Output
        %pos = OpVariable %ptr_out4 Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %v = OpVariable %ptr_fn Function
          %w = OpVariable %ptr_fn Function
          %x = OpLoad %int %xin
          %r = OpBitwiseAnd %int %x %int_3
         %ne = OpINotEqual %bool %r %int_0
        %sel = OpSelect %int %ne %int_0 %int_1
               OpStore %w %x
               OpStore %v %int_1
         %rb = OpBitcast %uint %r
         %gt = OpUGreaterThan %bool %rb %uint_1
          %c = OpLogicalNot %bool %gt
               OpSelectionMerge %merge None
               OpBranchConditional %c %arm %merge
        %arm = OpLabel
          %l = OpLoad %int %w
          %n = OpSNegate %int %l
          %m = OpIAdd %int %n %sel
               OpStore %v %m
               OpBranch %merge
      %merge = OpLabel
        %res = OpLoad %int %v
        %sum = OpIAdd %int %res %sel
        %fin = OpIAdd %int %sum %r
               OpStore %out %fin
         %pv = OpCompositeConstruct %v4float %f0 %f0 %f0 %f0
               OpStore %pos %pv
               OpReturn
               OpFunctionEnd
"""

# ldcfix: a triangle `if (c) { arm }` on the low two bits of an index. @BRANCH@ is the OpBranchConditional line, @ARM@ the
# arm's instructions (ending in the OpStore of %v), @CMP@ the compare.
VS_LDC = r"""
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %xin %out %pos
               OpName %main "main"
               OpDecorate %xin Location 0
               OpDecorate %out Location 0
               OpDecorate %pos BuiltIn Position
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
        %int = OpTypeInt 32 1
       %bool = OpTypeBool
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
      %int_1 = OpConstant %int 1
      %int_3 = OpConstant %int 3
       %f0 = OpConstant %float 0
     %ptr_in = OpTypePointer Input %int
    %ptr_out = OpTypePointer Output %int
     %ptr_fn = OpTypePointer Function %int
   %ptr_out4 = OpTypePointer Output %v4float
        %xin = OpVariable %ptr_in Input
        %out = OpVariable %ptr_out Output
        %pos = OpVariable %ptr_out4 Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %v = OpVariable %ptr_fn Function
          %w = OpVariable %ptr_fn Function
          %x = OpLoad %int %xin
          %r = OpBitwiseAnd %int %x %int_3
               OpStore %w %x
               OpStore %v %int_1
@CMP@
          %c = OpLogicalNot %bool %cmp
               OpSelectionMerge %merge None
@BRANCH@
        %arm = OpLabel
@ARM@
               OpBranch %merge
      %merge = OpLabel
        %res = OpLoad %int %v
               OpStore %out %res
         %pv = OpCompositeConstruct %v4float %f0 %f0 %f0 %f0
               OpStore %pos %pv
               OpReturn
               OpFunctionEnd
"""
LDC_CMP = "        %cmp = OpUGreaterThan %bool %r %int_1"
LDC_BR_ARM_TRUE = "               OpBranchConditional %c %arm %merge"
LDC_BR_ARM_FALSE = "               OpBranchConditional %c %merge %arm"
LDC_ARM = """          %l = OpLoad %int %w
          %n = OpSNegate %int %l
          %m = OpIAdd %int %n %r
               OpStore %v %m"""
LDC_ARM_OUT = LDC_ARM + """
               OpStore %out %m"""


def mul2_module(variant='product'):
    """r = (cond ? shl(lo(P)*hi(Q) + hi(P)*lo(Q), 16) + V : -1) & 3 with V = lo(P)*lo(Q) + C ('product') or V = C ('direct'); P is the int input,
    Q, C and the condition's word come from float inputs. The result stored is r + (r == 3), so r counts as a test-class AND. Variants that must
    be left alone: 'falsezero' (the select's false value is 0), 'noshift' (no shifted term), 'other_use' (the select is also stored whole)."""
    L = ['OpCapability Shader', 'OpMemoryModel Logical GLSL450',
         'OpEntryPoint Vertex %main "main" %xin %f0in %f1in %f2in %out %pos', 'OpName %main "main"',
         'OpDecorate %xin Location 0', 'OpDecorate %f0in Location 1', 'OpDecorate %f1in Location 2', 'OpDecorate %f2in Location 3',
         'OpDecorate %out Location 0', 'OpDecorate %pos BuiltIn Position',
         '%void = OpTypeVoid', '%fn = OpTypeFunction %void', '%int = OpTypeInt 32 1', '%bool = OpTypeBool', '%float = OpTypeFloat 32',
         '%v4float = OpTypeVector %float 4', '%int_0 = OpConstant %int 0', '%int_1 = OpConstant %int 1', '%int_3 = OpConstant %int 3',
         '%int_16 = OpConstant %int 16', '%int_m = OpConstant %int 65535', '%int_m1 = OpConstant %int -1', '%f0c = OpConstant %float 0',
         '%pi = OpTypePointer Input %int', '%pf = OpTypePointer Input %float', '%po = OpTypePointer Output %int',
         '%po4 = OpTypePointer Output %v4float', '%xin = OpVariable %pi Input', '%f0in = OpVariable %pf Input', '%f1in = OpVariable %pf Input',
         '%f2in = OpVariable %pf Input', '%out = OpVariable %po Output', '%pos = OpVariable %po4 Output', '%main = OpFunction %void None %fn',
         '%entry = OpLabel', '%x = OpLoad %int %xin', '%fq = OpLoad %float %f0in', '%q = OpBitcast %int %fq', '%fc = OpLoad %float %f1in',
         '%c = OpBitcast %int %fc', '%fz = OpLoad %float %f2in', '%z = OpBitcast %int %fz', '%cond = OpIEqual %bool %z %int_0',
         '%plo = OpBitwiseAnd %int %x %int_m', '%phi = OpShiftRightLogical %int %x %int_16', '%qlo = OpBitwiseAnd %int %q %int_m',
         '%qhi = OpShiftRightLogical %int %q %int_16', '%m1 = OpIMul %int %plo %qhi', '%m2 = OpIMul %int %phi %qlo', '%cs = OpIAdd %int %m1 %m2',
         '%cr = OpShiftLeftLogical %int %cs %int_16']
    if variant == 'direct':
        L += ['%v = OpCopyObject %int %c']
    else:
        L += ['%lo = OpIMul %int %plo %qlo', '%v = OpIAdd %int %lo %c']
    L += ['%sum = OpIAdd %int %cr %v' if variant != 'noshift' else '%sum = OpIAdd %int %m1 %v',
          '%sel = OpSelect %int %cond %sum %int_0' if variant == 'falsezero' else '%sel = OpSelect %int %cond %sum %int_m1',
          '%r = OpBitwiseAnd %int %sel %int_3', '%is3 = OpIEqual %bool %r %int_3', '%o = OpSelect %int %is3 %int_1 %int_0',
          '%res = OpIAdd %int %r %o']
    if variant == 'other_use':
        L += ['%res2 = OpIAdd %int %res %sel', 'OpStore %out %res2']   # the whole select value is used as well
    else:
        L += ['OpStore %out %res']
    L += ['%pv = OpCompositeConstruct %v4float %f0c %f0c %f0c %f0c', 'OpStore %pos %pv', 'OpReturn', 'OpFunctionEnd']
    return chr(10).join(L) + chr(10)


def rot_module(forward, buf=False):
    """r = x & 3; weights w_k = (r == k) and words f_k (float inputs) in function-local variables (or forwarded values); the sum of the
    32-bit products w_k * bits(f_k), each built from 16-bit halves as the translator does it, plus an extra int addend; stored as an int."""
    L = ['OpCapability Shader', 'OpMemoryModel Logical GLSL450',
         'OpEntryPoint Vertex %main "main" %xin %f0in %f1in %f2in %f3in %out %pos', 'OpName %main "main"',
         'OpDecorate %xin Location 0', 'OpDecorate %f0in Location 1', 'OpDecorate %f1in Location 2', 'OpDecorate %f2in Location 3',
         'OpDecorate %f3in Location 4', 'OpDecorate %out Location 0', 'OpDecorate %pos BuiltIn Position',
         '%void = OpTypeVoid', '%fn = OpTypeFunction %void', '%int = OpTypeInt 32 1', '%bool = OpTypeBool', '%float = OpTypeFloat 32',
         '%v4float = OpTypeVector %float 4', '%int_0 = OpConstant %int 0', '%int_1 = OpConstant %int 1', '%int_2 = OpConstant %int 2',
         '%int_3 = OpConstant %int 3', '%int_16 = OpConstant %int 16', '%int_m = OpConstant %int 65535', '%f0c = OpConstant %float 0',
         '%pi = OpTypePointer Input %int', '%pf = OpTypePointer Input %float', '%po = OpTypePointer Output %int',
         '%pfn_i = OpTypePointer Function %int', '%pfn_f = OpTypePointer Function %float', '%po4 = OpTypePointer Output %v4float',
         '%xin = OpVariable %pi Input']
    if buf:   # the four words come from a uniform buffer of 4 x vec4, addressed [x >> 2][x & 3] with x = (x & 12) + k
        L.insert(L.index('OpDecorate %pos BuiltIn Position') + 1, 'OpDecorate %blk Block')
        L.insert(L.index('OpDecorate %blk Block') + 1, 'OpMemberDecorate %blk 0 Offset 0')
        L.insert(L.index('OpDecorate %blk Block') + 2, 'OpDecorate %arr ArrayStride 16')
        L.insert(L.index('OpDecorate %blk Block') + 3, 'OpDecorate %ubo DescriptorSet 0')
        L.insert(L.index('OpDecorate %blk Block') + 4, 'OpDecorate %ubo Binding 0')
        L += ['%int_12 = OpConstant %int 12', '%int_4 = OpConstant %int 4', '%arr = OpTypeArray %v4float %int_4',
              '%blk = OpTypeStruct %arr', '%pblk = OpTypePointer Uniform %blk', '%pu_f = OpTypePointer Uniform %float',
              '%ubo = OpVariable %pblk Uniform']
    L += ['%%f%din = OpVariable %%pf Input' % k for k in range(4)]
    L += ['%out = OpVariable %po Output', '%pos = OpVariable %po4 Output', '%main = OpFunction %void None %fn', '%entry = OpLabel']
    L += ['%%W%d = OpVariable %%pfn_i Function' % k for k in range(4)] + ['%%F%d = OpVariable %%pfn_f Function' % k for k in range(4)]
    L += ['%x = OpLoad %int %xin', '%r = OpBitwiseAnd %int %x %int_3', '%extra = OpIAdd %int %x %int_1']
    if buf: L += ['%x12 = OpBitwiseAnd %int %x %int_12']
    for k in range(4):
        if buf:
            L += ['%%X%d = OpIAdd %%int %%x12 %%int_%d' % (k, k), '%%row%d = OpShiftRightLogical %%int %%X%d %%int_2' % (k, k),
                  '%%col%d = OpBitwiseAnd %%int %%X%d %%int_3' % (k, k), '%%ch%d = OpAccessChain %%pu_f %%ubo %%int_0 %%row%d %%col%d' % (k, k, k),
                  '%%fi%d = OpLoad %%float %%ch%d' % (k, k)]
        else:
            L += ['%%fi%d = OpLoad %%float %%f%din' % (k, k)]
        L += ['OpStore %%F%d %%fi%d' % (k, k),
              '%%e%d = OpIEqual %%bool %%r %%int_%d' % (k, k), '%%w%d = OpSelect %%int %%e%d %%int_1 %%int_0' % (k, k),
              'OpStore %%W%d %%w%d' % (k, k)]
    terms = []
    for k in range(4):
        if forward:
            wl = '%%w%d' % k
            L += ['%%fb%d = OpBitcast %%int %%fi%d' % (k, k)]
        else:
            L += ['%%wl%d = OpLoad %%int %%W%d' % (k, k), '%%fl%d = OpLoad %%float %%F%d' % (k, k), '%%fb%d = OpBitcast %%int %%fl%d' % (k, k)]
            wl = '%%wl%d' % k
        fb = '%%fb%d' % k
        L += ['%%wa%d = OpBitwiseAnd %%int %s %%int_m' % (k, wl), '%%fh%d = OpShiftRightLogical %%int %s %%int_16' % (k, fb),
              '%%m1_%d = OpIMul %%int %%wa%d %%fh%d' % (k, k, k), '%%wh%d = OpShiftRightLogical %%int %s %%int_16' % (k, wl),
              '%%fa%d = OpBitwiseAnd %%int %s %%int_m' % (k, fb), '%%m2_%d = OpIMul %%int %%wh%d %%fa%d' % (k, k, k),
              '%%cs%d = OpIAdd %%int %%m1_%d %%m2_%d' % (k, k, k), '%%cr%d = OpShiftLeftLogical %%int %%cs%d %%int_16' % (k, k),
              '%%lo%d = OpIMul %%int %%wa%d %%fa%d' % (k, k, k)]
        terms += ['%%cr%d' % k, '%%lo%d' % k]
    acc = '%extra'
    for n, t in enumerate(terms):
        L += ['%%acc%d = OpIAdd %%int %s %s' % (n, acc, t)]
        acc = '%%acc%d' % n
    L += ['OpStore %%out %s' % acc, '%pv = OpCompositeConstruct %v4float %f0c %f0c %f0c %f0c', 'OpStore %pos %pv', 'OpReturn', 'OpFunctionEnd']
    return chr(10).join(L) + chr(10)


def run_rot(words, x, fs, ubo=None):
    """Interprets the straight-line entry function of rot_module: x and the four floats f0..f3 -> the int stored to the Output."""
    seq = ops(words)
    M32 = 0xffffffff
    consts = {a[1]: a[2] for op, a in seq if op == 43}
    storage = {a[1]: a[2] for op, a in seq if op == 59}
    loc = {a[0]: a[2] for op, a in seq if op == 71 and a[1] == 30}
    val, mem, out = {k: v & M32 for k, v in consts.items()}, {}, None
    loads = run_rot.loads = []

    def bits(f):
        return struct.unpack('<I', struct.pack('<f', f))[0]

    def flt(b):
        return struct.unpack('<f', struct.pack('<I', b & M32))[0]
    for op, a in seq:
        if op == 65:   # the access chain into the uniform buffer: [0][row][col]
            val[a[1]] = ('ubo', val[a[4]] * 4 + val[a[5]])
        elif op == 61:
            if storage.get(a[2]) == 1:
                val[a[1]] = (x & M32) if loc[a[2]] == 0 else fs[loc[a[2]] - 1]
            elif a[2] in val and isinstance(val[a[2]], tuple):
                val[a[1]] = ubo[val[a[2]][1]]
                loads.append(val[a[2]][1])
            else:
                val[a[1]] = mem[a[2]]
        elif op == 62:
            if storage.get(a[0]) == 3:
                if a[0] in loc:   # the int Output (the position is not modelled)
                    out = val[a[1]]
            else:
                mem[a[0]] = val[a[1]]
        elif op == 124: val[a[1]] = bits(val[a[2]]) if isinstance(val[a[2]], float) else flt(val[a[2]])
        elif op == 83: val[a[1]] = val[a[2]]
        elif op == 128: val[a[1]] = (val[a[2]] + val[a[3]]) & M32
        elif op == 132: val[a[1]] = (val[a[2]] * val[a[3]]) & M32
        elif op == 199: val[a[1]] = val[a[2]] & val[a[3]]
        elif op == 194: val[a[1]] = (val[a[2]] & M32) >> (val[a[3]] & 31)
        elif op == 196: val[a[1]] = (val[a[2]] << (val[a[3]] & 31)) & M32
        elif op == 170: val[a[1]] = val[a[2]] == val[a[3]]
        elif op == 169: val[a[1]] = val[a[3]] if val[a[2]] else val[a[4]]
        elif op == 80: val[a[1]] = [val[e] for e in a[2:]]
        elif op == 77: val[a[1]] = val[a[2]][val[a[3]] & 3]
    return out


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
    if (argc < 3 || argc > 5) return 2;
    static uint32_t w[1 << 16];
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    size_t n = fread(w, 4, 1 << 16, f);
    fclose(f);
    char err[256] = "";
    uint64_t out_n = 0;
    uint32_t *o = nullptr;
    if (argc >= 4) {   // spirv_lq_patch_ex with the kinds bit mask (argv[4]: force_vertex); exit 4: left unchanged
        unsigned applied = 0;
        int r = spirv_lq_patch_ex(w, n, (unsigned)strtoul(argv[3], nullptr, 10), &o, &out_n, &applied, err, sizeof err, argc == 5 ? atoi(argv[4]) : 0);
        if (r < 0) return fprintf(stderr, "%s\n", err), 3;
        if (r == 0) return 4;
        if (err[0]) printf("%s\n", err);
    } else {
        o = spirv_opt_local_single_block(w, n, &out_n, err, sizeof err);
        if (!o) return fprintf(stderr, "%s\n", err), 3;
    }
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

    def lq(self, name, kinds, c0name='fp_c0', text=None, numeric=False, env=None, force=None):
        """Assembles the family fragment shader (or text) and patches it: (before, after), after None when unchanged."""
        asm = args.out / (name + '.spvasm')
        src, dst = args.out / (name + '.in.spv'), args.out / (name + '.out.spv')
        asm.write_text((text or LQ_FRAGMENT).replace('@C0NAME@', c0name), encoding='utf-8')
        subprocess.run([str(args.spirv_val.parent / 'spirv-as'), '--target-env', 'spv1.3', *(['--preserve-numeric-ids'] if numeric else []),
                        str(asm), '-o', str(src)],
                       check=True, capture_output=True, text=True, timeout=60)
        self.validate(src)
        before = list(struct.unpack('<%dI' % (src.stat().st_size // 4), src.read_bytes()))
        run = subprocess.run([str(self.exe), str(src), str(dst), str(kinds)] + ([] if force is None else [str(force)]), capture_output=True, text=True, timeout=60,
                             env={**{k: v for k, v in os.environ.items() if k != 'DKB_LQ_NOFLAGS_ONLY'}, **(env or {})})
        if run.returncode == 4:
            return before, None
        self.assertEqual(run.returncode, 0, run.stderr)
        self.last_info = run.stdout.strip()
        self.validate(dst)
        return before, list(struct.unpack('<%dI' % (dst.stat().st_size // 4), dst.read_bytes()))

    @staticmethod
    def caps(words):
        return sorted(a[0] for op, a in ops(words) if op == 17)

    def test_lq_caps_drops_unused_capabilities_and_extension(self):
        before, after = self.lq('lq_caps', 1)
        self.assertEqual(self.caps(before), [1, 10, 64, 4427, 5378])
        self.assertEqual(self.caps(after), [1])
        self.assertEqual(count(after, 10), 0)    # OpExtension
        self.assertEqual(count(after, 252), 1)   # OpKill untouched
        self.assertEqual(count(after, 87), 1)

    def test_lq_demote_replaces_kill_with_demote_and_return(self):
        before, after = self.lq('lq_demote', 2)
        self.assertEqual(count(after, 252), 0)
        self.assertEqual(count(after, 5380), 1)   # the EXT form: the module is SPIR-V 1.3
        self.assertIn(5379, self.caps(after))
        self.assertEqual(count(after, 253), count(before, 253) + 1)
        exts = [struct.pack('<%dI' % len(a), *a).rstrip(b'\0').decode() for op, a in ops(after) if op == 10]
        self.assertIn('SPV_EXT_demote_to_helper_invocation', exts)

    def test_lq_lod_makes_the_loop_sample_explicit(self):
        before, after = self.lq('lq_lod', 4)
        self.assertEqual(count(after, 87), 0)
        self.assertEqual(count(after, 88), 1)
        sample = [a for op, a in ops(after) if op == 88][0]
        self.assertEqual(sample[4], 2)   # image operands: Lod only, Bias dropped

    def test_lq_all_kinds_together(self):
        _, after = self.lq('lq_all', 7)
        self.assertEqual((count(after, 252), count(after, 87), count(after, 88), count(after, 5380)), (0, 0, 1, 1))
        self.assertEqual(self.caps(after), [1, 5379])

    def test_lq_leaves_other_fragment_shaders_alone(self):
        _, after = self.lq('lq_other', 7, c0name='other_block')
        self.assertIsNone(after)

    def test_nullfs_leaves_only_null_stores(self):
        before, after = self.lq('nullfs', 8, text=LQ_FRAGMENT_2OUT)
        self.assertEqual((count(before, 87), count(before, 246)), (1, 1))
        for op in (87, 88, 89, 90, 246, 252, 5380):
            self.assertEqual(count(after, op), 0, op)
        self.assertEqual(count(after, 54), 1)        # one function: the old entry is gone
        nulls = {a[1] for op, a in ops(after) if op == 46}
        stores = [a for op, a in ops(after) if op == 62]
        self.assertEqual(len(stores), 2)
        self.assertTrue(all(a[1] in nulls for a in stores))
        self.assertEqual(count(after, 253), 1)

    def test_nullfs_ignores_modules_outside_the_family(self):
        _, after = self.lq('nullfs_other', 8, c0name='other_block', text=LQ_FRAGMENT_2OUT)
        self.assertIsNone(after)

    def test_cullvs_stores_outside_the_clip_volume_before_return(self):
        for name, text in (('cull_block', VS_BLOCK.replace('@LOC@', '9')), ('cull_var', VS_VAR)):
            before, after = self.lq(name, 16, text=text)
            self.assertEqual(count(after, 61), count(before, 61), name)   # every input is still read
            self.assertEqual(count(after, 62), count(before, 62) + 1, name)
            seq = ops(after)
            ret = [i for i, (op, _) in enumerate(seq) if op == 253]
            self.assertEqual(len(ret), 1)
            store = seq[ret[0] - 1]
            self.assertEqual(store[0], 62)
            target, value = store[1]
            if name == 'cull_block':
                self.assertEqual(seq[ret[0] - 2][0], 65)                  # OpAccessChain into the block
                self.assertEqual(seq[ret[0] - 2][1][1], target)
            comp = [a for op, a in seq if op == 44 and a[1] == value]
            self.assertEqual(len(comp), 1)
            floats = {a[1]: struct.unpack('<f', struct.pack('<I', a[2]))[0] for op, a in seq if op == 43}
            self.assertEqual([floats[c] for c in comp[0][2:]], [0.0, 0.0, 2.0, 1.0])

    def test_cullvs_needs_an_input_at_location_9(self):
        _, after = self.lq('cull_other', 16, text=VS_BLOCK.replace('@LOC@', '8'))
        self.assertIsNone(after)
        _, after = self.lq('cull_frag', 16)    # a fragment module is never a cullvs target
        self.assertIsNone(after)

    @staticmethod
    def interface(words):
        """The interface ids of the first OpEntryPoint (the name's words skipped)."""
        for op, a in ops(words):
            if op == 15:
                k = 2
                while k < len(a):
                    word = a[k]
                    k += 1
                    if 0 in struct.pack('<I', word):
                        break
                return a[k:]

    def test_nullvs_keeps_only_the_position_store_and_null_outputs(self):
        for name, text in (('nullvs_block', VS_BLOCK.replace('@LOC@', '9')), ('nullvs_var', VS_VAR)):
            before, after = self.lq(name, 32, text=text)
            self.assertEqual(count(after, 61), 0, name)                   # no input is read
            self.assertEqual(count(after, 129), 0, name)                  # the OpFAdd is gone
            self.assertEqual(count(after, 54), 1, name)
            self.assertEqual(count(after, 253), 1, name)
            seq = ops(after)
            nulls = {a[1] for op, a in seq if op == 46}
            composites = {a[1] for op, a in seq if op == 44}
            stores = [a for op, a in seq if op == 62]
            expected = 2 if name == 'nullvs_block' else 1
            self.assertEqual(len(stores), expected, name)
            self.assertEqual(sum(a[1] in composites for a in stores), 1, name)
            self.assertEqual(sum(a[1] in nulls for a in stores), expected - 1, name)
            self.assertLess(len(self.interface(after)), len(self.interface(before)), name)

    def test_passvs_forwards_the_lowest_input_to_the_lowest_output(self):
        before, after = self.lq('passvs', 64, text=VS_BLOCK.replace('@LOC@', '9'))
        seq = ops(after)
        loads = [a for op, a in seq if op == 61]
        self.assertEqual(len(loads), 1)                                   # in9 has no output left to feed: dead, removed
        stores = [a for op, a in seq if op == 62]
        self.assertEqual(len(stores), 2)
        self.assertIn(loads[0][1], [a[1] for a in stores])
        self.assertEqual(count(after, 129), 0)
        self.assertIn(loads[0][2], self.interface(after))

    def test_nullvs_and_passvs_need_an_input_at_location_9(self):
        for kind in (32, 64):
            _, after = self.lq('nvp_other%d' % kind, kind, text=VS_BLOCK.replace('@LOC@', '8'))
            self.assertIsNone(after)
            _, after = self.lq('nvp_frag%d' % kind, kind)
            self.assertIsNone(after)

    def vsmask(self, name, mask, kinds=0):
        before, after = self.lq(name, 128 | kinds | mask << 8, text=VS_MULTI)
        return before, after

    def test_vsmask_keeps_only_the_code_of_the_kept_outputs(self):
        # (mask, kept arithmetic opcodes among FAdd 129, FSub 131, FDiv 136, FMul 133)
        for mask, kept in ((0x1, {129}), (0x2, {131}), (0x4, {136}), (0x10000, {133}), (0x10003, {129, 131, 133})):
            before, after = self.vsmask('vsmask_%x' % mask, mask)
            for op in (129, 131, 136, 133):
                self.assertEqual(count(after, op), int(op in kept), (hex(mask), op))
            self.assertEqual(len(self.interface(after)), len(self.interface(before)))   # the inputs stay declared
            self.assertEqual(count(after, 54), 1)

    def test_vsmask_stores_null_and_the_culling_position_before_return(self):
        _, after = self.vsmask('vsmask_pos', 0x1)
        seq = ops(after)
        ret = [i for i, (op, _) in enumerate(seq) if op == 253]
        self.assertEqual(len(ret), 1)
        nulls = {a[1] for op, a in seq if op == 46}
        composites = {a[1] for op, a in seq if op == 44}
        stores = [a for op, a in seq if op == 62]
        self.assertEqual(len(stores), 4)    # out0 (kept), out1 and out2 (null), Position
        self.assertEqual(sum(a[1] in nulls for a in stores), 2)
        self.assertEqual(sum(a[1] in composites for a in stores), 1)
        self.assertEqual(seq[ret[0] - 1][0], 62)
        self.assertIn(seq[ret[0] - 1][1][1], composites)    # the last store before OpReturn is Position
        floats = {a[1]: struct.unpack('<f', struct.pack('<I', a[2]))[0] for op, a in seq if op == 43}
        comp = [a for op, a in seq if op == 44][0]
        self.assertEqual([floats[c] for c in comp[2:]], [0.0, 0.0, 2.0, 1.0])

    def test_vsmask_reports_the_instruction_counts(self):
        self.vsmask('vsmask_info', 0x1)
        self.assertRegex(self.last_info, r'mask 00001: entry function \d+ instructions, \d+ with the stores swapped, \d+ after DCE')

    def test_vsmask_only_applies_to_vertex_modules_with_location_9(self):
        _, after = self.lq('vsmask_other', 128 | 1 << 8, text=VS_BLOCK.replace('@LOC@', '8'))
        self.assertIsNone(after)
        _, after = self.lq('vsmask_frag', 128 | 1 << 8)
        self.assertIsNone(after)

    def test_caps_applies_to_a_matched_vertex_module(self):
        text = VS_BLOCK.replace('@LOC@', '9').replace('OpCapability Shader', 'OpCapability Shader\n OpCapability Float64')
        before, after = self.lq('caps_vertex', 1, text=text)
        self.assertEqual(self.caps(before), [1, 10])
        self.assertEqual(self.caps(after), [1])
        _, other = self.lq('caps_vertex_other', 1, text=text.replace('Location 9', 'Location 8'))
        self.assertIsNone(other)

    def test_vscut_replaces_the_cut_results_with_nulls_and_drops_their_cone(self):
        before, after = self.lq('vscut_pos', 5 << 25, text=VS_CUT, numeric=True)
        self.assertEqual((count(before, 129), count(before, 61)), (6, 2))
        copies = [a for op, a in ops(after) if op == 83]
        self.assertEqual(sorted(a[1] for a in copies), [2651, 2666, 2681, 2696])
        nulls = {a[1] for op, a in ops(after) if op == 46}
        self.assertTrue(all(a[2] in nulls for a in copies))
        self.assertEqual(count(after, 129), 2)             # the two additions that use the cut results
        self.assertEqual(count(after, 61), 1)              # the Location 9 load fed only the cut: gone
        self.assertEqual(len(self.interface(after)), len(self.interface(before)))   # but the input stays declared
        self.assertRegex(self.last_info, r'cut 5 of 4 ids: entry function \d+ instructions')

    def test_vscut_needs_the_ids_with_their_opcodes(self):
        # Any other matched vertex module gets mask 0: no loads, every Location output null, Position culled.
        for name, text, numeric, stores in (
                ('vscut_other', VS_BLOCK.replace('@LOC@', '9'), False, 2),
                ('vscut_wrong_op', VS_CUT.replace('%2681 = OpFAdd', '%2681 = OpFSub'), True, 3)):
            _, after = self.lq(name, 5 << 25, text=text, numeric=numeric)
            self.assertEqual(count(after, 61), 0, name)
            seq = ops(after)
            nulls = {a[1] for op, a in seq if op == 46}
            composites = {a[1] for op, a in seq if op == 44}
            st = [a for op, a in seq if op == 62]
            self.assertEqual(len(st), stores, name)
            self.assertEqual(sum(a[1] in composites for a in st), 1, name)
            self.assertEqual(sum(a[1] in nulls for a in st), stores - 1, name)
            self.assertRegex(self.last_info, r'cut 5 not applicable, fallback')
        _, after = self.lq('vscut_loc8', 5 << 25, text=VS_BLOCK.replace('@LOC@', '8'))   # not a matched module
        self.assertIsNone(after)
        _, after = self.lq('vscut_frag', 5 << 25)
        self.assertIsNone(after)

    def test_vscut_combines_with_a_mask(self):
        _, after = self.lq('vscut_mask', 128 | 1 << 8 | 5 << 25, text=VS_CUT, numeric=True)
        self.assertEqual(count(after, 83), 2)   # out1 and Position are masked: the copies for 2681 and 2696 are dead

    def test_flatten_removes_a_pure_diamond(self):
        before, after = self.lq('flatten_pure', 1 << 28, text=VS_DIAMOND.replace('@THEN@', ARM_PURE))
        self.assertEqual((count(before, 250), count(before, 247)), (1, 1))
        self.assertEqual((count(after, 250), count(after, 247)), (0, 0))
        self.assertGreaterEqual(count(after, 169), 1)      # OpSelect
        self.assertEqual(count(after, 245), 0)             # no OpPhi left
        self.assertRegex(self.last_info, r'flatten: OpBranchConditional 1 -> 0, OpSelectionMerge 1 -> 0')

    def test_flatten_keeps_a_branch_whose_arm_computes_the_value_and_marks_it_flatten(self):
        for name, arm in (('load', ARM_LOAD), ('mul', ARM_MUL)):
            _, after = self.lq('flatten_' + name, 1 << 28, text=VS_DIAMOND.replace('@THEN@', arm))
            self.assertEqual((count(after, 250), count(after, 247)), (1, 1), name)
            merge = [a for op, a in ops(after) if op == 247][0]
            self.assertEqual(merge[1], 1, name)            # SelectionControl Flatten

    def test_notrig_and_noflags_replace_the_results_with_nulls(self):
        before, after = self.lq('notrig', 1 << 29, text=VS_TRIG)
        self.assertEqual((count(before, 12), count(after, 12), count(after, 83)), (2, 0, 2))
        self.assertEqual(count(after, 199), 1)             # noflags was not asked for
        _, after = self.lq('noflags', 1 << 30, text=VS_TRIG)
        self.assertEqual((count(after, 199), count(after, 83), count(after, 12)), (0, 1, 2))
        _, after = self.lq('both', (1 << 29) | (1 << 30), text=VS_TRIG)
        self.assertEqual((count(after, 12), count(after, 199), count(after, 83)), (0, 0, 3))

    def test_noflags_only_nulls_the_listed_constants(self):
        # (list, ANDs nulled of 3); the three have constants 1 (right), 0x10 (left) and none
        for lst, nulled in ((None, 3), ('1', 1), ('10', 1), ('nonconst', 1), ('1,10', 2), ('1,10,nonconst', 3), ('0x10,nonconst', 2)):
            before, after = self.lq('only_%s' % (lst or 'all').replace(',', '_'), 1 << 30, text=VS_ANDS,
                                    env={'DKB_LQ_NOFLAGS_ONLY': lst} if lst else None)
            self.assertEqual((count(before, 199), count(after, 199), count(after, 83)), (3, 3 - nulled, nulled), lst)
            self.assertRegex(self.last_info, r'noflags: %d of 3 OpBitwiseAnd nulled' % nulled)
        _, after = self.lq('only_none_match', 1 << 30, text=VS_ANDS, env={'DKB_LQ_NOFLAGS_ONLY': '20'})
        self.assertIsNone(after)                           # nothing in the list is present: the module is left alone

    def ubo(self, name, n, sc, it, extra, kinds=1 << 31):
        text = UBO_COMP.replace('@N@', str(n)).replace('@SC@', sc).replace('@IT@', '%' + it).replace('@EXTRA@', extra)
        return self.lq(name, kinds, text=text)

    @staticmethod
    def eval_component(words, vec, idx):
        """Runs the entry block's pure part for one index value: the vector load gives vec, the rest is evaluated."""
        seq = ops(words)
        consts = {a[1]: a[2] for op, a in seq if op == 43}
        types = {}   # result id -> value kind
        val = {}
        # the index: the one OpLoad from the Input variable; the vector: the OpLoad whose result type is a vector
        vec_types = {a[0] for op, a in seq if op == 23}
        for op, a in seq:
            if op == 61:
                val[a[1]] = list(vec) if a[0] in vec_types else idx
            elif op == 81:
                val[a[1]] = val[a[2]][a[3]]
            elif op == 170:
                val[a[1]] = val[a[2]] == consts[a[3]] if a[3] in consts else val[a[2]] == val[a[3]]
            elif op == 169:
                val[a[1]] = val[a[3]] if val[a[2]] else val[a[4]]
            elif op == 129:
                val[a[1]] = val[a[2]] + val[a[3]]
        return val

    def test_ubocomp_loads_the_vector_and_selects_the_component(self):
        for n in (2, 3, 4):
            for sc in ('Uniform', 'StorageBuffer'):
                for it in ('uint', 'int'):
                    name = 'ubocomp_%d_%s_%s' % (n, sc, it)
                    before, after = self.ubo(name, n, sc, it, UBO_LOAD1)
                    seq = ops(after)
                    chains = [a for op, a in seq if op == 65]
                    self.assertEqual(len(chains), 1, name)
                    self.assertEqual(len(chains[0]), 5, name)           # type, id, base, two constant indices: no dynamic one
                    self.assertEqual(count(after, 61), 2, name)         # the index load and one whole-vector load
                    self.assertEqual((count(after, 81), count(after, 170), count(after, 169)), (n, n - 1, n - 1), name)
                    self.assertRegex(self.last_info, r'ubocomp: 1 dynamic component loads rewritten')
                    # the final select carries the original load's result id: evaluate it for every index
                    orig = [a for op, a in ops(before) if op == 61 and a[1] != 0][-1][1]
                    for idx in range(0, 9):
                        got = self.eval_component(after, [10.0, 20.0, 30.0, 40.0][:n], idx)[orig]
                        want = [10.0, 20.0, 30.0, 40.0][idx] if idx < n else 10.0   # out of range: the first component
                        self.assertEqual(got, want, (name, idx))

    def test_ubocomp_rewrites_each_load_of_a_shared_pointer(self):
        _, after = self.ubo('ubocomp_two_loads', 4, 'Uniform', 'uint', UBO_LOAD2)
        self.assertEqual((count(after, 65), count(after, 61), count(after, 81)), (1, 3, 8))
        self.assertRegex(self.last_info, r'ubocomp: 2 dynamic component loads rewritten')

    def test_ubocomp_keeps_the_chain_a_store_uses_but_rewrites_its_load(self):
        _, after = self.ubo('ubocomp_store', 4, 'StorageBuffer', 'uint', UBO_STORE)
        self.assertEqual(count(after, 65), 2)                  # the original (for the store) and the shortened one
        self.assertRegex(self.last_info, r'ubocomp: 1 dynamic component loads rewritten; chains also used by a store')
        stores = [a for op, a in ops(after) if op == 62]
        orig = [a for op, a in ops(after) if op == 65 and len(a) == 6][0]
        self.assertEqual(stores[0][0], orig[1])                # the store still goes through the dynamic chain

    def test_ubocomp_leaves_constant_indices_and_other_storage_alone(self):
        _, after = self.lq('ubocomp_const', 1 << 31, text=UBO_COMP.replace('@N@', '4').replace('@SC@', 'Uniform')
                           .replace('@IT@', '%uint').replace('@EXTRA@', UBO_LOAD1).replace('%idx = OpLoad', '%unused = OpLoad')
                           .replace('%int_2 %idx', '%int_2 %int_2'))
        self.assertIsNone(after)

    def test_ubocomp_applies_to_every_stage(self):
        text = (UBO_COMP.replace('@N@', '4').replace('@SC@', 'Uniform').replace('@IT@', '%uint').replace('@EXTRA@', UBO_LOAD1)
                .replace('Fragment', 'Vertex').replace('               OpExecutionMode %main OriginUpperLeft\n', '')
                .replace('               OpDecorate %idxin Flat\n', ''))
        _, after = self.lq('ubocomp_vertex', 1 << 31, text=text)           # not a family module, any stage
        self.assertEqual(count(after, 169), 3)
        _, after = self.lq('ubocomp_nobuffer', 1 << 31, text=VS_CUT)       # nothing to rewrite
        self.assertIsNone(after)

    def test_and3_classes_are_nulled_one_by_one(self):
        # (DKB_LQ_AND3_CLASS, ANDs nulled, comp / round / test nulled); the plain &3 (feeds an add only) is never nulled
        for classes, nulled, per in (('comp', 1, (1, 0, 0)), ('round', 1, (0, 1, 0)), ('test', 1, (0, 0, 1)),
                                     ('comp,test', 2, (1, 0, 1)), ('comp,round,test', 3, (1, 1, 1))):
            before, after = self.lq('and3_' + classes.replace(',', '_'), 1 << 30, text=VS_AND3,
                                    env={'DKB_LQ_NOFLAGS_ONLY': '3', 'DKB_LQ_AND3_CLASS': classes})
            self.assertEqual((count(before, 199), count(after, 199), count(after, 83)), (4, 4 - nulled, nulled), classes)
            self.assertRegex(self.last_info, r'noflags: %d of 4 OpBitwiseAnd nulled \(&3 classes nulled: comp %d, round %d, test %d; '
                                             r'unclassified left 1\)' % ((nulled,) + per))
        # without the class list every &3 goes, as before
        _, after = self.lq('and3_all', 1 << 30, text=VS_AND3, env={'DKB_LQ_NOFLAGS_ONLY': '3'})
        self.assertEqual(count(after, 83), 4)
        # the classes alone (no constant list) still only touch &3
        _, after = self.lq('and3_classes_only', 1 << 30, text=VS_AND3, env={'DKB_LQ_AND3_CLASS': 'comp,round,test'})
        self.assertEqual(count(after, 83), 3)

    @staticmethod
    def run_function(words, x):
        """Interprets the entry function (locals, branches, int and bool ops) and returns what it stores to the Output."""
        seq = ops(words)
        consts = {a[1]: a[2] for op, a in seq if op == 43}
        for op, a in seq:   # bool constants and composites of them (the tables of and3exact)
            if op == 41: consts[a[1]] = True
            elif op == 42: consts[a[1]] = False
            elif op == 44: consts[a[1]] = [consts[e] for e in a[2:]]
        storage = {a[1]: a[2] for op, a in seq if op == 59}
        M32 = 0xffffffff
        w32 = lambda v: ((v + (1 << 31)) & M32) - (1 << 31)
        blocks, order, cur = {}, [], None
        for op, a in seq:
            if op == 248:
                cur = a[0]
                blocks[cur] = []
                order.append(cur)
            elif cur is not None:
                blocks[cur].append((op, a))
        val, mem, out = dict(consts), {}, None
        label, steps = order[0], 0
        while label is not None:
            nxt = None
            for op, a in blocks[label]:
                steps += 1
                assert steps < 1000
                if op == 61:
                    val[a[1]] = x if storage.get(a[2]) == 1 else mem[a[2]]
                elif op == 62:
                    if storage.get(a[0]) == 3:
                        out = val.get(a[1], out)   # the position's vector is not modelled
                    else: mem[a[0]] = val[a[1]]
                elif op in (124, 83): val[a[1]] = val[a[2]]
                elif op == 126: val[a[1]] = -val[a[2]]
                elif op == 128: val[a[1]] = val[a[2]] + val[a[3]]
                elif op == 199: val[a[1]] = val[a[2]] & val[a[3]]
                elif op == 194: val[a[1]] = w32((val[a[2]] & M32) >> (val[a[3]] & 31))
                elif op == 196: val[a[1]] = w32(val[a[2]] << (val[a[3]] & 31))
                elif op == 130: val[a[1]] = w32(val[a[2]] - val[a[3]])
                elif op == 77: val[a[1]] = val[a[2]][val[a[3]]]
                elif op == 168: val[a[1]] = not val[a[2]]
                elif op == 170: val[a[1]] = val[a[2]] == val[a[3]]
                elif op == 171: val[a[1]] = val[a[2]] != val[a[3]]
                elif op == 172: val[a[1]] = (val[a[2]] & M32) > (val[a[3]] & M32)
                elif op == 169: val[a[1]] = val[a[3]] if val[a[2]] else val[a[4]]
                elif op == 249: nxt = a[0]
                elif op == 250: nxt = a[1] if val[a[0]] else a[2]
                elif op == 253: nxt = None
            label = nxt
        return out

    def test_and3exact_computes_the_same_value_for_every_residue(self):
        before, after = self.lq('and3exact', 512, text=VS_EX3)
        self.assertEqual((count(before, 199), count(before, 171), count(before, 172)), (1, 1, 1))
        self.assertEqual((count(after, 199), count(after, 171), count(after, 172)), (0, 0, 0))   # no AND, no compare left at the sites
        self.assertEqual((count(after, 77), count(after, 194), count(after, 196), count(after, 130)), (2, 1, 1, 1))
        self.assertRegex(self.last_info, r'and3exact: 1 test-class ANDs rewritten, 2 compares replaced')
        for x in list(range(-20, 40)) + [0x7fffffff, -0x80000000, -1, 123456789, -987654321]:
            self.assertEqual(self.run_function(after, x), self.run_function(before, x), x)
        # the four residues really occur
        self.assertEqual({x & 3 for x in range(-8, 8)}, {0, 1, 2, 3})

    def test_rotexact_replaces_the_weighted_sum_by_a_pick(self):
        nan, inf, tiny = float('nan'), float('inf'), 1.4e-45
        sets = [(1.5, -2.25, 3.0, 4.5), (nan, 7.0, inf, -inf), (tiny, nan, 0.0, -0.0), (1e30, -1e-30, nan, 5.0)]
        for forward in (False, True):
            before, after = self.lq('rotexact_%d' % forward, 1024, text=rot_module(forward))
            self.assertGreater(count(before, 132), 0)
            self.assertEqual((count(after, 132), count(after, 77)), (0, 1), forward)       # the multiplies are gone, one pick is left
            self.assertRegex(self.last_info, r'rotexact: 1 weighted sums replaced by a pick')
            for x in list(range(-5, 12)) + [0x7fffffff, -0x80000000]:
                for fs in sets:
                    self.assertEqual(run_rot(after, x, fs), run_rot(before, x, fs), (forward, x, fs))
            # the pick really returns the word whose weight is hot: r = 2 picks f2, whatever the others hold (plus x + 1 = 3)
            self.assertEqual(run_rot(after, 2, (1.0, 2.0, 3.0, 4.0)), (struct.unpack('<I', struct.pack('<f', 3.0))[0] + 3) & 0xffffffff)

    def test_loadexact_loads_only_the_selected_word(self):
        nan, inf = float('nan'), float('inf')
        sets = [[1.5 * (i + 1) for i in range(16)], [nan, inf, -inf, 0.0, -0.0, 1.4e-45, 2.0, 3.0] + [float(i) for i in range(8)]]
        for forward in (False, True):
            before, after = self.lq('loadexact_%d' % forward, 2048, text=rot_module(forward, buf=True))
            self.assertEqual((count(before, 65), count(after, 65)), (4, 1))                # three of the four buffer loads are gone
            self.assertEqual(count(after, 132), 0)                                          # and so are the multiplies
            self.assertRegex(self.last_info, r'loadexact: 1 weighted sums replaced by a single load')
            for x in list(range(-5, 20)) + [0x7fffffff, -0x80000000]:
                for u in sets:
                    got = run_rot(after, x, (0, 0, 0, 0), u)
                    self.assertEqual(len(run_rot.loads), 1)                                  # only the picked word is read
                    self.assertEqual(got, run_rot(before, x, (0, 0, 0, 0), u), (forward, x))
            # r = 2 picks the word at (x & 12) + 2
            self.assertEqual(run_rot(after, 6, (0, 0, 0, 0), sets[0]), (struct.unpack('<I', struct.pack('<f', sets[0][6]))[0] + 7) & 0xffffffff)

    def test_loadexact_leaves_words_that_are_not_buffer_loads_alone(self):
        _, after = self.lq('loadexact_inputs', 2048, text=rot_module(False))   # the words are Input loads
        self.assertIsNone(after)

    # The fold (kind 4096, gfx_shader.c lq_fold): exactly two test-class &3 ANDs are nulled; any other count leaves the module alone.
    VS_FOLD = VS_AND3.replace("        %tot = OpIAdd %uint %s2 %sel\n",
                              "        %an5 = OpBitwiseAnd %uint %ia %uint_3\n"
                              "         %e5 = OpIEqual %bool %an5 %uint_0\n"
                              "       %sel5 = OpSelect %uint %e5 %uint_1 %uint_0\n"
                              "       %tot0 = OpIAdd %uint %s2 %sel5\n"
                              "        %tot = OpIAdd %uint %tot0 %sel\n")

    def test_fold_nulls_the_two_test_class_ands_and_nothing_else(self):
        self.assertIn('%an5', self.VS_FOLD)
        for env in ({}, {'DKB_LQ_NOFLAGS_ONLY': 'ff', 'DKB_LQ_AND3_CLASS': 'comp'}):   # the environment does not change it
            before, after = self.lq('fold', 4096, text=self.VS_FOLD, env=env)
            self.assertEqual(count(before, 199) - count(after, 199), 2)
            self.assertEqual(count(after, 83), count(before, 83) + 2)
            self.assertRegex(self.last_info, r'noflags: 2 of [0-9]+ OpBitwiseAnd nulled .*test 2')

    @staticmethod
    def fold_site_module(both_branch=False):
        """VS_FOLD with a selection construct on `an3 != 3` (the first test-class AND reaches a branch condition); the second one, an5,
        reaches the output only, unless both_branch adds a construct on it as well."""
        def branch(name, cmp_src, tag):
            return (f"        %bc{tag} = OpINotEqual %bool {cmp_src} %uint_3\n               OpSelectionMerge %mrg{tag} None\n"
                    f"               OpBranchConditional %bc{tag} %thn{tag} %mrg{tag}\n       %thn{tag} = OpLabel\n               OpBranch %mrg{tag}\n"
                    f"      %mrg{tag} = OpLabel\n")
        text = SpirvOpt.VS_FOLD
        marker = "       %tot0 = OpIAdd %uint %s2 %sel5\n"
        assert marker in text
        extra = branch('a', '%an3', 'a') + (branch('b', '%an5', 'b') if both_branch else '')
        return text.replace(marker, extra + marker)

    @staticmethod
    def fold_exact_module(arm):
        """fold_site_module with a function-local float stored before the construct on `an3 != 3` and in its arm (`arm`: pure, or a store to
        an Output), loaded after the merge and stored to the output."""
        text = SpirvOpt.fold_site_module()
        assert '%thn' in text
        text = text.replace("     %ptr_in = OpTypePointer Input %float\n", "     %ptr_in = OpTypePointer Input %float\n     %ptr_fn = OpTypePointer Function %float\n")
        text = text.replace("      %entry = OpLabel\n", "      %entry = OpLabel\n         %lv = OpVariable %ptr_fn Function\n")
        text = text.replace("        %bca = OpINotEqual", "               OpStore %lv %b\n        %bca = OpINotEqual")
        text = text.replace("       %thna = OpLabel\n", "       %thna = OpLabel\n" + arm)
        text = text.replace("      %mrga = OpLabel\n", "      %mrga = OpLabel\n        %lvl = OpLoad %float %lv\n               OpStore %out0 %lvl\n")
        return text

    def test_fold_site_exact_makes_the_selector_branches_selects_and_nulls_nothing(self):
        text = self.fold_exact_module("               OpStore %lv %a\n")
        before, after = self.lq('fold_exact', 4096 | 3 << 16, text=text)
        self.assertEqual((count(before, 250), count(after, 250), count(after, 247)), (1, 0, 0))   # the branch and its merge are gone
        self.assertEqual(count(before, 199), count(after, 199))                                       # no AND is nulled
        self.assertGreater(count(after, 169), count(before, 169))                                     # the arm's store became a select
        self.assertIn('fold site exact: 1 selection constructs', self.last_info)
        # an arm that stores to an Output is not pure: fail closed
        _, impure = self.lq('fold_exact_impure', 4096 | 3 << 16, text=self.fold_exact_module("               OpStore %out0 %a\n"))
        self.assertIsNone(impure)
        # both ANDs reaching branches, or none: the AND of each kind cannot be told apart, so fail closed
        for t in (self.fold_site_module(both_branch=True), SpirvOpt.VS_FOLD):
            _, other = self.lq('fold_exact_other', 4096 | 3 << 16, text=t)
            self.assertIsNone(other)

    def test_fold_site_nulls_only_the_and_that_reaches_a_branch_or_the_other(self):
        text = self.fold_site_module()
        _, both = self.lq('fold_site_both', 4096, text=text)
        self.assertEqual(count(both, 199), count(self.lq('fold_site_ref', 4096, text=text)[0], 199) - 2)
        for kinds, word, want in ((4096 | 1 << 16, 'export', 'reaches no branch condition'), (4096 | 2 << 16, 'branch', 'reaches the r != 3 branch conditions')):
            before, after = self.lq('fold_site_' + word, kinds, text=text)
            self.assertEqual(count(before, 199) - count(after, 199), 1, word)               # one AND is nulled, not two
            self.assertIn(want, self.last_info)
            self.assertIn('test 1', self.last_info)
        # export and branch null different ANDs
        _, exp = self.lq('fold_site_exp2', 4096 | 1 << 16, text=text)
        _, brn = self.lq('fold_site_brn2', 4096 | 2 << 16, text=text)
        self.assertNotEqual(exp, brn)
        # both ANDs reach a branch: the sites cannot be told apart, so the module is left alone
        for kinds in (4096 | 1 << 16, 4096 | 2 << 16):
            _, after = self.lq('fold_site_inseparable_%d' % kinds, kinds, text=self.fold_site_module(both_branch=True))
            self.assertIsNone(after)
        _, after = self.lq('fold_site_no_branch', 4096 | 2 << 16, text=SpirvOpt.VS_FOLD)       # neither reaches a branch
        self.assertIsNone(after)

    def test_the_default_fold_rules_are_the_seven_known_ones(self):
        src = (ROOT / 'runtime' / 'gfx_shader.c').read_text(encoding='utf-8')
        body = src[src.index('static const LqFoldRule lq_fold_rules[]'):]
        body = body[:body.index('};')]
        got = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r'\{0x([0-9a-f]{16})ull, (\d+), (\d+)\}', body)}
        whole = {o: (0, 0) for o in ('67a23e6f132386f9', 'e2bf75f927acd249', '3a10ce1a4f3cdf0b', '8475bd9dadb18f45', '4645f0dd7b4502a2', '7a76ae8de5016fd9')}
        self.assertEqual(got, {**whole, '1ee77eac998beea5': (4, 31)})                # Fort: only the ANDs reaching construct 31
        # the diagnostic site pick applies to the stages DKB_LQ_FOLD_ORIGINS names, never to the defaults by themselves
        self.assertIn('if (df_match_origin(&extra, origin) >= 0) {', src)

    def test_fold_site_construct_nulls_the_ands_that_reach_a_named_construct(self):
        kinds = 4096 | 262144
        text = self.fold_site_module().replace('Location 9', 'Location 8')              # a module with no Location 9 input (Fort's shape)
        before, after = self.lq('fold_construct0', kinds, text=text, env={'DKB_LQ_FOLD_CONSTRUCT': '0'}, force=1)
        self.assertEqual(count(before, 199) - count(after, 199), 1)                    # only the AND that reaches construct 0
        self.assertIn('fold site construct 0: nulled the 1 test-class', self.last_info)
        _, none = self.lq('fold_construct5', kinds, text=text, env={'DKB_LQ_FOLD_CONSTRUCT': '5'}, force=1)   # no such construct: nothing changes
        self.assertIsNone(none)
        _, unforced = self.lq('fold_construct_unforced', kinds, text=text, env={'DKB_LQ_FOLD_CONSTRUCT': '0'}, force=0)   # needs the vertex force
        self.assertIsNone(unforced)
        _, plain = self.lq('fold_construct_nobranch', kinds, text=SpirvOpt.VS_FOLD.replace('Location 9', 'Location 8'), env={'DKB_LQ_FOLD_CONSTRUCT': '0'}, force=1)
        self.assertIsNone(plain)                                                         # no construct at all

    def test_fold_fails_closed_without_the_pattern(self):
        _, after = self.lq('fold_one', 4096, text=VS_AND3)                       # one test-class AND, not two
        self.assertIsNone(after)
        _, after = self.lq('fold_frag', 4096)                                    # not a vertex module
        self.assertIsNone(after)
        _, after = self.lq('fold_loc8', 4096, text=self.VS_FOLD.replace('Location 9', 'Location 8'))   # no input at Location 9
        self.assertIsNone(after)
        _, after = self.lq('fold_masked', 4096 | 128, text=self.VS_FOLD)         # vsmask set: the bit is ignored
        self.assertIsNotNone(after)
        self.assertEqual(count(after, 199), count(self.lq('fold_ref', 128, text=self.VS_FOLD)[1], 199))

    def test_light_quad_skip_decision_for_folded_and_unfolded_draws(self):
        src = args.out / 'skip.cpp'
        src.write_text(r'''
#include <cstdio>
#include "lq_skip.h"
int main() {
    for (int e = 0; e < 2; e++)
        for (int m = 0; m < 3; m++)
            for (int f = 0; f < 2; f++) std::printf("%d%d%d=%d ", e, m, f, lq_skip_decision(e, m, f));
}
''', encoding='utf-8')
        exe = args.out / 'skip.exe'
        subprocess.run([str(args.cxx), '-std=c++20', '-Wall', '-Wextra', '-Werror', f'-I{ROOT / "runtime"}', str(src), '-o', str(exe)],
                       check=True, capture_output=True, text=True, timeout=120)
        got = dict(l.split('=') for l in subprocess.run([str(exe)], capture_output=True, text=True, timeout=30).stdout.split())
        for e in (0, 1):
            for m in (0, 1, 2):
                for f in (0, 1):
                    self.assertEqual(got['%d%d%d' % (e, m, f)], str(int(e and m and not f)), (e, m, f))
        # a folded draw is drawn; a matched draw that was not folded (known stage, family) is skipped as before
        self.assertEqual((got['121'], got['110'], got['120'], got['101']), ('0', '1', '1', '0'))

    def test_fold_fail_safe_and_refold_decisions(self):
        # The fold must hold whichever path loaded the module (gfx_shader.c gfx_shader_fold_fix, vk_exec_draw.c lq_fold_missing).
        src = args.out / 'fold_decisions.cpp'
        src.write_text(r'''
#include <cstdio>
#include "lq_skip.h"
int main() {
    for (int p = 0; p < 2; p++)
        for (int r = 0; r < 2; r++)
            for (int f = 0; f < 2; f++) std::printf("m%d%d%d=%d ", p, r, f, lq_fold_missing_decision(p, r, f));
    for (int r = 0; r < 2; r++)
        for (int f = 0; f < 2; f++)
            for (int t = 0; t < 2; t++) std::printf("f%d%d%d=%d ", r, f, t, lq_refold_decision(r, f, t));
}
''', encoding='utf-8')
        exe = args.out / 'fold_decisions.exe'
        subprocess.run([str(args.cxx), '-std=c++20', '-Wall', '-Wextra', '-Werror', f'-I{ROOT / "runtime"}', str(src), '-o', str(exe)],
                       check=True, capture_output=True, text=True, timeout=120)
        got = dict(l.split('=') for l in subprocess.run([str(exe)], capture_output=True, text=True, timeout=30).stdout.split())
        for p in (0, 1):
            for r in (0, 1):
                for f in (0, 1):   # skip only on the platform, for a stage with a rule, when its module was not folded
                    self.assertEqual(got['m%d%d%d' % (p, r, f)], str(int(p and r and not f)), (p, r, f))
        for r in (0, 1):
            for f in (0, 1):
                for t in (0, 1):   # a cached module of a stage with a rule is folded once, when it is not folded yet
                    self.assertEqual(got['f%d%d%d' % (r, f, t)], str(int(r and not f and not t)), (r, f, t))
        self.assertEqual((got['m111'], got['m110'], got['f100'], got['f101']), ('0', '1', '1', '0'))   # Fort/Tower: skipped, or refolded once

    def test_draw_filter_parsing_and_skip_decision(self):
        src = args.out / 'draw_filter.cpp'
        src.write_text(r'''
#include <cstdio>
#include "draw_filter.h"
static void origins(const char *list, const uint64_t *probe, int n) {
    DfOrigins o;
    int c = df_parse_origins(&o, list);
    std::printf("list %s -> %d:", list, c);
    for (int i = 0; c > 0 && i < n; i++) std::printf(" %d", df_match_origin(&o, probe[i]));
    std::printf("\n");
}
int main() {
    const uint64_t p[] = {0x352c48a38e39d4d6ull, 0x352c48a38e39d4d7ull, 0x352c000000000000ull, 0x0ull, 0xcae0ef8e9fe9147full};
    origins("352c48a38e39d4d6", p, 5);
    origins("352c48a3,CAE0EF8E", p, 5);
    origins("3", p, 5);
    origins("352c48a38e39d4d6,", p, 5);
    origins("352c48a38e39d4d60", p, 5);
    origins("zz", p, 5);
    origins(",", p, 5);
    DfOrigins w;
    std::printf("whole %d %d %d %d %d\n", df_parse_whole_origins(&w, "4645f0dd7b4502a2,8475bd9dadb18f45"), df_match_origin(&w, 0x8475bd9dadb18f45ull),
                df_match_origin(&w, 0x8475bd9dadb18f46ull), df_parse_whole_origins(&w, "4645f0dd"), df_parse_whole_origins(&w, "4645f0dd7b4502a2,"));
    DfBisect multi;
    int c1 = df_parse_bisect_list(&multi, "f:3,ff:40,8000000000000000:8000000000000000");
    std::printf("terms %d %d %d %d %d %d %d\n", c1, df_bisect_hit(&multi, 0x13), df_bisect_hit(&multi, 0x140), df_bisect_hit(&multi, 0x9000000000000000ull),
                df_bisect_hit(&multi, 0x14), df_bisect_hit(&multi, 0), (int)multi.n);
    DfBisect bad;
    std::printf("badterms %d %d %d %d %d %d\n", df_parse_bisect_list(&bad, "f:3,"), df_parse_bisect_list(&bad, "f:3,x:1"),
                df_parse_bisect_list(&bad, "f:10"), df_parse_bisect_list(&bad, "f:3,,f:4"), df_parse_bisect_list(&bad, ",f:3"),
                df_parse_bisect_list(&bad, "f:3,f:3,f:3,f:3,f:3,f:3,f:3,f:3,f:3,f:3,f:3,f:3,f:3,f:3,f:3,f:3,f:3"));
    uint64_t m, v;
    std::printf("bisect %d %d %d %d\n", df_parse_bisect("8000000000000000:8000000000000000", &m, &v), df_parse_bisect("f:1", &m, &v),
                df_parse_bisect("f:10", &m, &v), df_parse_bisect("nonsense", &m, &v));
    DfOrigins o;
    df_parse_origins(&o, "aaaa");
    DfBisect none, vb, fb;
    df_parse_bisect_list(&none, "");
    df_parse_bisect_list(&vb, "f:3");
    df_parse_bisect_list(&fb, "ff:4f");
    int why;
    int r1 = df_skip(&o, &none, &none, 0xaaaa000000000001ull, 0, &why), w1 = why;
    int r2 = df_skip(&o, &none, &none, 1, 0xaaaa000000000002ull, &why), w2 = why;
    int r3 = df_skip(&o, &none, &none, 1, 2, &why), w3 = why;
    int r4 = df_skip(&o, &vb, &none, 0x13, 0x4f, &why), w4 = why;
    int r5 = df_skip(&o, &vb, &none, 0, 0x4f, &why), w5 = why;
    int r6 = df_skip(&o, &none, &fb, 0x10, 0x14f, &why), w6 = why;
    int r7 = df_skip(&o, &none, &fb, 0x10, 0, &why), w7 = why;
    int r8 = df_skip(&o, &vb, &fb, 0x13, 0x4f, &why), w8 = why;
    int r9 = df_skip(&o, &vb, &fb, 0x10, 0x4e, &why), w9 = why;
    std::printf("skip %d%d %d%d %d%d %d%d %d%d %d%d %d%d %d%d %d%d\n", r1, w1, r2, w2, r3, w3, r4, w4, r5, w5, r6, w6, r7, w7, r8, w8, r9, w9);
}
''', encoding='utf-8')
        exe = args.out / 'draw_filter.exe'
        subprocess.run([str(args.cxx), '-std=c++20', '-Wall', '-Wextra', '-Werror', f'-I{ROOT / "runtime"}', str(src), '-o', str(exe)],
                       check=True, capture_output=True, text=True, timeout=120)
        out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30).stdout.splitlines()
        self.assertEqual(out[0], 'list 352c48a38e39d4d6 -> 1: 0 -1 -1 -1 -1')            # a whole origin matches only itself
        self.assertEqual(out[1], 'list 352c48a3,CAE0EF8E -> 2: 0 0 -1 -1 1')              # prefixes, any case
        self.assertEqual(out[2], 'list 3 -> 1: 0 0 0 -1 -1')                              # one digit; origin 0 never matches
        self.assertEqual(out[3], 'list 352c48a38e39d4d6, -> -1:')                         # an empty token is refused
        self.assertEqual(out[4], 'list 352c48a38e39d4d60 -> -1:')                         # 17 digits
        self.assertEqual(out[5], 'list zz -> -1:')
        self.assertEqual(out[6], 'list , -> -1:')
        self.assertEqual(out[10], 'bisect 1 1 0 0')                                        # the value must lie inside the mask
        self.assertIn('whole 2 1 -1 -1 -1', out)                                   # whole 16-digit origins only (DKB_LQ_FOLD_ORIGINS)
        self.assertEqual(out[8], 'terms 3 1 1 1 0 0 3')        # f:3 hits 0x13, ff:40 hits 0x140, the top-bit term hits; origin 0 never
        self.assertEqual(out[9], 'badterms -1 -1 -1 -1 -1 -1')    # trailing comma, bad hex, value outside the mask, empty term, leading comma, 17 terms
        self.assertEqual(out[11], 'skip 11 11 00 12 00 14 00 16 00')   # list on the vertex or the fragment origin; each bisect alone; both; neither; an origin of 0 never matches

    def test_vsblock_forces_selection_constructs_and_removes_the_dead_path(self):
        text = VS_DIAMOND.replace('@THEN@', ARM_PURE)
        before, after = self.lq('vsblock_auto', 8192, text=text, env={'DKB_LQ_VSBLOCK': '0-0'})
        self.assertEqual((count(before, 250), count(after, 250)), (1, 0))
        self.assertEqual(count(before, 62) - count(after, 62), 1)                       # the arm's store is gone
        self.assertRegex(self.last_info, r'vsblock: 1 selection constructs in the module, 1 of 0..0 forced')
        _, after_t = self.lq('vsblock_true', 8192, text=text, env={'DKB_LQ_VSBLOCK': '0-3:t'})
        self.assertEqual((count(after_t, 250), count(after_t, 62)), (0, count(before, 62)))   # the arm is kept, the way past it goes
        _, after_f = self.lq('vsblock_false', 8192, text=text, env={'DKB_LQ_VSBLOCK': '0-0:f'})
        self.assertEqual(count(after_f, 250), 0)
        self.assertEqual(count(after_f, 62), count(after, 62))
        _, after_none = self.lq('vsblock_none', 8192, text=text, env={'DKB_LQ_VSBLOCK': '5-9'})   # no such construct: nothing changes
        self.assertIsNone(after_none)

    def test_vertex_kinds_need_location_9_unless_forced(self):
        diamond = VS_DIAMOND.replace('@THEN@', ARM_PURE).replace('Location 9', 'Location 8')
        trig = VS_TRIG.replace('Location 9', 'Location 8')
        for kind, env, text in ((8192, {'DKB_LQ_VSBLOCK': '0-0'}, diamond), (1 << 30, {}, trig), (1 << 29, {}, trig), (128 | 1 << 8, {}, diamond),
                                (32, {}, diamond), (16, {}, diamond)):
            _, plain = self.lq('force_off_%d' % kind, kind, text=text, env=env)
            self.assertIsNone(plain, kind)
            _, forced = self.lq('force_on_%d' % kind, kind, text=text, env=env, force=1)
            self.assertIsNotNone(forced, kind)
        # the fold is the exception: its pattern is the check, so force does not widen it
        _, fold = self.lq('force_fold', 4096, text=self.VS_FOLD.replace('Location 9', 'Location 8'), force=1)
        self.assertIsNone(fold)

    def test_rotexact_leaves_a_sum_with_a_non_one_hot_weight_alone(self):
        text = rot_module(False).replace('%w1 = OpSelect %int %e1 %int_1 %int_0', '%w1 = OpSelect %int %e1 %int_2 %int_0')
        _, after = self.lq('rotexact_not_one_hot', 1024, text=text)
        self.assertIsNone(after)                      # a weight of 2 is not a pick: the module is left alone

    def test_and3exact_leaves_other_ands_alone(self):
        # the comp / round / unclassified &3 of VS_AND3 are not test-class: only the test one (an3) changes
        before, after = self.lq('and3exact_other', 512, text=VS_AND3)
        self.assertEqual(count(before, 199) - count(after, 199), 1)
        self.assertEqual(count(after, 77), 1)

    def test_ldcfix_removes_the_branch_and_keeps_the_value(self):
        for name, branch in (('arm_true', LDC_BR_ARM_TRUE), ('arm_false', LDC_BR_ARM_FALSE)):
            text = (VS_LDC.replace('@CMP@', LDC_CMP).replace('@BRANCH@', branch).replace('@ARM@', LDC_ARM))
            if name == 'arm_false':
                text = text.replace('%c = OpLogicalNot %bool %cmp', '%c = OpCopyObject %bool %cmp')   # the arm runs when c is false
            before, after = self.lq('ldcfix_' + name, 256, text=text)
            self.assertEqual((count(before, 250), count(before, 247)), (1, 1), name)
            self.assertEqual((count(after, 250), count(after, 247)), (0, 0), name)
            self.assertEqual(count(after, 248), count(before, 248) - 1, name)         # the arm block is gone
            self.assertRegex(self.last_info, r'ldcfix: 1 data-dependent branches removed')
            for x in (0, 1, 2, 3, 4, 5, 6, 7, -1, -2):        # all four alignments, several rows, negative values
                self.assertEqual(self.run_function(after, x), self.run_function(before, x), (name, x))

    def test_ldcfix_leaves_arms_with_side_effects_and_other_conditions(self):
        text = VS_LDC.replace('@CMP@', LDC_CMP).replace('@BRANCH@', LDC_BR_ARM_TRUE).replace('@ARM@', LDC_ARM_OUT)
        _, after = self.lq('ldcfix_out_store', 256, text=text)            # the arm stores to an Output: not pure
        self.assertIsNone(after)
        text = (VS_LDC.replace('@CMP@', '        %cmp = OpUGreaterThan %bool %x %int_1').replace('@BRANCH@', LDC_BR_ARM_TRUE)
                .replace('@ARM@', LDC_ARM))
        _, after = self.lq('ldcfix_not_and3', 256, text=text)             # the condition is not about x & 3
        self.assertIsNone(after)

    def test_selfix_runs_a_named_triangle_both_ways_with_a_select(self):
        # ldcfix refuses a condition that is not about x & 3; selfix takes whatever construct it is told to, and only that one
        text = (VS_LDC.replace('@CMP@', '        %cmp = OpUGreaterThan %bool %x %int_1').replace('@BRANCH@', LDC_BR_ARM_TRUE).replace('@ARM@', LDC_ARM))
        before, after = self.lq('selfix_named', 16384, text=text, env={'DKB_LQ_SELFIX': '0'})
        self.assertEqual((count(before, 250), count(after, 250), count(after, 247)), (1, 0, 0))
        self.assertRegex(self.last_info, r'selfix: 1 data-dependent branches removed')
        for x in (-5, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 100, 0x7fffffff, -0x80000000):   # the condition goes both ways
            self.assertEqual(self.run_function(after, x), self.run_function(before, x), x)
        _, other = self.lq('selfix_other_index', 16384, text=text, env={'DKB_LQ_SELFIX': '1-3,7'})   # no such construct named: unchanged
        self.assertIsNone(other)
        _, range_hit = self.lq('selfix_range', 16384, text=text, env={'DKB_LQ_SELFIX': '2,0-0'})
        self.assertIsNotNone(range_hit)
        text = VS_LDC.replace('@CMP@', LDC_CMP).replace('@BRANCH@', LDC_BR_ARM_TRUE).replace('@ARM@', LDC_ARM_OUT)
        _, out_arm = self.lq('selfix_out_store', 16384, text=text, env={'DKB_LQ_SELFIX': '0'})   # an arm that stores to an Output stays a branch
        self.assertIsNone(out_arm)

    def test_mul2_reduces_the_selector_product_to_its_low_two_bits(self):
        rnd = __import__('random').Random(5)
        fl = lambda n: struct.unpack('<f', struct.pack('<I', n & 0x7f7fffff))[0]
        sets = [(fl(rnd.getrandbits(31)), fl(rnd.getrandbits(31)), 0.0 if i % 2 else 1.5) for i in range(40)] + [(1.0, 2.0, 0.0), (0.0, 0.0, 0.0), (3.5, 7.0, -0.0)]
        for variant in ('product', 'direct'):
            before, after = self.lq('mul2_' + variant, 32768, text=mul2_module(variant))
            self.assertEqual(count(before, 196), 1)
            self.assertEqual((count(after, 196), count(after, 132)), (0, 1 if variant == 'product' else 0), variant)   # the shifted chain is gone
            self.assertRegex(self.last_info, r'mul2: 1 &3 ANDs now read')
            for x in list(range(-6, 40)) + [0x7fffffff, -0x80000000, 65535, 65536, 0x12345678]:
                for fs in sets:
                    self.assertEqual(run_rot(after, x, fs), run_rot(before, x, fs), (variant, x, fs))
            self.assertEqual({run_rot(before, x, (1.0, 2.0, 0.0)) & 3 for x in range(16)} <= {0, 1, 2, 3}, True)

    def test_mul2_fails_closed_without_the_exact_pattern(self):
        for variant in ('falsezero', 'noshift'):
            _, after = self.lq('mul2_' + variant, 32768, text=mul2_module(variant))
            self.assertIsNone(after, variant)
        _, after = self.lq('mul2_not_a_selector', 32768, text=rot_module(False))     # no test-class &3 of that form
        self.assertIsNone(after)

    def test_mul2_keeps_the_select_for_its_other_users(self):
        before, after = self.lq('mul2_other_use', 32768, text=mul2_module('other_use'))
        self.assertEqual(count(after, 196), 1)                         # the original select still feeds the whole-value user
        for x in (-3, 0, 1, 2, 5, 77, 0x7fffffff):
            for fs in ((1.0, 2.0, 0.0), (0.5, 7.25, 0.0), (3.0, 4.0, 1.5)):
                self.assertEqual(run_rot(after, x, fs), run_rot(before, x, fs), (x, fs))

    def test_generic_vertex_kinds_leave_other_modules_alone(self):
        for kind in (1 << 28, 1 << 29, 1 << 30):
            _, after = self.lq('gen_loc8_%d' % kind, kind, text=VS_TRIG.replace('Location 9', 'Location 8'))
            self.assertIsNone(after)
            _, after = self.lq('gen_frag_%d' % kind, kind)
            self.assertIsNone(after)

    def test_draw_fastpath_caches(self):
        src = args.out / 'draw_cache.cpp'
        src.write_text(r'''
#include <cstdio>
#include <thread>
#include <vector>
#include "draw_cache.h"
#include "origin_cache.h"
int main() {
    int a = 0, b = 0, c = 0, a2 = 0;
    DcMemo m = {};
    int v = -1;
    std::printf("empty=%d ", dc_memo_get(&m, &a, &b, &c, 0, &v));
    dc_memo_put(&m, &a, &b, &c, 7, 2);
    std::printf("hit=%d:%d ", dc_memo_get(&m, &a, &b, &c, 7, &v), v);
    std::printf("prog=%d shader=%d frag=%d key=%d ", dc_memo_get(&m, &a2, &b, &c, 7, &v), dc_memo_get(&m, &a, &a2, &c, 7, &v),
                dc_memo_get(&m, &a, &b, &a2, 7, &v), dc_memo_get(&m, &a, &b, &c, 8, &v));
    dc_memo_put(&m, &a, &b, &c, 7, 0);
    std::printf("zero=%d:%d ", dc_memo_get(&m, &a, &b, &c, 7, &v), v);
    DcIndex x = {};
    std::printf("first=%d same=%d ", dc_index_redundant(&x, 1, 16, 1), dc_index_redundant(&x, 1, 16, 1));
    std::printf("buf=%d ", dc_index_redundant(&x, 2, 16, 1));
    std::printf("off=%d ", dc_index_redundant(&x, 2, 32, 1));
    std::printf("type=%d ", dc_index_redundant(&x, 2, 32, 0));
    std::printf("again=%d ", dc_index_redundant(&x, 2, 32, 0));
    dc_index_reset(&x);
    std::printf("reset=%d after=%d\n", dc_index_redundant(&x, 2, 32, 0), dc_index_redundant(&x, 2, 32, 0));
    // origin cache: a miss, a put, a hit; two addresses in one slot keep the latest only; threads never read a wrong name
    static OriginCache oc;
    uint64_t o = 0;
    char buf[4096];
    std::printf("miss=%d ", oc.get(buf, &o));
    oc.put(buf, 0x1234);
    std::printf("hit=%d:%llx ", oc.get(buf, &o), (unsigned long long)o);
    const void *p2 = nullptr;
    for (int i = 1; i < 4096 && !p2; i++)
        if (OriginCache::index(buf + i) == OriginCache::index(buf)) p2 = buf + i;
    if (p2) {
        oc.put(p2, 0x99);
        std::printf("evicted=%d new=%d:%llx ", oc.get(buf, &o), oc.get(p2, &o), (unsigned long long)o);
    }
    static OriginCache sh;
    std::atomic<int> bad{0}, stop{0};
    std::vector<std::thread> t;
    for (int k = 0; k < 3; k++)
        t.emplace_back([&] {
            uint64_t r;
            while (!stop)
                for (int i = 0; i < 64; i++) {
                    const void *code = buf + i * 64;
                    if (sh.get(code, &r) && r != (uint64_t)(i + 1)) bad++;
                }
        });
    for (int n = 0; n < 200000; n++) {
        int i = n % 64;
        sh.put(buf + i * 64, (uint64_t)(i + 1));
    }
    stop = 1;
    for (auto &th : t) th.join();
    std::printf("torn=%d\n", bad.load());
}
''', encoding='utf-8')
        exe = args.out / 'draw_cache.exe'
        subprocess.run([str(args.cxx), '-std=c++20', '-Wall', '-Wextra', '-Werror', f'-I{ROOT / "runtime"}', str(src), '-o', str(exe)],
                       check=True, capture_output=True, text=True, timeout=120)
        out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60).stdout
        self.assertIn('empty=0 hit=1:2 prog=0 shader=0 frag=0 key=0 zero=1:0 first=0 same=1 buf=0 off=0 type=0 again=1 reset=0 after=1', out)
        self.assertIn('miss=0 hit=1:1234', out)
        self.assertIn('evicted=0 new=1:99', out)
        self.assertIn('torn=0', out)

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
