"""Offline: runs a captured vertex shader on the CPU, for every (instance, vertex) of a captured draw, and reports dynamic uniform
indices that fall outside the declared array or the bound buffer, non-finite values that reach an index or an integer conversion,
and the final outputs. CPU only: no Vulkan, no GPU.

    python tools/lq_vseval.py <capture dir> [--vertices N] [--instances N] [--out report.txt] [--verbose]

The directory is what the runtime writes with DKB_LQ_CAPTURE_VS (runtime/vk_exec_draw.c lq_capture): vertex.spv, draw.txt, support.bin,
ubo-<slot>.bin, stream-<n>.bin, ssbo-<slot>.bin, index.bin. The interpreter covers what the translator's vertex shaders use: scalar and
vector int, float and bool operations (floats rounded to 32 bits per operation), GLSL.std.450 math, access chains into
Input / Output / Function / Uniform / StorageBuffer variables, structured control flow with phis and locals. A uniform or storage read is
checked against the declared array length at every dynamic index and against the captured bound size at the load; an index that is out
of range is reported and the read returns 0 (robust access), so the run continues. Not modelled: texture sampling, derivatives,
atomics, function calls, matrices in buffers.
"""
import argparse
import math
from pathlib import Path
import re
import struct
import sys

M32 = 0xffffffff


def f32(x):
    try:
        return struct.unpack('<f', struct.pack('<f', x))[0]
    except OverflowError:
        return math.inf if x > 0 else -math.inf
    except struct.error:
        return x


def bits_of_float(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def float_of_bits(b):
    return struct.unpack('<f', struct.pack('<I', b & M32))[0]


def fdiv(a, b):
    try:
        return f32(a / b)
    except ZeroDivisionError:
        if a == 0 or a != a:
            return math.nan
        return math.copysign(math.inf, a) * (math.copysign(1.0, b))


GLSL = {1: 'Round', 2: 'RoundEven', 3: 'Trunc', 4: 'FAbs', 5: 'SAbs', 6: 'FSign', 7: 'SSign', 8: 'Floor', 9: 'Ceil', 10: 'Fract',
        13: 'Sin', 14: 'Cos', 15: 'Tan', 26: 'Pow', 27: 'Exp', 28: 'Log', 29: 'Exp2', 30: 'Log2', 31: 'Sqrt', 32: 'InverseSqrt',
        37: 'FMin', 38: 'UMin', 39: 'SMin', 40: 'FMax', 41: 'UMax', 42: 'SMax', 43: 'FClamp', 44: 'UClamp', 45: 'SClamp',
        46: 'FMix', 48: 'Step', 50: 'Fma', 66: 'Length', 79: 'NMin', 80: 'NMax', 81: 'NClamp'}

# NVN vertex attribute format codes (runtime/vk_pipeline.c vtx_formats): code -> (components, kind, bits)
VTX = {}
for base, n in ((0x01, 1), (0x0D, 2), (0x19, 3), (0x25, 4)):
    kinds = [('unorm', 8), ('snorm', 8), ('uint', 8), ('sint', 8), ('float', 16), ('unorm', 16), ('snorm', 16), ('uint', 16),
             ('sint', 16), ('float', 32), ('uint', 32), ('sint', 32)] if n < 3 else [
        ('unorm', 8), ('snorm', 8), ('uint', 8), ('sint', 8), ('float', 16), ('unorm', 16), ('snorm', 16), ('uint', 16),
        ('sint', 16), ('float', 32), ('uint', 32), ('sint', 32)]
    for k, (kind, bits) in enumerate(kinds):
        VTX[base + k] = (n, kind, bits)


# Scaled formats (integer data read as float; runtime/vk_pipeline.c vtx_formats 0x51..0x68) and the 32-bit packed ones.
SCALED = {}
for base, n in ((0x51, 1), (0x57, 2), (0x5D, 3), (0x63, 4)):
    for k, (kind, bits) in enumerate((('uscaled', 8), ('sscaled', 8), ('uscaled', 16), ('sscaled', 16))):
        SCALED[base + k] = (n, kind, bits)
for base, n in ((0x55, 1), (0x5B, 2), (0x61, 3), (0x67, 4)):
    SCALED[base] = (n, 'uint', 32)
    SCALED[base + 1] = (n, 'sint', 32)
VTX.update(SCALED)
PACKED = {0x3D: 'unorm', 0x3E: 'uint', 0x3F: 'ufloat', 0x69: 'snorm', 0x6A: 'sint', 0x6B: 'uscaled', 0x6C: 'sscaled', 0x9A: 'bgra8'}


def _small_float(bits, exp_bits, man_bits):
    e = bits >> man_bits
    m = bits & ((1 << man_bits) - 1)
    if e == 0:
        return (m / (1 << man_bits)) * 2.0 ** (1 - ((1 << (exp_bits - 1)) - 1))
    if e == (1 << exp_bits) - 1:
        return math.inf if m == 0 else math.nan
    return (1 + m / (1 << man_bits)) * 2.0 ** (e - ((1 << (exp_bits - 1)) - 1))


def decode_packed(fmt, raw):
    kind = PACKED[fmt]
    v = int.from_bytes(raw[:4].ljust(4, bytes(1)), 'little')
    if kind == 'bgra8':
        b, g, r, a = v & 255, (v >> 8) & 255, (v >> 16) & 255, v >> 24
        return [r / 255, g / 255, b / 255, a / 255]
    if kind == 'ufloat':   # B10G11R11: R 11 bits (5+6), G 11, B 10 (5+5)
        return [_small_float(v & 0x7ff, 5, 6), _small_float((v >> 11) & 0x7ff, 5, 6), _small_float((v >> 22) & 0x3ff, 5, 5), 1.0]
    r, g, b, a = v & 1023, (v >> 10) & 1023, (v >> 20) & 1023, v >> 30
    sg = lambda x, bits: x - (1 << bits) if x >> (bits - 1) else x
    if kind == 'unorm': return [r / 1023, g / 1023, b / 1023, a / 3]
    if kind == 'uint' or kind == 'uscaled': return [r, g, b, a]
    if kind == 'snorm': return [max(sg(r, 10) / 511, -1.0), max(sg(g, 10) / 511, -1.0), max(sg(b, 10) / 511, -1.0), max(sg(a, 2), -1.0)]
    return [sg(r, 10), sg(g, 10), sg(b, 10), sg(a, 2)]   # sint, sscaled


def decode_attr(fmt, raw, want_comps):
    """The vec4 a vertex attribute of NVN format fmt gives from its bytes (missing components 0, w 1)."""
    if fmt in PACKED:
        return decode_packed(fmt, raw)
    if fmt not in VTX:
        raise ValueError(f'vertex format {fmt:#x} is not modelled')
    n, kind, bits = VTX[fmt]
    size = bits // 8
    out = []
    for c in range(n):
        b = raw[c * size:(c + 1) * size]
        if len(b) < size:
            b = b + bytes(size - len(b))
        if kind == 'float':
            v = struct.unpack('<e' if bits == 16 else '<f', b)[0]
        elif kind in ('uint', 'unorm', 'uscaled'):
            v = int.from_bytes(b, 'little')
            if kind == 'unorm':
                v = v / ((1 << bits) - 1)
        else:
            v = int.from_bytes(b, 'little', signed=True)
            if kind == 'snorm':
                v = max(v / ((1 << (bits - 1)) - 1), -1.0)
        out.append(v)
    out += [0] * (3 - len(out)) + ([1.0 if kind in ('float', 'unorm', 'snorm') else 1] if len(out) < 4 else [])
    return out[:4]


class Module:
    def __init__(self, words):
        w = list(words)
        if w[0] != 0x07230203:
            raise ValueError('not a SPIR-V module')
        self.types, self.consts, self.vars = {}, {}, {}
        self.names, self.deco, self.mdeco = {}, {}, {}
        self.fn_order, self.funcs = [], {}
        self.entry = None
        self.glsl = None
        pending = []
        i, cur = 5, None
        while i < len(w):
            n, op = w[i] >> 16, w[i] & 0xffff
            a = w[i + 1:i + n]
            i += n
            if op == 5:
                self.names[a[0]] = struct_str(a[1:])
            elif op == 71:
                self.deco.setdefault(a[0], []).append((a[1], a[2:]))
            elif op == 72:
                self.mdeco.setdefault((a[0], a[1]), []).append((a[2], a[3:]))
            elif op == 15:
                if self.entry is None:
                    self.entry = a[1]
            elif op == 11:
                self.glsl = a[0]
            elif op == 19: self.types[a[0]] = ('void',)
            elif op == 20: self.types[a[0]] = ('bool',)
            elif op == 21: self.types[a[0]] = ('int', a[1], a[2])
            elif op == 22: self.types[a[0]] = ('float', a[1])
            elif op == 23: self.types[a[0]] = ('vec', a[1], a[2])
            elif op == 24: self.types[a[0]] = ('mat', a[1], a[2])
            elif op == 28: self.types[a[0]] = ('arr', a[1], a[2])
            elif op == 29: self.types[a[0]] = ('rarr', a[1])
            elif op == 30: self.types[a[0]] = ('struct', list(a[1:]))
            elif op == 32: self.types[a[0]] = ('ptr', a[1], a[2])
            elif op == 33: self.types[a[0]] = ('fn',)
            elif op in (41, 42):
                self.consts[a[1]] = (a[0], op == 41)
            elif op == 43:
                t = self.types[a[0]]
                if t[0] == 'float':
                    v = float_of_bits(a[2]) if t[1] == 32 else struct.unpack('<d', struct.pack('<II', a[2], a[3]))[0]
                else:
                    v = a[2] | (a[3] << 32 if t[1] == 64 and len(a) > 3 else 0)
                    v &= (1 << t[1]) - 1
                self.consts[a[1]] = (a[0], v)
            elif op == 44:
                self.consts[a[1]] = (a[0], ('composite', list(a[2:])))
            elif op == 46:
                self.consts[a[1]] = (a[0], ('null',))
            elif op == 59:
                self.vars[a[1]] = (a[0], a[2])
                if cur is not None:
                    cur['locals'].append(a[1])
            elif op == 54:
                cur = {'id': a[1], 'blocks': {}, 'order': [], 'locals': []}
                self.funcs[a[1]] = cur
                block = None
            elif op == 56:
                cur = None
            elif cur is not None:
                if op == 248:
                    block = {'id': a[0], 'ins': []}
                    cur['blocks'][a[0]] = block
                    cur['order'].append(a[0])
                elif block is not None:
                    block['ins'].append((op, a))
        for cid, (t, v) in list(self.consts.items()):
            pass

    def deco_value(self, i, d):
        for k, v in self.deco.get(i, []):
            if k == d:
                return v[0] if v else True
        return None

    def mdeco_value(self, s, m, d):
        for k, v in self.mdeco.get((s, m), []):
            if k == d:
                return v[0] if v else True
        return None

    def zero(self, t):
        ty = self.types[t]
        k = ty[0]
        if k == 'float': return 0.0
        if k == 'int': return 0
        if k == 'bool': return False
        if k in ('vec', 'mat', 'arr'):
            n = ty[2] if k != 'arr' else self.const_int(ty[2])
            return [self.zero(ty[1]) for _ in range(n)]
        if k == 'struct': return [self.zero(m) for m in ty[1]]
        return 0

    def const_int(self, cid):
        return self.consts[cid][1]

    def const_value(self, cid):
        t, v = self.consts[cid]
        if isinstance(v, tuple):
            if v[0] == 'composite':
                return [self.const_value(x) for x in v[1]]
            return self.zero(t)
        return v

    def size_of(self, t):
        ty = self.types[t]
        k = ty[0]
        if k == 'float': return ty[1] // 8
        if k == 'int': return ty[1] // 8
        if k == 'bool': return 4
        if k == 'vec': return self.size_of(ty[1]) * ty[2]
        return 0


def struct_str(words):
    b = b''.join(struct.pack('<I', x) for x in words)
    return b.split(b'\0')[0].decode('latin1')


class Fault(Exception):
    pass


class Machine:
    def __init__(self, mod, buffers, report):
        self.m, self.buffers, self.rep = mod, buffers, report   # buffers: var id -> bytes (uniform / storage)
        self.glsl = mod.glsl

    # ---- type helpers
    def is_float(self, t): return self.m.types[t][0] == 'float'

    def elem_t(self, t):
        ty = self.m.types[t]
        return ty[1] if ty[0] in ('vec', 'mat', 'arr', 'rarr') else None

    def sc(self, t):
        """The scalar type under vectors."""
        ty = self.m.types[t]
        return self.sc(ty[1]) if ty[0] == 'vec' else t

    # ---- value helpers
    def to_signed(self, v, bits):
        v &= (1 << bits) - 1
        return v - (1 << bits) if v >> (bits - 1) else v

    def lift(self, fn, t, *vals):
        ty = self.m.types[t]
        if ty[0] == 'vec':
            return [fn(*[v[k] if isinstance(v, list) else v for v in vals]) for k in range(ty[2])]
        return fn(*vals)

    # ---- execution of one invocation
    def run(self, inputs, builtins, label=''):
        m = self.m
        self.mem, self.taint, self.vals, self.vtaint = {}, {}, {}, {}
        self.label, self.nonfinite_seen = label, set()
        for vid, (pt, sc) in m.vars.items():
            if sc in (1, 3, 6, 7, 9):
                self.mem[vid] = m.zero(m.types[pt][2])
        for vid, val in inputs.items():
            self.mem[vid] = val
        for vid, val in builtins.items():
            self.mem[vid] = val
        fn = m.funcs[m.entry]
        label_id, prev = fn['order'][0], None
        steps = 0
        while label_id is not None:
            blk = fn['blocks'][label_id]
            # phis first, evaluated together from the predecessor
            pend = {}
            for op, a in blk['ins']:
                if op == 245:
                    for k in range(2, len(a), 2):
                        if a[k + 1] == prev:
                            pend[a[1]] = (self.get(a[k]), self.vtaint.get(a[k]))
            for k, (v, t) in pend.items():
                self.vals[k] = v
                if t: self.vtaint[k] = t
            nxt = None
            for op, a in blk['ins']:
                steps += 1
                if steps > 2_000_000:
                    raise Fault('step limit')
                if op == 245:
                    continue
                r = self.step(op, a)
                if r is not None:
                    kind, target = r
                    if kind == 'br': nxt = target
                    elif kind == 'ret': nxt = None
                    break
            prev, label_id = label_id, nxt
        return {vid: self.mem[vid] for vid, (pt, sc) in m.vars.items() if sc == 3}

    def get(self, i):
        if i in self.vals: return self.vals[i]
        if i in self.m.consts:
            self.vals[i] = self.m.const_value(i)
            return self.vals[i]
        raise Fault(f'undefined id {i}')

    def put(self, i, v, taint=None):
        self.vals[i] = v
        if taint: self.vtaint[i] = taint

    def tn(self, *ids):
        for i in ids:
            t = self.vtaint.get(i)
            if t: return t
        return None

    def note_nonfinite(self, rid, v, what):
        flat = v if isinstance(v, list) else [v]
        if any(isinstance(x, float) and (x != x or x in (math.inf, -math.inf)) for x in flat):
            if rid not in self.nonfinite_seen:
                self.nonfinite_seen.add(rid)
                self.rep.event('nonfinite', self.label, f'%{rid} = {what} is {v}')
            return rid
        return None

    # ---- memory
    def ptr_type(self, vid):
        return self.m.types[self.m.vars[vid][0]][2]

    def walk(self, t, path):
        """For a buffer variable: (byte offset, final type, [(level type, index)]) of the path under the block type t."""
        m, off, levels = self.m, 0, []
        for ix in path:
            ty = m.types[t]
            levels.append((ty, ix))
            if ty[0] == 'struct':
                mo = m.mdeco_value(t, ix, 35)
                off += mo or 0
                t = ty[1][ix]
            elif ty[0] in ('arr', 'rarr'):
                stride = m.deco_value(t, 6) or m.size_of(ty[1])
                off += ix * stride
                t = ty[1]
            elif ty[0] == 'vec':
                off += ix * m.size_of(ty[1])
                t = ty[1]
            elif ty[0] == 'mat':
                off += ix * (m.mdeco_value(t, 0, 7) or 16)
                t = ty[1]
            else:
                raise Fault('index into a scalar')
        return off, t, levels

    def read_buf(self, vid, off, t):
        m = self.m
        ty = m.types[t]
        data = self.buffers.get(vid, b'')
        if ty[0] in ('float', 'int', 'bool'):
            size = max(m.size_of(t), 1)
            if off < 0 or off + size > len(data):
                return None
            raw = int.from_bytes(data[off:off + size], 'little')
            if ty[0] == 'float':
                return float_of_bits(raw) if ty[1] == 32 else struct.unpack('<d', data[off:off + 8])[0]
            if ty[0] == 'bool':
                return raw != 0
            return raw
        if ty[0] == 'vec':
            vals = [self.read_buf(vid, off + k * m.size_of(ty[1]), ty[1]) for k in range(ty[2])]
            return None if any(v is None for v in vals) else vals
        if ty[0] == 'arr':
            stride = m.deco_value(t, 6) or m.size_of(ty[1])
            vals = [self.read_buf(vid, off + k * stride, ty[1]) for k in range(m.const_int(ty[2]))]
            return None if any(v is None for v in vals) else vals
        if ty[0] == 'struct':
            vals = [self.read_buf(vid, off + (m.mdeco_value(t, k, 35) or 0), mt) for k, mt in enumerate(ty[1])]
            return None if any(v is None for v in vals) else vals
        raise Fault(f'cannot read a {ty[0]} from a buffer')

    def load(self, ptr, result_t, rid):
        vid, path = ptr
        sc = self.m.vars[vid][1]
        if sc in (2, 12):
            off, t, _ = self.walk(self.ptr_type(vid), path)
            v = self.read_buf(vid, off, t)
            if v is None:
                self.rep.event('oob-read', self.label, f'{self.name(vid)} byte {off} of {len(self.buffers.get(vid, b""))} captured: read as 0')
                return self.m.zero(result_t)
            return v
        v = self.mem[vid]
        for ix in path:
            v = v[ix]
        return v

    def store(self, ptr, value):
        vid, path = ptr
        if self.m.vars[vid][1] in (2, 12):
            self.rep.event('buffer-store', self.label, f'store into {self.name(vid)} ignored')
            return
        if not path:
            self.mem[vid] = value
            return
        v = self.mem[vid]
        for ix in path[:-1]:
            v = v[ix]
        v[path[-1]] = value

    def name(self, vid):
        return self.m.names.get(vid, f'%{vid}')

    def chain(self, rid, base_ptr, idx_ids, hdr):
        vid, path = base_ptr
        m = self.m
        sc = m.vars[vid][1]
        t = self.ptr_type(vid)
        for k in range(len(path)):
            ty = m.types[t]
            t = ty[1][path[k]] if ty[0] == 'struct' else ty[1]
        newpath = list(path)
        for idx_id in idx_ids:
            ty = m.types[t]
            raw = self.get(idx_id)
            itype = None
            ix = raw
            dyn = idx_id not in m.consts
            # indices are int values; read the signedness from the producing type when known
            if idx_id in m.consts:
                itype = m.consts[idx_id][0]
            else:
                itype = self.itype.get(idx_id)
            if itype is not None and m.types[itype][0] == 'int' and m.types[itype][2]:
                ix = self.to_signed(raw, m.types[itype][1])
            if ty[0] == 'struct':
                nxt = ty[1][ix]
                length = len(ty[1])
            elif ty[0] == 'arr':
                nxt, length = ty[1], m.const_int(ty[2])
            elif ty[0] == 'rarr':
                nxt, length = ty[1], None
            elif ty[0] in ('vec', 'mat'):
                nxt, length = ty[1], ty[2]
            else:
                raise Fault('index into a scalar')
            if dyn and sc in (2, 12):
                bad = ix < 0 or (length is not None and ix >= length)
                self.rep.index(hdr, self.name(vid), self.label, ix, length, ty[0], bad, self.tn(idx_id))
            elif dyn and (ix < 0 or (length is not None and ix >= length)):
                self.rep.event('local-index-out-of-range', self.label, f'{self.name(vid)} index {ix} of {length}')
            if length is not None and not (0 <= ix < length):
                if sc in (2, 12):
                    # keep going: the read will return 0 (robust access), after the byte range check
                    pass
                else:
                    ix = 0 if length else 0
            newpath.append(ix)
            t = nxt
        return (vid, tuple(newpath))

    # ---- one instruction
    def step(self, op, a):
        m = self.m
        if op == 248 or op == 247 or op == 246: return None
        if op == 249: return ('br', a[0])
        if op == 250: return ('br', a[1] if self.get(a[0]) else a[2])
        if op == 251:
            sel = self.get(a[0])
            for k in range(2, len(a), 2):
                if a[k] == sel: return ('br', a[k + 1])
            return ('br', a[1])
        if op in (252, 253, 255, 254): return ('ret', None)
        if op == 59: return None
        if op in (61, 62, 65, 66):
            return self.memop(op, a)
        if op == 12:
            return self.extinst(a)
        rt, rid = a[0], a[1]
        self.itype[rid] = rt
        ty = m.types[rt]
        v = self.alu(op, rt, a)
        taint = self.tn(*[x for x in a[2:] if x in self.vtaint])
        if op in (109, 110):
            src = self.get(a[2])
            flat = src if isinstance(src, list) else [src]
            if any(x != x or x in (math.inf, -math.inf) for x in flat):
                self.rep.event('nonfinite-convert', self.label, f'%{rid} converts non-finite {src}')
            elif any(abs(x) >= 2 ** 31 for x in flat):
                self.rep.event('convert-out-of-range', self.label, f'%{rid} converts {src}')
        nf = self.note_nonfinite(rid, v, m.names.get(rid, 'op%d' % op)) if m.types[self.sc(rt)][0] == 'float' else None
        self.put(rid, v, taint or nf)
        return None

    @property
    def itype(self):
        if not hasattr(self, '_itype'): self._itype = {}
        return self._itype

    def memop(self, op, a):
        m = self.m
        if op == 61:
            ptr = self.ptrs[a[2]]
            v = self.load(ptr, a[0], a[1])
            self.itype[a[1]] = a[0]
            self.put(a[1], v if not isinstance(v, list) else list(v), self.vtaint_mem.get(ptr[0]))
            return None
        if op == 62:
            ptr = self.ptrs[a[0]]
            val = self.get(a[1])
            self.store(ptr, list(val) if isinstance(val, list) else val)
            t = self.vtaint.get(a[1])
            if t: self.vtaint_mem[ptr[0]] = t
            return None
        # access chain
        base = a[2]
        if base in self.ptrs: bp = self.ptrs[base]
        else: bp = (base, ())
        self.ptrs[a[1]] = self.chain(a[1], bp, a[3:], a[1])
        return None

    @property
    def ptrs(self):
        if not hasattr(self, '_ptrs'):
            self._ptrs = {}
        return _PtrMap(self._ptrs, self.m)

    @property
    def vtaint_mem(self):
        if not hasattr(self, '_vtm'): self._vtm = {}
        return self._vtm

    def extinst(self, a):
        rt, rid, setid, inst = a[0], a[1], a[2], a[3]
        name = GLSL.get(inst)
        if name is None:
            raise Fault(f'GLSL.std.450 instruction {inst} is not modelled')
        args = [self.get(x) for x in a[4:]]
        bits = self.m.types[self.sc(rt)]
        f = {'Floor': math.floor, 'Ceil': math.ceil, 'Trunc': math.trunc, 'FAbs': abs, 'Sin': math.sin, 'Cos': math.cos,
             'Tan': math.tan}

        def one(*x):
            def sat(fn, *y):
                try:
                    return f32(fn(*y))
                except (ValueError, OverflowError):
                    return math.nan
            if name in ('Floor', 'Ceil', 'Trunc'):
                return x[0] if (x[0] != x[0] or x[0] in (math.inf, -math.inf)) else f32(f[name](x[0]))
            if name == 'Round' or name == 'RoundEven': return x[0] if x[0] != x[0] or abs(x[0]) == math.inf else f32(round(x[0]))
            if name == 'FAbs': return abs(x[0])
            if name == 'SAbs': return abs(self.to_signed(x[0], 32)) & M32
            if name == 'FSign': return 0.0 if x[0] == 0 or x[0] != x[0] else math.copysign(1.0, x[0])
            if name == 'Fract': return f32(x[0] - math.floor(x[0])) if abs(x[0]) != math.inf and x[0] == x[0] else math.nan
            if name in ('Sin', 'Cos', 'Tan'): return sat(f[name], x[0])
            if name == 'Exp2': return sat(lambda v: 2.0 ** v, x[0])
            if name == 'Exp': return sat(math.exp, x[0])
            if name == 'Log2': return math.nan if x[0] < 0 else (-math.inf if x[0] == 0 else sat(math.log2, x[0]))
            if name == 'Log': return math.nan if x[0] < 0 else (-math.inf if x[0] == 0 else sat(math.log, x[0]))
            if name == 'Sqrt': return math.nan if x[0] < 0 else sat(math.sqrt, x[0])
            if name == 'InverseSqrt': return math.nan if x[0] < 0 else (math.inf if x[0] == 0 else sat(lambda v: 1 / math.sqrt(v), x[0]))
            if name == 'Pow': return sat(math.pow, x[0], x[1])
            if name in ('FMin', 'NMin'): return x[1] if x[0] != x[0] else (x[0] if x[1] != x[1] else min(x))
            if name in ('FMax', 'NMax'): return x[1] if x[0] != x[0] else (x[0] if x[1] != x[1] else max(x))
            if name == 'FClamp' or name == 'NClamp': return min(max(x[0], x[1]), x[2])
            if name == 'SMax': return max(self.to_signed(x[0], 32), self.to_signed(x[1], 32)) & M32
            if name == 'SMin': return min(self.to_signed(x[0], 32), self.to_signed(x[1], 32)) & M32
            if name == 'UMax': return max(x[0], x[1])
            if name == 'UMin': return min(x[0], x[1])
            if name == 'SClamp': return min(max(self.to_signed(x[0], 32), self.to_signed(x[1], 32)), self.to_signed(x[2], 32)) & M32
            if name == 'UClamp': return min(max(x[0], x[1]), x[2])
            if name == 'Fma': return f32(x[0] * x[1] + x[2]) if all(abs(v) != math.inf and v == v for v in x) else f32(x[0] * x[1] + x[2])
            if name == 'FMix': return f32(x[0] * (1 - x[2]) + x[1] * x[2])
            if name == 'Step': return 1.0 if x[1] >= x[0] else 0.0
            raise Fault(f'GLSL {name}')
        ty = self.m.types[rt]
        v = self.lift(one, rt, *args) if ty[0] == 'vec' else one(*args)
        self.itype[rid] = rt
        taint = self.tn(*a[4:])
        nf = self.note_nonfinite(rid, v, name) if self.m.types[self.sc(rt)][0] == 'float' else None
        self.put(rid, v, taint or nf)
        return None

    def alu(self, op, rt, a):
        m = self.m
        ty = m.types[rt]
        sct = m.types[self.sc(rt)]
        bits = sct[1] if sct[0] in ('int', 'float') else 32
        mask = (1 << bits) - 1
        x = [self.get(i) for i in a[2:]]
        S = lambda v: self.to_signed(v, bits)

        def cmpf(fn, ordered=True):
            def g(p, q):
                if p != p or q != q:
                    return not ordered
                return fn(p, q)
            return g
        T = {
            126: lambda p: (-S(p)) & mask, 127: lambda p: -p,
            128: lambda p, q: (p + q) & mask, 129: lambda p, q: f32(p + q), 130: lambda p, q: (p - q) & mask,
            131: lambda p, q: f32(p - q), 132: lambda p, q: (p * q) & mask, 133: lambda p, q: f32(p * q),
            134: lambda p, q: (p // q) & mask if q else mask, 135: lambda p, q: int(S(p) / S(q)) & mask if S(q) else mask,
            136: fdiv,
            137: lambda p, q: p % q if q else 0,
            156: lambda p: p != p, 157: lambda p: abs(p) == math.inf,
            164: lambda p, q: p == q, 165: lambda p, q: p != q, 166: lambda p, q: p or q, 167: lambda p, q: p and q,
            168: lambda p: not p, 169: lambda c, p, q: p if c else q,
            170: lambda p, q: p == q, 171: lambda p, q: p != q, 172: lambda p, q: p > q, 173: lambda p, q: S(p) > S(q),
            174: lambda p, q: p >= q, 175: lambda p, q: S(p) >= S(q), 176: lambda p, q: p < q, 177: lambda p, q: S(p) < S(q),
            178: lambda p, q: p <= q, 179: lambda p, q: S(p) <= S(q),
            180: cmpf(lambda p, q: p == q), 181: cmpf(lambda p, q: p == q, False), 182: cmpf(lambda p, q: p != q),
            183: cmpf(lambda p, q: p != q, False), 184: cmpf(lambda p, q: p < q), 185: cmpf(lambda p, q: p < q, False),
            186: cmpf(lambda p, q: p > q), 187: cmpf(lambda p, q: p > q, False), 188: cmpf(lambda p, q: p <= q),
            189: cmpf(lambda p, q: p <= q, False), 190: cmpf(lambda p, q: p >= q), 191: cmpf(lambda p, q: p >= q, False),
            194: lambda p, q: (p >> (q & 31)) & mask, 195: lambda p, q: (S(p) >> (q & 31)) & mask,
            196: lambda p, q: (p << (q & 31)) & mask, 197: lambda p, q: p | q, 198: lambda p, q: p ^ q, 199: lambda p, q: p & q,
            200: lambda p: (~p) & mask, 83: lambda p: p,
        }
        if op == 169:   # select: the condition may be a scalar while the objects are vectors
            c = x[0]
            if ty[0] == 'vec':
                return [x[1][k] if (c[k] if isinstance(c, list) else c) else x[2][k] for k in range(ty[2])]
            return x[1] if c else x[2]
        if op == 80:
            out = []
            for v in x:
                out.extend(v if isinstance(v, list) else [v])
            return out
        if op == 81:
            v = x[0]
            for ix in a[3:]:
                v = v[ix]
            return v
        if op == 82:
            v = [list(e) if isinstance(e, list) else e for e in x[1]]
            t = v
            for ix in a[4:-1]:
                t = t[ix]
            t[a[-1]] = x[0]
            return v
        if op == 79:
            vecs = x[0] + x[1]
            return [vecs[k] if k != 0xffffffff else 0 for k in a[4:]]
        if op == 77:
            ix = x[1]
            return x[0][ix] if ix < len(x[0]) else 0
        if op == 124:
            src = x[0]
            return self.lift(lambda p: self.bitcast(p, a[2], rt), rt, src)
        if op in (109, 110, 111, 112):
            return self.lift(lambda p: self.convert(op, p, mask), rt, x[0])
        if op in (113, 114, 115):
            return self.lift(lambda p: p, rt, x[0])
        if op == 142:
            return [f32(p * x[1]) for p in x[0]]
        if op == 148:
            return f32(sum(p * q for p, q in zip(x[0], x[1])))
        if op not in T:
            raise Fault(f'op {op} is not modelled')
        return self.lift(T[op], rt, *x) if ty[0] == 'vec' else T[op](*x)

    def bitcast(self, p, src_id, rt):
        sty = self.m.types[self.itype.get(src_id, rt)] if src_id not in self.m.consts else self.m.types[self.m.consts[src_id][0]]
        dty = self.m.types[self.sc(rt)]
        if sty[0] == 'vec': sty = self.m.types[sty[1]]
        if dty[0] == 'float':
            return float_of_bits(p) if isinstance(p, int) else p
        if isinstance(p, float):
            return bits_of_float(p)
        return p & ((1 << dty[1]) - 1)

    def convert(self, op, p, mask):
        if op == 111:
            return f32(float(self.to_signed(p, 32)))
        if op == 112:
            return f32(float(p))
        if p != p:
            return 0
        if p in (math.inf, -math.inf):
            return (mask >> 1 if p > 0 else (1 << 31)) & mask if op == 110 else (mask if p > 0 else 0)
        t = int(p)
        if op == 110:
            t = max(-(1 << 31), min((1 << 31) - 1, t))
        else:
            t = max(0, min(mask, t))
        return t & mask


class _PtrMap:
    """ids -> pointers; a variable id is its own pointer."""
    def __init__(self, d, m): self.d, self.m = d, m
    def __contains__(self, k): return k in self.d or k in self.m.vars
    def __getitem__(self, k): return self.d[k] if k in self.d else (k, ())
    def __setitem__(self, k, v): self.d[k] = v


class Report:
    def __init__(self, verbose=False):
        self.lines, self.events, self.chains, self.verbose = [], {}, {}, verbose
        self.flagged = 0

    def event(self, kind, label, text):
        key = (kind, text.split(' ')[0] if kind.startswith('nonfinite') else text)
        n = self.events.setdefault((kind, text), [0, label])
        n[0] += 1
        if kind in ('oob-read', 'nonfinite-convert', 'convert-out-of-range', 'nonfinite') :
            self.flagged += 1 if kind != 'nonfinite' else 0

    def index(self, chain_id, var, label, ix, length, kind, bad, taint):
        c = self.chains.setdefault((chain_id, var, kind, length), {'n': 0, 'bad': 0, 'min': None, 'max': None, 'first_bad': None, 'taint': 0})
        c['n'] += 1
        c['min'] = ix if c['min'] is None else min(c['min'], ix)
        c['max'] = ix if c['max'] is None else max(c['max'], ix)
        if taint: c['taint'] += 1
        if bad:
            c['bad'] += 1
            self.flagged += 1
            if c['first_bad'] is None: c['first_bad'] = (label, ix)


def load_capture(d):
    d = Path(d)
    info = {'attr': [], 'stream': [], 'ubo': [], 'ssbo': [], 'index': None, 'top': {}}
    for line in (d / 'draw.txt').read_text().splitlines():
        if '=' not in line: continue
        if line.split(' ')[0] in ('attr', 'stream', 'ubo', 'ssbo', 'index'):
            kind, rest = line.split(' ', 1)
            rec = dict(re.findall(r'(\w+)=(\S+)', rest))
            if kind == 'index': info['index'] = rec
            else: info[kind].append(rec)
        else:
            k, v = line.split('=', 1)
            info['top'][k] = v
    return info


def evaluate(d, max_vertices=None, max_instances=None, verbose=False, out=None, spv=None, ubo_patch=None, probe=None):
    """spv: another module to run instead of <d>/vertex.spv; ubo_patch: {cbuf slot: [(byte offset, bytes)]} applied to the captured
    blocks; probe: SPIR-V result ids whose final values are recorded per invocation in rep.probes."""
    d = Path(d)
    info = load_capture(d)
    top = info['top']
    spv_bytes = Path(spv).read_bytes() if spv else (d / 'vertex.spv').read_bytes()
    words = struct.unpack('<%dI' % (len(spv_bytes) // 4), spv_bytes)
    mod = Module(words)
    rep = Report(verbose)
    # buffers by variable: Binding decoration -> resource
    buffers, bound_notes = {}, []
    for vid, (pt, sc) in mod.vars.items():
        if sc not in (2, 12):
            continue
        b = mod.deco_value(vid, 33)
        if b == 0 and (d / 'support.bin').exists():
            buffers[vid] = (d / 'support.bin').read_bytes()
            continue
        for res in info['ubo'] + info['ssbo']:
            if int(res['binding']) == b and res.get('readable') == '1':
                buffers[vid] = bytearray((d / res['file']).read_bytes())
                for off, blob in (ubo_patch or {}).get(int(res['cbuf_slot']) if 'cbuf_slot' in res else -1, []):
                    buffers[vid][off:off + len(blob)] = blob
                buffers[vid] = bytes(buffers[vid])
                bound_notes.append((mod.names.get(vid, vid), len(buffers[vid]), int(res['bound_size'])))
        if vid not in buffers:
            rep.event('no-data', '', f'binding {b} ({mod.names.get(vid, vid)}) has no readable capture: reads as 0')
    count = int(top['count'])
    first = int(top['first'])
    instances = int(top['instances'])
    base_instance = int(top['base_instance'])
    base_vertex = int(top.get('base_vertex', 0))
    kind = top['kind']
    nv = min(count, max_vertices or 4)
    ni = min(instances, max_instances or instances)
    streams = {int(s['index']): s for s in info['stream']}
    stream_data = {b: (d / f'stream-{b}.bin').read_bytes() if (d / f'stream-{b}.bin').exists() else b'' for b in streams}
    index_data = (d / 'index.bin').read_bytes() if info['index'] and (d / 'index.bin').exists() else b''
    itype = int(top.get('itype', -1))
    attr_in = {}   # input variable id by Location
    builtin_in = {}
    for vid, (pt, sc) in mod.vars.items():
        if sc == 1:
            loc = mod.deco_value(vid, 30)
            bi = mod.deco_value(vid, 11)
            if bi is not None: builtin_in[vid] = bi
            elif loc is not None: attr_in[loc] = vid
    machine = Machine(mod, buffers, rep)
    rep.probes = []
    machine.probe_ids = list(probe or [])
    results = []
    for inst in range(ni):
        for v in range(nv):
            if kind == 'indexed' and index_data:
                w = 1 << itype
                vertex_index = (int.from_bytes(index_data[v * w:(v + 1) * w], 'little') + base_vertex) & M32
            else:
                vertex_index = first + v
            inputs = {}
            for a in info['attr']:
                loc = int(a['location'])
                if loc not in attr_in: continue
                s = streams.get(int(a['stream']))
                if s is None: continue
                stride, div = int(s['stride']), int(s['divisor'])
                unit = base_instance + inst // div if div else vertex_index
                off = unit * stride + int(a['offset'])
                data = stream_data[int(a['stream'])]
                raw = data[off:off + 16] if off + 1 <= len(data) else b''
                if off >= len(data):
                    rep.event('stream-oob', f'i{inst} v{v}', f'location {loc} reads stream {a["stream"]} byte {off} beyond the {len(data)} captured')
                vec = decode_attr(int(a['nvn_format']), raw, 4)
                vt = mod.vars[attr_in[loc]][0]
                ty = mod.types[mod.types[vt][2]]
                if ty[0] == 'vec':
                    et = mod.types[ty[1]]
                    vals = vec[:ty[2]]
                    if et[0] == 'int':
                        vals = [int(x) & M32 for x in vals]
                    else:
                        vals = [f32(float(x)) for x in vals]
                    inputs[attr_in[loc]] = vals
                else:
                    et = ty
                    inputs[attr_in[loc]] = f32(float(vec[0])) if et[0] == 'float' else int(vec[0]) & M32
            built = {}
            for vid, bi in builtin_in.items():
                built[vid] = {42: vertex_index, 5: vertex_index, 43: base_instance + inst, 6: base_instance + inst,
                              4424: base_vertex, 4425: base_instance, 4426: 0}.get(bi, 0)
            label = f'instance {inst} vertex {v}'
            machine.label = label
            machine._ptrs, machine._vtm, machine._itype = {}, {}, {}
            try:
                outs = machine.run(inputs, built, label)
            except Fault as e:
                rep.event('fault', label, str(e))
                continue
            results.append((label, outs))
            rep.probes.append((label, {i: (machine.mem.get(i) if i in mod.vars else machine.vals.get(i)) for i in machine.probe_ids}))
    return mod, rep, results, bound_notes


def format_report(mod, rep, results, bound_notes, top):
    L = [f'origin={top.get("origin")} draw: {top.get("kind")} count={top.get("count")} instances={top.get("instances")} '
         f'base_instance={top.get("base_instance")} first={top.get("first")}']
    for name, got, bound in bound_notes:
        L.append(f'buffer {name}: {got} bytes captured of {bound} bound')
    L.append('dynamic uniform / storage indices (access chain id, buffer, level, declared length): uses, index range, out of range')
    for (cid, var, kind, length), c in sorted(rep.chains.items()):
        flag = '  <-- OUT OF RANGE' if c['bad'] else ''
        extra = f', first at {c["first_bad"][0]} index {c["first_bad"][1]}' if c['bad'] else ''
        tt = f', {c["taint"]} from a non-finite value' if c['taint'] else ''
        L.append(f'  %{cid} {var} {kind}[{length}]: {c["n"]} uses, indices {c["min"]}..{c["max"]}, {c["bad"]} out of range{extra}{tt}{flag}')
    if not rep.chains:
        L.append('  (none)')
    L.append('events:')
    for (kind, text), (n, label) in sorted(rep.events.items()):
        L.append(f'  {kind}: {text} (x{n}, first at {label})')
    if not rep.events:
        L.append('  (none)')
    L.append('outputs:')
    for label, outs in results:
        parts = []
        for vid, v in sorted(outs.items()):
            nm = mod.names.get(vid) or f'loc{mod.deco_value(vid, 30)}'
            parts.append(f'{nm}={_fmt(v)}')
        L.append(f'  {label}: ' + '  '.join(parts))
    L.append(f'flagged: {rep.flagged}')
    return '\n'.join(L) + '\n'


def _fmt(v):
    if isinstance(v, list):
        return '[' + ', '.join(_fmt(x) for x in v) + ']'
    if isinstance(v, float):
        return f'{v:.6g}'
    return str(v)


def convert_quad_capture(src, dst):
    """Rewrites a build/quad-replay-captures/ground-* directory (the first capture, quad-capture-v1) as a lq-vs-capture-v1 directory
    this evaluator reads, for the draw it holds: 4 vertices, 3 instances, nine instance-rate vec4 streams, VS blocks c8 / c9 / c10."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'vertex.spv').write_bytes((src / 'vertex.spv').read_bytes())
    sup = bytearray(1408)
    w, h = 2880.0, 1620.0
    sup[144:160] = struct.pack('<4f', 2 / w, 2 / h, 0, 0)
    sup[160:176] = struct.pack('<4f', w, h, 0, 0)
    sup[1392:1396] = struct.pack('<f', 1.0)
    for i in range(73):
        sup[192 + 16 * i:192 + 16 * i + 4] = struct.pack('<f', 1.0)
    (dst / 'support.bin').write_bytes(bytes(sup))
    lines = ['format=lq-vs-capture-v1', 'origin=ground-20261108-01', 'cache_key=converted', 'skipped=1', 'prim=5', 'kind=direct',
             'count=4', 'first=0', 'instances=3', 'base_instance=0', 'base_vertex=0', 'itype=-1', 'support_bytes=1408']
    attr_stream = {1: 6, 2: 7, 3: 8, 4: 5, 5: 2, 6: 0, 7: 1, 8: 4, 9: 3}
    for loc, b in attr_stream.items():
        lines.append(f'attr location={loc} nvn_format=46 offset=0 stream={b} type=0 comps=4')
    for b in range(9):
        data = (src / f'vertex-{b:02d}.bin').read_bytes()
        (dst / f'stream-{b}.bin').write_bytes(data)
        lines.append(f'stream index={b} stride=16 divisor=1 address=0 bound_size={len(data)} captured={len(data)} readable=1 file=stream-{b}.bin')
    for binding, slot in ((1, 8), (2, 10), (3, 9)):
        data = (src / f'ubo-0-{slot - 3:02d}.bin').read_bytes()
        (dst / f'ubo-{slot}.bin').write_bytes(data)
        lines.append(f'ubo binding={binding} cbuf_slot={slot} address=0 bound_size={len(data)} captured={len(data)} readable=1 file=ubo-{slot}.bin')
    (dst / 'draw.txt').write_text(chr(10).join(lines) + chr(10), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture', type=Path)
    ap.add_argument('--from-quad', type=Path, help='convert this build/quad-replay-captures/ground-* directory into <capture> first')
    ap.add_argument('--vertices', type=int)
    ap.add_argument('--instances', type=int)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    if a.from_quad:
        convert_quad_capture(a.from_quad, a.capture)
    mod, rep, results, notes = evaluate(a.capture, a.vertices, a.instances, a.verbose)
    text = format_report(mod, rep, results, notes, load_capture(a.capture)['top'])
    (a.out or a.capture / 'vseval-report.txt').write_text(text, encoding='utf-8')
    sys.stdout.write(text)
    return 1 if rep.flagged else 0


if __name__ == '__main__':
    raise SystemExit(main())
