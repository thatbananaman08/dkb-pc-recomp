"""Offline: the OpBitwiseAnd constants of the family vertex modules (a Vertex entry point and an Input at Location 9) in a
shader cache, and where the other operand comes from.

    python tools/lq_and_histogram.py --cache build/dkb100-lqpatch/shader-cache --out build/quad-replay/analysis [--top 15]

For every OpBitwiseAnd with an integer OpConstant operand it counts the constant, the modules using it and the total, and
traces the other operand back through bitcasts, loads, access chains and function-local variables (their stores) to its
source: a uniform row and component (bitcast), a builtin input (VertexIndex, InstanceIndex, ...), an input attribute, or
other. ANDs without a constant operand are listed as "nonconst". Writes <out>/and-histogram.txt. CPU only.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import struct

from lq_family_scan import is_family_vertex, words

BUILTINS = {0: 'Position', 1: 'PointSize', 5: 'VertexId', 6: 'InstanceId', 42: 'VertexIndex', 43: 'InstanceIndex',
            4416: 'SubgroupEqMask', 4424: 'BaseVertex', 4425: 'BaseInstance', 4426: 'DrawIndex'}


class Mod:
    def __init__(self, w):
        self.defs = {}                    # result id -> (opcode, operands)
        self.names, self.loc, self.builtin, self.binding = {}, {}, {}, {}
        self.stores = defaultdict(list)   # pointer id -> stored value ids
        self.ints = set()
        self.var = {}                     # variable id -> storage class
        self.ands = []
        i = 5
        while i < len(w):
            n, op = w[i] >> 16, w[i] & 0xffff
            a = w[i + 1:i + n]
            if op == 5 and len(a) >= 2:
                raw = struct.pack('<%dI' % (len(a) - 1), *a[1:])
                self.names[a[0]] = raw.split(b'\0')[0].decode('latin1')
            elif op == 71 and len(a) >= 3:
                if a[1] == 30: self.loc[a[0]] = a[2]
                elif a[1] == 11: self.builtin[a[0]] = a[2]
                elif a[1] == 33: self.binding[a[0]] = a[2]
            elif op == 21 and a:
                self.ints.add(a[0])
            elif op == 59 and len(a) >= 3:
                self.var[a[1]] = a[2]
                self.defs[a[1]] = (op, a)
            elif op == 62 and len(a) >= 2:
                self.stores[a[0]].append(a[1])
            elif op == 199 and len(a) >= 4:
                self.ands.append(a)
            if op not in (62,) and len(a) >= 2 and op in (43, 61, 65, 66, 124, 199, 197, 198, 169, 133, 129, 131, 83, 222, 245, 12):
                self.defs[a[1]] = (op, a)
            i += n

    def const(self, i):
        d = self.defs.get(i)
        return d[1][2] if d and d[0] == 43 and d[1][0] in self.ints else None

    def uniform_name(self, v):
        return self.names.get(v) or f'binding{self.binding.get(v, "?")}'

    def source(self, i, depth=0):
        d = self.defs.get(i)
        if depth > 12 or not d:
            return 'other'
        op, a = d
        if op == 124:                     # OpBitcast
            return 'bitcast ' + self.source(a[2], depth + 1)
        if op == 61:                      # OpLoad
            return self.ptr(a[2], depth + 1)
        if op == 43:
            return 'constant'
        return 'other op%d' % op

    def ptr(self, p, depth):
        d = self.defs.get(p)
        if depth > 12 or not d:
            return 'other'
        op, a = d
        if op == 65 or op == 66:          # access chain: base, indices
            base = a[2]
            idx = [self.const(x) for x in a[3:]]
            if self.var.get(base) == 2:   # Uniform
                if len(idx) >= 3 and None not in idx[1:3]:
                    return f'{self.uniform_name(base)} row {idx[1]} .{"xyzw"[idx[2] & 3]}'
                return f'{self.uniform_name(base)} computed index'
            return self.ptr(base, depth + 1)
        if op == 59:
            sc = a[2]
            if sc == 1:
                if p in self.builtin:
                    return 'builtin ' + BUILTINS.get(self.builtin[p], str(self.builtin[p]))
                return f'input L{self.loc.get(p, "?")}'
            if sc == 2:
                return self.uniform_name(p)
            if sc == 7:                   # function-local: what is stored into it
                st = self.stores.get(p, [])
                if not st:
                    return 'local (never stored)'
                srcs = sorted({self.source(s, depth + 1) for s in st})
                return srcs[0] if len(srcs) == 1 else 'local, ' + ' | '.join(srcs[:3])
        return 'other'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--top', type=int, default=15)
    a = ap.parse_args()
    total, modules = Counter(), defaultdict(set)
    src = defaultdict(Counter)
    nfam = 0
    for f in sorted(a.cache.glob('*.spv')):
        w = words(f)
        if not is_family_vertex(w):
            continue
        nfam += 1
        m = Mod(w)
        for op in m.ands:
            x, y = op[2], op[3]
            cx, cy = m.const(x), m.const(y)
            if cx is None and cy is None:
                key, other = 'nonconst', m.source(x) + ' & ' + m.source(y)
            else:
                key, other = (cy, x) if cy is not None else (cx, y)
                other = m.source(other)
            total[key] += 1
            modules[key].add(f.stem)
            src[key][other] += 1
    lines = [f'{nfam} family vertex modules; {sum(total.values())} OpBitwiseAnd, {len(total)} distinct constants (or nonconst)',
             'constant       modules    total   other operand (top sources)']
    for key, n in total.most_common(a.top):
        name = key if isinstance(key, str) else f'{key:#010x}'
        top = '; '.join(f'{c}x {s}' for s, c in src[key].most_common(3))
        lines.append(f'{name:12s} {len(modules[key]):8d} {n:8d}   {top}')
    rest = total.most_common()[a.top:]
    if rest:
        lines.append(f'(and {len(rest)} more constants, {sum(n for _, n in rest)} ANDs)')
    text = '\n'.join(lines) + '\n'
    (a.out / 'and-histogram.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
