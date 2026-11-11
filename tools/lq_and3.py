"""Offline: the OpBitwiseAnd-with-constant patterns of the family vertex modules, backwards (where the other operand comes
from) and forwards (what its result feeds), and the mapping of NVN 2 stage origins to cache modules.

    python tools/lq_and3.py --cache build/dkb100-lqpatch/shader-cache --out build/quad-replay/analysis [--const 3]
        [--map build/quad-replay/analysis/origins/map/origin-map.json --group bad=67a2...,... --group good=2e97...,...]

For every family vertex module (a Vertex entry point and an Input at Location 9) it reads the spirv-dis text and, for each
OpBitwiseAnd with the integer constant --const, writes the pattern: the expression that makes the other operand (leaves: the
uniform row and component, builtin inputs, input attributes, constants; local variables are followed through their stores)
and the terminal consumers of the result (followed through bitcasts, integer ops, compares, selects and local variables):
a dynamic index into a uniform or local, a switch, a branch condition, a select, a conversion to float, ... Counts modules and
ANDs per pattern into <out>/and3-patterns.txt.

--map FILE is the origin-map.json tools/lq_origin_map.py writes ({origin: [cache key, ...]}); each origin is listed with its
cache modules and their patterns, and --group NAME=origin,origin,... prints the pattern counts per group (origins without
a stage dump are reported as unmapped). CPU only.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess

from lq_family_scan import is_family_vertex, words

DIS = 'C:/msys64/ucrt64/bin/spirv-dis.exe'
PASS = {'OpBitcast', 'OpSelect', 'OpIAdd', 'OpISub', 'OpIMul', 'OpShiftLeftLogical', 'OpShiftRightLogical',
        'OpShiftRightArithmetic', 'OpBitwiseAnd', 'OpBitwiseOr', 'OpBitwiseXor', 'OpSNegate', 'OpNot', 'OpCopyObject',
        'OpPhi', 'OpSMax', 'OpSMin', 'OpUMax', 'OpUMin', 'OpLogicalNot', 'OpLogicalOr', 'OpLogicalAnd',
        'OpIEqual', 'OpINotEqual', 'OpSLessThan', 'OpSGreaterThan', 'OpSLessThanEqual', 'OpSGreaterThanEqual',
        'OpULessThan', 'OpUGreaterThan', 'OpULessThanEqual', 'OpUGreaterThanEqual', 'OpFOrdEqual', 'OpFOrdNotEqual'}
BUILTIN = {0: 'Position', 5: 'VertexId', 6: 'InstanceId', 42: 'VertexIndex', 43: 'InstanceIndex', 4424: 'BaseVertex',
           4425: 'BaseInstance', 4426: 'DrawIndex'}


class Prog:
    def __init__(self, text):
        self.ins = {}          # result id -> (op, [operand tokens])
        self.uses = defaultdict(list)   # id -> [(consumer result id or None, op, position, operands)]
        self.stores = defaultdict(list)
        self.loads = defaultdict(list)
        self.names, self.loc, self.builtin = {}, {}, {}
        self.consts = {}
        self.ints = set()
        self.var = {}          # id -> storage class
        self.ands = []
        inside = False
        for line in text.splitlines():
            s = line.strip()
            m = re.match(r'(?:(%\d+) = )?(Op\w+)\s*(.*)', s)
            if not m:
                continue
            res, op, rest = m.groups()
            ops = rest.split()
            if op == 'OpName':
                mm = re.match(r'(%\d+) "(.*)"', rest)
                if mm: self.names[mm.group(1)] = mm.group(2)
                continue
            if op == 'OpDecorate':
                if len(ops) >= 3 and ops[1] == 'Location': self.loc[ops[0]] = int(ops[2])
                if len(ops) >= 3 and ops[1] == 'BuiltIn': self.builtin[ops[0]] = ops[2]
                continue
            if op == 'OpTypeInt':
                self.ints.add(res)
            if op == 'OpConstant' and ops and ops[0] in self.ints:
                self.consts[res] = int(ops[1]) & 0xffffffff
            if op == 'OpFunction':
                inside = True
            if op == 'OpFunctionEnd':
                inside = False
            if op == 'OpVariable' and len(ops) >= 2:
                self.var[res] = ops[1]
            if res:
                self.ins[res] = (op, ops)
            ids = [t for t in ops if t.startswith('%')]
            if op == 'OpStore' and len(ids) >= 2:
                self.stores[ids[0]].append(ids[1])
            if op == 'OpLoad' and len(ids) >= 2:
                self.loads[ids[1]].append(res)
            for pos, t in enumerate(ids):
                self.uses[t].append((res, op, pos, ids))
            if op == 'OpBitwiseAnd' and inside and len(ids) >= 3:
                self.ands.append(res)

    def uname(self, v):
        return self.names.get(v, v)

    def expr(self, i, depth=0, seen=()):
        if i in self.consts:
            return f'c{self.consts[i]:#x}'
        d = self.ins.get(i)
        if depth > 14 or not d or i in seen:
            return '?'
        op, ops = d
        ids = [t for t in ops if t.startswith('%')]
        seen = seen + (i,)
        if op == 'OpBitcast':
            return 'bc(' + self.expr(ids[1], depth + 1, seen) + ')'
        if op == 'OpLoad':
            return self.ptr(ids[1], depth + 1, seen)
        if op == 'OpSelect':
            return 'sel(' + self.expr(ids[1], depth + 1, seen) + ',' + self.expr(ids[2], depth + 1, seen) + ',' + self.expr(ids[3], depth + 1, seen) + ')'
        if op in ('OpIAdd', 'OpISub', 'OpIMul', 'OpBitwiseAnd', 'OpBitwiseOr', 'OpBitwiseXor', 'OpShiftLeftLogical',
                  'OpShiftRightLogical', 'OpShiftRightArithmetic', 'OpFMul', 'OpFAdd', 'OpFSub', 'OpFDiv'):
            return op[2:] + '(' + ','.join(self.expr(x, depth + 1, seen) for x in ids[1:3]) + ')'
        if op in ('OpIEqual', 'OpINotEqual', 'OpSLessThan', 'OpSGreaterThan', 'OpFOrdGreaterThan', 'OpFOrdLessThan',
                  'OpFOrdGreaterThanEqual', 'OpLogicalNot', 'OpLogicalOr', 'OpIsNan', 'OpFOrdNotEqual'):
            return op[2:] + '(' + ','.join(self.expr(x, depth + 1, seen) for x in ids[1:3]) + ')'
        if op in ('OpConvertSToF', 'OpConvertFToS', 'OpConvertUToF', 'OpConvertFToU', 'OpSNegate', 'OpNot', 'OpCopyObject'):
            return op[2:] + '(' + self.expr(ids[1], depth + 1, seen) + ')'
        if op == 'OpExtInst' and len(ids) >= 3:
            return 'ext' + ops[2] + '(' + ','.join(self.expr(x, depth + 1, seen) for x in ids[2:4]) + ')'
        return op[2:]

    def ptr(self, p, depth, seen):
        d = self.ins.get(p)
        if not d or depth > 7:
            return '?'
        op, ops = d
        ids = [t for t in ops if t.startswith('%')]
        if op in ('OpAccessChain', 'OpInBoundsAccessChain'):
            base = ids[1]
            idx = [self.consts.get(x) for x in ids[2:]]
            sc = self.var.get(base)
            if sc == 'Uniform':
                if len(idx) >= 3 and None not in idx[1:3]:
                    return f'U({self.uname(base)}[{idx[1]}].{"xyzw"[idx[2] & 3]})'
                return f'U({self.uname(base)}[dyn])'
            if sc == 'Input':
                return self.ptr(base, depth + 1, seen) + ('[' + ','.join(str(x) if x is not None else 'dyn' for x in idx[1:]) + ']' if len(idx) > 1 else '')
            return self.ptr(base, depth + 1, seen)
        if op == 'OpVariable':
            sc = ops[1]
            if sc == 'Input':
                b = self.builtin.get(p)
                return b if b else f'IN{self.loc.get(p, "?")}'
            if sc == 'Uniform':
                return f'U({self.uname(p)})'
            if sc == 'Function':
                st = self.stores.get(p, [])
                srcs = sorted({self.expr(s, depth + 1, seen) for s in st})
                if not srcs:
                    return 'local(unset)'
                return srcs[0] if len(srcs) == 1 else 'local{' + '|'.join(srcs[:3]) + '}'
        return '?'

    def consumers(self, i):
        """Terminal consumer kinds of a value: through integer ops, compares, selects, locals."""
        kinds, seen, work = set(), set(), [i]
        while work and len(seen) < 600:
            v = work.pop()
            if v in seen:
                continue
            seen.add(v)
            for res, op, pos, ids in self.uses.get(v, []):
                if op in ('OpAccessChain', 'OpInBoundsAccessChain') and pos >= 2:
                    base = ids[1]
                    sc = self.var.get(base)
                    kinds.add('index:' + (f'U({self.uname(base)})' if sc == 'Uniform' else sc if sc else 'local'))
                elif op == 'OpSwitch' and pos == 0:
                    kinds.add('switch')
                elif op == 'OpBranchConditional' and pos == 0:
                    kinds.add('branch')
                elif (op == 'OpVectorExtractDynamic' and pos == 2) or (op == 'OpVectorInsertDynamic' and pos == 3):
                    kinds.add('dynamic-vector')
                elif op == 'OpStore' and pos == 1:
                    for ld in self.loads.get(ids[0], []):
                        work.append(ld)
                    if self.var.get(ids[0]) == 'Output':
                        kinds.add('output')
                elif op in ('OpConvertSToF', 'OpConvertUToF'):
                    kinds.add('toFloat')
                    if res: work.append(res)
                elif op == 'OpSelect' and pos == 1:
                    kinds.add('select-cond')
                    if res: work.append(res)
                elif op in PASS and res:
                    if op.startswith('OpShift') and pos == 2:
                        kinds.add('shift-amount')
                    work.append(res)
                else:
                    kinds.add('other:' + op[2:])
        return kinds


def dynamic_uniform(p):
    """Counts of uniform access chains with a computed row / component index, per buffer name: {buffer: [row, comp]}."""
    out = defaultdict(lambda: [0, 0])
    for res, (op, ops) in p.ins.items():
        if op not in ('OpAccessChain', 'OpInBoundsAccessChain'):
            continue
        ids = [t for t in ops if t.startswith('%')]
        base = ids[1]
        if p.var.get(base) != 'Uniform':
            continue
        idx = ids[2:]
        if len(idx) >= 2 and idx[1] not in p.consts:
            out[p.uname(base)][0] += 1
        if len(idx) >= 3 and idx[2] not in p.consts:
            out[p.uname(base)][1] += 1
    return out


COMPARE = {'OpIEqual', 'OpINotEqual', 'OpUGreaterThan', 'OpSGreaterThan', 'OpUGreaterThanEqual', 'OpSGreaterThanEqual',
           'OpULessThan', 'OpSLessThan', 'OpULessThanEqual', 'OpSLessThanEqual'}


def and3_classes(p):
    """Per OpBitwiseAnd with the constant 3: round ((x>>31)&3), else comp (the last index of a uniform access chain, through
    bitcasts), else test (feeds an integer compare, OpSelect or OpSwitch), else none. Returns a Counter of the classes."""
    out = Counter()
    for r in p.ands:
        ids = [t for t in p.ins[r][1] if t.startswith('%')]
        x, y = ids[1], ids[2]
        if p.consts.get(y) == 3: other = x
        elif p.consts.get(x) == 3: other = y
        else: continue
        d = p.ins.get(other)
        sh = [t for t in d[1] if t.startswith('%')] if d else []
        if d and d[0] == 'OpShiftRightArithmetic' and len(sh) >= 3 and p.consts.get(sh[2]) == 31:
            out['round'] += 1
            continue
        comp = test = False
        work, seen = [r], set()
        while work:
            v = work.pop()
            if v in seen: continue
            seen.add(v)
            for res, op, pos, uids in p.uses.get(v, []):
                if op in ('OpAccessChain', 'OpInBoundsAccessChain') and pos == len(uids) - 1 and p.var.get(uids[1]) in ('Uniform', 'StorageBuffer'):
                    comp = True
                elif op in ('OpBitcast', 'OpCopyObject') and res:
                    work.append(res)
                elif op in COMPARE or op == 'OpSelect' or (op == 'OpSwitch' and pos == 0):
                    test = True
        out['comp' if comp else 'test' if test else 'none'] += 1
    return out


def body_hash(w):
    """Hash of the opcode sequence of the function bodies."""
    h = hashlib.sha1()
    i, inside = 5, False
    while i < len(w):
        n, op = w[i] >> 16, w[i] & 0xffff
        if op == 54: inside = True
        if inside: h.update(struct_pack(op))
        if op == 56: inside = False
        i += n
    return h.hexdigest()


def struct_pack(op):
    return op.to_bytes(2, 'little')


def analyze(path, const):
    text = subprocess.run([DIS, '--raw-id', '--no-header', str(path)], capture_output=True, text=True).stdout
    p = Prog(text)
    pats = Counter()
    pats.dyn = dynamic_uniform(p)
    for r in p.ands:
        ops = p.ins[r][1]
        ids = [t for t in ops if t.startswith('%')]
        x, y = ids[1], ids[2]
        if p.consts.get(y) == const: other = x
        elif p.consts.get(x) == const: other = y
        else: continue
        pats[(p.expr(other), ','.join(sorted(p.consumers(r))))] += 1
    return pats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--const', type=lambda s: int(s, 0), default=3)
    ap.add_argument('--map', type=Path)
    ap.add_argument('--classes', action='store_true', help='count the &3 classes (comp, round, test) over the family modules and exit')
    ap.add_argument('--group', action='append', default=[])
    a = ap.parse_args()
    files = sorted(a.cache.glob('*.spv'))
    fam = {}
    hashes = defaultdict(list)
    for f in files:
        w = words(f)
        hashes[body_hash(w)].append(f)
        if is_family_vertex(w):
            fam[f] = True
    famfiles = sorted(fam)
    if a.classes:
        tot, mods = Counter(), Counter()
        def cls(f):
            return and3_classes(Prog(subprocess.run([DIS, '--raw-id', '--no-header', str(f)], capture_output=True, text=True).stdout))
        with ThreadPoolExecutor(max_workers=6) as ex:
            for c in ex.map(cls, famfiles):
                tot.update(c)
                mods.update(k for k in c)
        text = (f'{len(famfiles)} family vertex modules; &3 ANDs by class: '
                + ', '.join(f'{k} {tot[k]} ({mods[k]} modules)' for k in ('comp', 'round', 'test', 'none')) + chr(10))
        (a.out / 'and3-classes.txt').write_text(text, encoding='utf-8')
        print(text)
        return
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = dict(zip(famfiles, ex.map(lambda f: analyze(f, a.const), famfiles)))
    dyns = {f: r.dyn for f, r in results.items()}
    per_pat_mod, per_pat_n, example = defaultdict(set), Counter(), {}
    for f, pats in results.items():
        for k, n in pats.items():
            per_pat_mod[k].add(f.stem)
            per_pat_n[k] += n
            example.setdefault(k, f.stem[:12])
    nmod = sum(1 for f in famfiles if results[f])
    lines = [f'{len(famfiles)} family vertex modules; {nmod} have OpBitwiseAnd with constant {a.const:#x}; '
             f'{len(per_pat_n)} distinct patterns, {sum(per_pat_n.values())} ANDs',
             'modules   count  pattern (other operand)  =>  terminal consumers of the result  [example module]']
    for k, n in per_pat_n.most_common():
        lines.append(f'{len(per_pat_mod[k]):6d} {n:7d}  {k[0]}  =>  {k[1]}  [{example[k]}]')
    with_dyn = Counter()
    for f, d in dyns.items():
        for b, (r, c) in d.items():
            with_dyn[(b, 'row' if r else '', 'comp' if c else '')] += 1
    lines += ['', 'family modules with computed uniform indices (buffer, computed row, computed component): modules']
    lines += [f'  {k[0]:8s} {k[1] or "-":4s} {k[2] or "-":4s} {n}' for k, n in sorted(with_dyn.items())]
    lines.append(f'  {sum(1 for d in dyns.values() if not d)} family modules have no computed uniform index at all')
    if a.map:
        omap = json.loads(a.map.read_text())
        lines += ['', 'origins (NVN 2 stage -> cache modules it translated to):']
        origin_pats = {}
        for origin, keys in sorted(omap.items()):
            mods = [a.cache / (k + '.spv') for k in keys]
            fams = [m for m in mods if m in fam]
            pats = Counter()
            lines.append(f'{origin}: cache module(s) {", ".join(k[:12] for k in keys)}; '
                         + ('in the family (Location 9 input)' if fams else 'NOT a family vertex module: never patched'))
            for m in mods:
                r = results[m] if m in results else analyze(m, a.const)
                d = r.dyn
                lines.append(f'      {m.stem[:12]} computed uniform indices: '
                             + (', '.join(f'{b} row {x} comp {y}' for b, (x, y) in sorted(d.items())) or 'none')
                             + f'; {sum(r.values())} ANDs with constant {a.const:#x}')
                pats.update(r)
            origin_pats[origin] = (fams, pats)
            for k, n in pats.most_common():
                lines.append(f'      {n:3d}  {k[0]}  =>  {k[1]}')
        for g in a.group:
            name, ids = g.split('=')
            ids = ids.split(',')
            tot = Counter()
            members = [o for o in ids if o in origin_pats]
            for o in members:
                tot.update(origin_pats[o][1])
            lines += ['', f'group {name}: {len(members)} of {len(ids)} origins mapped ({", ".join(o for o in ids if o not in origin_pats) or "none"} unmapped); '
                          f'{sum(tot.values())} ANDs with constant {a.const:#x}']
            for k, n in tot.most_common():
                lines.append(f'      {n:3d}  {k[0]}  =>  {k[1]}')
    text = '\n'.join(lines) + '\n'
    (a.out / 'and3-patterns.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
