"""Offline: what the `test`-class &3 ANDs (DKB_LQ_AND3_CLASS=test) decide in the family vertex modules.

    python tools/lq_and3_detail.py --modules <dir of local-single-block-optimized .spv> --out build/quad-replay/analysis

For every OpBitwiseAnd with the constant 3 whose result feeds an integer compare, an OpSelect or an OpSwitch (the class
tools/lq_and3.py --classes counts), it records the compare (operator and the other operand when it is a constant), what the
compare result feeds (a select: the two chosen values; a branch: the arms, with instruction, load, dynamically indexed
uniform access, nested branch and ExtInst counts), and what nulling does: the AND becomes 0, so the compare takes a fixed
value and one side is forced. It prints a table of the distinct shapes with module counts. CPU only. The modules are
the local-single-block-optimized ones the runtime's lq_patch sees (tools/lq_family_scan.py leaves them as
<out>/family/<key>.pre.spv).
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess

from lq_and3 import COMPARE, DIS, Prog
from lq_family_scan import is_family_vertex, words

CMP_EVAL = {   # compare(0, c) with the AND nulled to 0, the AND on the left
    'OpIEqual': lambda c: 0 == c, 'OpINotEqual': lambda c: 0 != c, 'OpUGreaterThan': lambda c: 0 > c,
    'OpUGreaterThanEqual': lambda c: 0 >= c, 'OpULessThan': lambda c: 0 < c, 'OpULessThanEqual': lambda c: 0 <= c,
    'OpSGreaterThan': lambda c: 0 > c, 'OpSGreaterThanEqual': lambda c: 0 >= c, 'OpSLessThan': lambda c: 0 < c,
    'OpSLessThanEqual': lambda c: 0 <= c}
SWAP = {'OpUGreaterThan': 'OpULessThan', 'OpULessThan': 'OpUGreaterThan', 'OpUGreaterThanEqual': 'OpULessThanEqual',
        'OpULessThanEqual': 'OpUGreaterThanEqual', 'OpSGreaterThan': 'OpSLessThan', 'OpSLessThan': 'OpSGreaterThan',
        'OpSGreaterThanEqual': 'OpSLessThanEqual', 'OpSLessThanEqual': 'OpSGreaterThanEqual'}


class Cfg:
    def __init__(self, text):
        self.blocks, self.order, self.merge = {}, [], {}
        cur = None
        for line in text.splitlines():
            s = line.strip()
            m = re.match(r'(%\d+) = OpLabel', s)
            if m:
                cur = m.group(1)
                self.blocks[cur] = []
                self.order.append(cur)
                continue
            if cur is None:
                continue
            m = re.match(r'(?:(%\d+) = )?(Op\w+)\s*(.*)', s)
            if not m:
                continue
            res, op, rest = m.groups()
            ops = rest.split()
            if op == 'OpSelectionMerge':
                self.merge[cur] = ops[0]
            self.blocks[cur].append((res, op, ops))
        self.def_block = {}
        for b, ins in self.blocks.items():
            for res, op, ops in ins:
                if res:
                    self.def_block[res] = b

    def succ(self, b):
        t = self.blocks[b][-1] if self.blocks[b] else None
        if not t:
            return []
        op, ops = t[1], t[2]
        if op == 'OpBranch':
            return [ops[0]]
        if op == 'OpBranchConditional':
            return [ops[1], ops[2]]
        if op == 'OpSwitch':
            return [x for x in ops[1:] if x.startswith('%')]
        return []

    def region(self, start, stop):
        seen, work = set(), [start]
        while work:
            b = work.pop()
            if b in seen or b == stop or b not in self.blocks:
                continue
            seen.add(b)
            work.extend(self.succ(b))
        return seen

    def arm_stats(self, start, stop, p):
        st = Counter()
        for b in self.region(start, stop):
            for res, op, ops in self.blocks[b][:-1]:
                st['ins'] += 1
                if op == 'OpLoad': st['loads'] += 1
                if op == 'OpStore': st['stores'] += 1
                if op == 'OpExtInst': st['ext'] += 1
                if op == 'OpBranchConditional': st['branches'] += 1
                if op in ('OpAccessChain', 'OpInBoundsAccessChain'):
                    ids = [t for t in ops if t.startswith('%')]
                    if p.var.get(ids[1]) == 'Uniform':
                        st['ubo'] += 1
                        if any(x not in p.consts for x in ids[2:]):
                            st['ubo_dyn'] += 1
        return st


def shape_of(arm):
    if arm is None:
        return 'empty'
    parts = []
    for k in ('ins', 'loads', 'ubo', 'ubo_dyn', 'branches', 'ext'):
        if arm[k]:
            parts.append(f'{k}{arm[k]}')
    return ','.join(parts) or 'empty'


def analyze(path):
    text = subprocess.run([DIS, '--raw-id', '--no-header', str(path)], capture_output=True, text=True).stdout
    p, cfg = Prog(text), Cfg(text)
    out = []
    for r in p.ands:
        ids = [t for t in p.ins[r][1] if t.startswith('%')]
        x, y = ids[1], ids[2]
        if p.consts.get(y) == 3: other = x
        elif p.consts.get(x) == 3: other = y
        else: continue
        d = p.ins.get(other)
        sh = [t for t in d[1] if t.startswith('%')] if d else []
        if d and d[0] == 'OpShiftRightArithmetic' and len(sh) >= 3 and p.consts.get(sh[2]) == 31:
            continue                                         # round
        # comp: the last index of a uniform access chain
        comp, uses, work, seen = False, [], [r], set()
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
                    uses.append((res, op, pos, uids, v))
        if comp or not uses:
            continue
        for res, op, pos, uids, v in uses:
            rec = {'cmp': op, 'src': r}
            if op in COMPARE:
                a, b = uids[1], uids[2]
                c = p.consts.get(b if a == v else a)
                opn = op if a == v else SWAP.get(op, op)
                rec['cmp'] = opn
                rec['const'] = c
                forced = CMP_EVAL[opn](c) if c is not None and opn in CMP_EVAL else None
                rec['forced'] = forced
                # follow the bool through OpLogicalNot to its consumers
                bools, neg, seenb = [(res, False)], [], set()
                kinds = []
                while bools:
                    bv, inv = bools.pop()
                    if bv in seenb: continue
                    seenb.add(bv)
                    for r2, op2, pos2, u2 in [(a_[0], a_[1], a_[2], a_[3]) for a_ in p.uses.get(bv, [])]:
                        if op2 == 'OpLogicalNot' and r2:
                            bools.append((r2, not inv))
                        elif op2 == 'OpBranchConditional' and pos2 == 0:
                            blk = None
                            for b_, ins in cfg.blocks.items():
                                if ins and ins[-1][1] == 'OpBranchConditional' and ins[-1][2][0] == bv:
                                    blk = b_
                            if blk is None: continue
                            term = cfg.blocks[blk][-1][2]
                            tt, ff = term[1], term[2]
                            mg = cfg.merge.get(blk)
                            at, af = cfg.arm_stats(tt, mg, p), cfg.arm_stats(ff, mg, p)
                            cv = None if forced is None else (forced != inv)
                            kinds.append(('branch', cv, shape_of(at), shape_of(af)))
                        elif op2 == 'OpSelect' and pos2 == 1:
                            vals = [p.consts.get(t, t) for t in u2[2:4]]
                            cv = None if forced is None else (forced != inv)
                            kinds.append(('select', cv, tuple('c%s' % (x,) if isinstance(x, int) else 'v' for x in vals), None))
                        elif op2 in ('OpLogicalOr', 'OpLogicalAnd'):
                            kinds.append((op2[2:], None, None, None))
                        else:
                            kinds.append((op2[2:], None, None, None))
                rec['kinds'] = kinds
            else:
                rec['kinds'] = [(op[2:], None, None, None)]
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modules', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    files = [f for f in sorted(a.modules.glob('*.pre.spv')) if is_family_vertex(words(f))]
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(analyze, files))
    shapes, mods, example = Counter(), defaultdict(set), {}
    total = 0
    for f, recs in zip(files, results):
        for rec in recs:
            total += 1
            for kind in rec['kinds'] or [('none', None, None, None)]:
                key = (rec['cmp'], rec.get('const'), kind[0], kind[1], kind[2], kind[3])
                shapes[key] += 1
                mods[key].add(f.stem)
                example.setdefault(key, f.stem)
    lines = [f'{len(files)} family vertex modules; {total} test-class &3 ANDs (an AND feeding several compares counts once per use)',
             'count modules  compare(AND, const)  feeds  [nulled compare value -> forced side]  true-arm / false-arm shape (or select values)  [example]']
    for key, n in shapes.most_common():
        cmpop, c, kind, cv, s1, s2 = key
        forced = '?' if cv is None else ('true side' if cv else 'false side')
        lines.append(f'{n:5d} {len(mods[key]):5d}  {cmpop[2:]} {c}  {kind}  forced={forced}  {s1} / {s2}  [{example[key][:12]}]')
    text = '\n'.join(lines) + '\n'
    (a.out / 'and3-test-detail.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
