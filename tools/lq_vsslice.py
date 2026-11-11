"""Offline: the instruction slice a vertex output needs, from spirv-dis text of masked modules (tools/lq_vsmask_analyze.py
writes the modules). CPU only: no Vulkan, no GPU.

    for m in 00001 00002 00004 00008; do spirv-dis --raw-id --no-header vertex-mask-$m.spv > dis-$m.txt; done
    spirv-dis --raw-id --no-header vertex.spv > dis-orig.txt
    python tools/lq_vsslice.py --dir build/quad-replay/analysis --keep 00008 --clean 00001,00002,00004 \
        [--list] [--annotate FILE]

Reads dis-<mask>.txt for the original and each mask, and prints the function-body instructions of the --keep module that
none of the --clean modules keep: an opcode histogram, what they read (uniform buffer indices, inputs, builtins) and with
--annotate a listing with constants and variables written inline. ADCE keeps ids, so the same instruction prints the same
in every module and its text is the key.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re


def body(path):
    lines, inside = [], False
    for ln in path.read_text(encoding='utf-8').splitlines():
        s = ln.strip()
        if re.search(r'= OpFunction ', s):
            inside = True
            continue
        if s == 'OpFunctionEnd':
            inside = False
        if inside and s:
            lines.append(s)
    return lines


def opcode(s):
    m = re.match(r'(?:%\d+ = )?(Op\w+)', s)
    return m.group(1) if m else '?'


class Module:
    """Names, constants and variable kinds of the original module."""

    def __init__(self, path):
        self.names, self.consts, self.locs, self.builtin, self.kind = {}, {}, {}, {}, {}
        for ln in path.read_text(encoding='utf-8').splitlines():
            s = ln.strip()
            m = re.match(r'OpName (%\d+) "(.*)"', s)
            if m: self.names[m.group(1)] = m.group(2)
            m = re.match(r'(%\d+) = OpConstant %\d+ (\S+)', s)
            if m: self.consts[m.group(1)] = m.group(2)
            m = re.match(r'OpDecorate (%\d+) Location (\d+)', s)
            if m: self.locs[m.group(1)] = int(m.group(2))
            m = re.match(r'OpDecorate (%\d+) BuiltIn (\w+)', s)
            if m: self.builtin[m.group(1)] = m.group(2)
            m = re.match(r'(%\d+) = OpVariable %\d+ (Input|Output|Uniform)', s)
            if m: self.kind[m.group(1)] = m.group(2)

    def label(self, t):
        if t in self.names: return self.names[t]
        if t in self.consts: return '#' + self.consts[t]
        if self.kind.get(t) == 'Input': return 'in_' + (('L%d' % self.locs[t]) if t in self.locs else self.builtin.get(t, t))
        if self.kind.get(t) == 'Output': return 'out_L%s' % self.locs.get(t, '?')
        return None

    def annotate(self, s):
        return re.sub(r'%\d+', lambda m: m.group(0) + ('{' + self.label(m.group(0)) + '}' if self.label(m.group(0)) else ''), s)


def unique_lines(d, keep, clean):
    k = body(d / f'dis-{keep}.txt')
    c = set()
    for m in clean.split(','):
        c |= set(body(d / f'dis-{m}.txt'))
    return [s for s in k if s not in c], k, c


def reads(mod, lines):
    """What the instructions read: uniform rows by buffer (constant row -> components), computed indices, inputs."""
    ubo, comp, inp = defaultdict(lambda: defaultdict(set)), Counter(), Counter()
    for s in lines:
        m = re.match(r'%\d+ = OpAccessChain %\d+ (%\d+)((?: %\d+)+)', s)
        if m and m.group(1) in mod.names and mod.kind.get(m.group(1)) == 'Uniform':
            base = mod.names[m.group(1)]
            idx = re.findall(r'%\d+', m.group(2))   # [member 0, row, component]
            if len(idx) >= 3 and idx[1] in mod.consts and idx[2] in mod.consts:
                ubo[base][int(mod.consts[idx[1]])].add(int(mod.consts[idx[2]]))
            else:
                comp[base] += 1
        for v in re.findall(r'%\d+', s):
            if mod.kind.get(v) == 'Input' and not s.startswith('OpDecorate'): inp[mod.label(v)] += 1
    return ubo, comp, inp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', type=Path, required=True)
    ap.add_argument('--keep', required=True)
    ap.add_argument('--clean', required=True)
    ap.add_argument('--list', action='store_true', help='print every unique line')
    ap.add_argument('--annotate', type=Path, help='write the unique lines with constants and variables written inline')
    a = ap.parse_args()
    mod = Module(a.dir / 'dis-orig.txt')
    uniq, keep, clean = unique_lines(a.dir, a.keep, a.clean)
    print(f'orig body {len(body(a.dir / "dis-orig.txt"))} lines; mask {a.keep}: {len(keep)}; clean union: {len(clean)}; '
          f'unique to {a.keep}: {len(uniq)}')
    for op, n in Counter(opcode(s) for s in uniq).most_common():
        print(f'  {n:5d} {op}')
    ubo, comp, inp = reads(mod, uniq)
    for b in sorted(ubo):
        rows = ', '.join(f'{r}.{"".join("xyzw"[c] for c in sorted(cs))}' for r, cs in sorted(ubo[b].items()))
        print(f'uniform {b}: {len(ubo[b])} constant rows (row.components): {rows}')
    for b, n in comp.items():
        print(f'uniform {b}: {n} computed-index access chains')
    for k, n in sorted(inp.items()):
        print(f'reads {k}: {n} uses')
    if a.annotate:
        a.annotate.write_text('\n'.join(mod.annotate(s) for s in uniq) + '\n', encoding='utf-8')
    if a.list:
        print('\n'.join(uniq))


if __name__ == '__main__':
    main()
