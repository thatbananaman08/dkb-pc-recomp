"""Conservatively classify direct fragment output writes across entry-point control flow.

This is an offline feasibility check, not a shader transformation or runtime proof.
This analysis is incomplete and must not be used to authorize instrumentation.
"""
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import re
import subprocess

ID = r'%[A-Za-z0-9_]+'


def analyze(text):
    lines = text.splitlines()
    entry = re.findall(rf'OpEntryPoint\s+Fragment\s+({ID})\s+"[^"]+"', text)
    if len(entry) != 1:
        return {'analysis_complete_for_listed_patterns': False, 'reasons': ['expected exactly one fragment entry point']}
    locations = {m[1]: int(m[2]) for line in lines if (m := re.search(rf'OpDecorate\s+({ID})\s+Location\s+(\d+)', line))}
    outputs = {m[1] for line in lines if (m := re.search(rf'^\s*({ID})\s*=\s*OpVariable\s+{ID}\s+Output\b', line))}
    consts = {m[1]: int(m[2]) for line in lines if (m := re.search(rf'^\s*({ID})\s*=\s*OpConstant\s+{ID}\s+(-?\d+)\s*$', line))}
    ptrs = {x: (x, ()) for x in outputs}
    for line in lines:
        if m := re.search(rf'^\s*({ID})\s*=\s*Op(?:InBounds)?AccessChain\s+{ID}\s+({ID})(.*)', line):
            if m[2] in ptrs:
                base, path = ptrs[m[2]]
                ptrs[m[1]] = base, path + tuple(re.findall(ID, m[3]))
    reasons = []
    blocks, function, current = {}, None, None
    for line in lines:
        if m := re.search(rf'^\s*({ID})\s*=\s*OpFunction\s', line):
            function, current = m[1], None
        elif 'OpFunctionEnd' in line:
            function, current = None, None
        elif m := re.search(rf'^\s*({ID})\s*=\s*OpLabel\b', line):
            current = m[1]
            if function == entry[0]:
                blocks[current] = {'writes': set(), 'succ': [], 'return': False}
        elif function:
            if m := re.search(rf'\bOpStore\s+({ID})\s+({ID})', line):
                if m[1] in ptrs:
                    if function != entry[0]:
                        reasons.append('output store outside entry function')
                    elif current in blocks:
                        base, path = ptrs[m[1]]
                        if len(path) != 1 or path[0] not in consts or consts[path[0]] not in range(4):
                            reasons.append('whole-vector or dynamic/aggregate output store is unsupported')
                        elif base not in locations:
                            reasons.append('output has no location decoration')
                        else:
                            blocks[current]['writes'].add((locations[base], consts[path[0]]))
            if 'OpFunctionCall' in line and any(x in re.findall(ID, line) for x in ptrs):
                reasons.append('output pointer passed to function')
            if function != entry[0] or current not in blocks:
                continue
            if m := re.search(rf'\bOpBranchConditional\s+{ID}\s+({ID})\s+({ID})', line):
                blocks[current]['succ'] = [m[1], m[2]]
            elif m := re.search(rf'\bOpBranch\s+({ID})', line):
                blocks[current]['succ'] = [m[1]]
            elif 'OpSwitch' in line:
                reasons.append('switch control flow unsupported')
            elif re.search(r'\bOpReturn\b', line):
                blocks[current]['return'] = True
    if not blocks:
        reasons.append('entry point has no blocks')
        return {'analysis_complete_for_listed_patterns': False, 'reasons': sorted(set(reasons))}
    start = next(iter(blocks))
    reachable, todo = set(), deque([start])
    while todo:
        b = todo.popleft()
        if b in reachable:
            continue
        if b not in blocks:
            reasons.append('branch target unavailable')
            continue
        reachable.add(b)
        todo.extend(blocks[b]['succ'])
    preds = defaultdict(set)
    for b in reachable:
        for target in blocks[b]['succ']:
            if target in reachable:
                preds[target].add(b)
    universe = set().union(*(blocks[b]['writes'] for b in reachable))
    after = {b: set(universe) for b in reachable}
    changed = True
    while changed:
        changed = False
        for b in reachable:
            before = set() if b == start else (set.intersection(*(after[p] for p in preds[b])) if preds[b] else set())
            value = before | blocks[b]['writes']
            if value != after[b]:
                after[b], changed = value, True
    returns = [b for b in reachable if blocks[b]['return']]
    if not returns:
        reasons.append('no reachable ordinary return')
    definite = set.intersection(*(after[b] for b in returns)) if returns else set()
    return {'analysis_complete_for_listed_patterns': not reasons, 'reasons': sorted(set(reasons)), 'entry': entry[0],
            'reachable_blocks': len(reachable), 'returns': len(returns),
            'definite_components': [list(x) for x in sorted(definite)],
            'possibly_written_components': [list(x) for x in sorted(universe)],
            'limitations': 'Only direct constant-index scalar output stores and branch control flow are analyzed; pointer aliasing and unsupported SPIR-V need independent validation.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('shader', type=Path)
    ap.add_argument('--spirv-dis', default='spirv-dis')
    a = ap.parse_args()
    r = subprocess.run([a.spirv_dis, str(a.shader)], capture_output=True, text=True, check=True)
    print(json.dumps(analyze(r.stdout), indent=2))


if __name__ == '__main__':
    main()
