"""Offline: finds the family vertex modules in a shader cache and reports what a DKB_LQ_PATCH kind does to each one.

    python tools/lq_family_scan.py --cache build/dkb100-lqpatch/shader-cache --out build/quad-replay/analysis \
        --kinds flatten[,notrig,noflags,ubocomp] [--limit N] [--extra] [--all]

A family vertex module is a <key>.spv with a Vertex entry point and an Input variable decorated Location 9: the match
runtime/gfx_shader.c lq_patch hands to spirv_lq_patch (it sees the cache's modules after spirv_optimize, so this tool runs
the same --eliminate-local-single-block pass first). Writes <out>/family-<kinds>.txt: one row per module with the
OpBranchConditional / OpSelectionMerge / entry-instruction counts before and after, and whether spirv-val accepts the
result. --extra is the offline experiment: also CombineAccessChains, and the sequence twice with DCE. CPU only.
"""
import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BITS = {'fold': 4096, 'and3exact': 512, 'rotexact': 1024, 'loadexact': 2048, 'flatten': 1 << 28, 'notrig': 1 << 29, 'noflags': 1 << 30, 'caps': 1, 'ubocomp': 1 << 31, 'ldcfix': 256}
HARNESS = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "spirv_opt.h"
int main(int argc, char **argv) {
    if (argc != 5) return 2;
    static uint32_t w[1 << 22];
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    size_t n = fread(w, 4, 1 << 22, f);
    fclose(f);
    char err[512] = "";
    uint64_t pn = 0;
    uint32_t *pre = spirv_opt_local_single_block(w, n, &pn, err, sizeof err);   // what lq_patch sees
    if (!pre) return fprintf(stderr, "%s\n", err), 3;
    f = fopen(argv[3], "wb");
    fwrite(pre, 4, pn, f);
    fclose(f);
    uint64_t out_n = 0;
    uint32_t *o = nullptr;
    unsigned applied = 0;
    int r = spirv_lq_patch(pre, pn, (unsigned)strtoul(argv[4], nullptr, 10), &o, &out_n, &applied, err, sizeof err);
    if (r < 0) return fprintf(stderr, "%s\n", err), 5;
    if (r == 0) return 4;
    f = fopen(argv[2], "wb");
    fwrite(o, 4, out_n, f);
    fclose(f);
    printf("%s\n", err);
    return 0;
}
'''


def words(path):
    data = path.read_bytes()
    return struct.unpack('<%dI' % (len(data) // 4), data[:len(data) // 4 * 4])


def is_family_vertex(w):
    if len(w) < 5 or w[0] != 0x07230203:
        return False
    vertex, loc9, inputs = False, set(), set()
    i = 5
    while i < len(w):
        n = w[i] >> 16
        if not n:
            return False
        op = w[i] & 0xffff
        if op == 15 and w[i + 1] == 0:
            vertex = True
        elif op == 71 and n >= 4 and w[i + 2] == 30 and w[i + 3] == 9:
            loc9.add(w[i + 1])
        elif op == 59 and n >= 4 and w[i + 3] == 1:
            inputs.add(w[i + 2])
        elif op == 54:   # functions: the declarations are over
            break
        i += n
    return vertex and bool(loc9 & inputs)


def stats(w):
    """(instructions, OpBranchConditional, OpSelectionMerge, OpLoopMerge) of the whole module's functions."""
    i, ins, br, sm, lm, inside = 5, 0, 0, 0, 0, False
    while i < len(w):
        n, op = w[i] >> 16, w[i] & 0xffff
        if op == 54:
            inside = True
        elif op == 56:
            inside = False
        elif inside:
            ins += 1
            br += op == 250
            sm += op == 247
            lm += op == 246
        i += n
    return ins, br, sm, lm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--kinds', default='flatten')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--all', action='store_true', help='every module of the cache, not only the family vertex modules')
    ap.add_argument('--extra', action='store_true')
    ap.add_argument('--env', action='append', default=[], help='KEY=VALUE for the patcher (DKB_LQ_NOFLAGS_ONLY, DKB_LQ_AND3_CLASS, ...)')
    ap.add_argument('--cxx', type=Path, default=Path('C:/msys64/ucrt64/bin/clang++.exe'))
    ap.add_argument('--spirv-val', type=Path, default=Path('C:/msys64/ucrt64/bin/spirv-val.exe'))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    kinds = sum(BITS[k] for k in a.kinds.split(','))
    env = {**os.environ, **({'DKB_LQ_FLATTEN_EXTRA': '1'} if a.extra else {}), **dict(e.split('=', 1) for e in a.env)}
    work = a.out / 'family'
    work.mkdir(exist_ok=True)
    (work / 'scan_harness.cpp').write_text(HARNESS, encoding='utf-8')
    exe = work / 'scan_harness.exe'
    libs = a.cxx.parent.parent / 'lib'
    subprocess.run([str(a.cxx), '-std=c++20', '-O1', f'-I{ROOT / "runtime"}', str(work / 'scan_harness.cpp'),
                    str(ROOT / 'runtime/spirv_opt.cpp'), f'-L{libs}', '-l:libSPIRV-Tools-opt.a', '-l:libSPIRV-Tools.a',
                    '-o', str(exe)], check=True, capture_output=True, text=True, timeout=600)
    files = sorted(a.cache.glob('*.spv'))
    fam = [f for f in files if a.all or is_family_vertex(words(f))]
    family_keys = {f.stem for f in files if is_family_vertex(words(f))} if a.all else {f.stem for f in fam}
    if a.limit:
        fam = fam[:a.limit]
    tag = a.kinds.replace(',', '+') + ('+extra' if a.extra else '') + ('+all' if a.all else '')

    extra_counts, ldc_counts = {}, {}

    def one(f):
        key = f.stem[:12]
        pre, dst = work / f'{key}.pre.spv', work / f'{key}.{tag}.spv'
        run = subprocess.run([str(exe), str(f), str(dst), str(pre), str(kinds)], capture_output=True, text=True, timeout=600, env=env)
        # argv order of the harness: in, out, pre, kinds
        if run.returncode == 4:
            return key, None, 'not patched (nothing applies)', 0, f.stem in family_keys
        if run.returncode != 0:
            return key, None, f'failed ({run.returncode}): {run.stderr.strip()[:100]}', 0, f.stem in family_keys
        b, af = stats(words(pre)), stats(words(dst))
        v = subprocess.run([str(a.spirv_val), '--target-env', 'vulkan1.3', str(dst)], capture_output=True, text=True)
        valid = 'valid' if v.returncode == 0 else 'INVALID ' + (v.stdout + v.stderr).strip().splitlines()[0][:80]
        m = re.search(r'ubocomp: (\d+) dynamic component loads', run.stdout)
        cm = re.search(r'classes nulled: comp (\d+), round (\d+), test (\d+); unclassified left (\d+)', run.stdout)
        lm = re.search(r'ldcfix: (\d+) data-dependent', run.stdout)
        ldc_counts[key] = int(lm.group(1)) if lm else 0
        extra_counts[key] = tuple(int(x) for x in cm.groups()) if cm else (0, 0, 0, 0)
        return key, (b, af), valid, int(m.group(1)) if m else 0, f.stem in family_keys

    rows = list(ThreadPoolExecutor(max_workers=4).map(one, fam))
    lines = [f'{len(files)} modules in the cache, {len(fam)} modules scanned ' + ('(all of them)' if a.all else '(family vertex modules: Vertex, Input at Location 9)') + f'; kinds {a.kinds}'
             + (' +extra' if a.extra else ''),
             'module        instr before->after   branchcond  selmerge  loopmerge  result']
    tot = [0] * 8
    for key, st, note, _loads, _fam in rows:
        if st is None:
            lines.append(f'{key}  {note}')
            continue
        (i0, b0, s0, l0), (i1, b1, s1, l1) = st
        for k, v in enumerate((i0, i1, b0, b1, s0, s1, l0, l1)):
            tot[k] += v
        lines.append(f'{key}  {i0:6d} -> {i1:6d}   {b0:4d} -> {b1:4d}  {s0:4d} -> {s1:4d}  {l0:2d} -> {l1:2d}   {note}')
    lines.append(f'total         {tot[0]:6d} -> {tot[1]:6d}   {tot[2]:4d} -> {tot[3]:4d}  {tot[4]:4d} -> {tot[5]:4d}  {tot[6]:2d} -> {tot[7]:2d}')
    patched = [r for r in rows if r[1] is not None]
    lines.append(f'{len(patched)} of {len(rows)} scanned modules patched; {sum(1 for r in patched if r[2] == "valid")} valid, '
                 f'{sum(1 for r in patched if r[2] != "valid")} invalid')
    if any('AND3_CLASS' in e for e in a.env):
        t = [sum(c[k] for c in extra_counts.values()) for k in range(4)]
        lines.append(f'&3 ANDs nulled by class: comp {t[0]}, round {t[1]}, test {t[2]}; unclassified &3 left {t[3]}; '
                     f'modules with any nulled: {sum(1 for c in extra_counts.values() if any(c[:3]))}')
    if 'ldcfix' in a.kinds:
        lines.append(f'ldcfix: {sum(1 for c in ldc_counts.values() if c)} modules, {sum(ldc_counts.values())} branches removed')
    if 'ubocomp' in a.kinds:
        lines.append(f'ubocomp: {sum(1 for r in rows if r[3])} modules, {sum(r[3] for r in rows)} loads rewritten; '
                     f'in the family vertex modules: {sum(1 for r in rows if r[3] and r[4])} modules, '
                     f'{sum(r[3] for r in rows if r[4])} loads')
    text = '\n'.join(lines) + '\n'
    (a.out / f'family-{tag}.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
