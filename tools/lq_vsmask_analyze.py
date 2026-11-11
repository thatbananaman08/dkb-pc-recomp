"""Offline: applies DKB_LQ_VSMASK masks (runtime/spirv_opt.cpp spirv_lq_patch, bit 7) to a vertex SPIR-V module and prints
the entry function's instruction counts, to see which outputs carry the code. CPU only: no Vulkan, no GPU, no game.

    python tools/lq_vsmask_analyze.py --spv build/quad-replay-captures/ground-20261108-01/vertex.spv \
        --out build/quad-replay/analysis [--masks 1,2,4,10000] [--cxx C:/msys64/ucrt64/bin/clang++.exe] [--spirv-val ...]

Default masks: each single bit 0..10 and 16 (a mask keeps the outputs whose bit is set, so a single bit keeps one output
and shows the code that output needs). With --cut NAME every mask also applies that DKB_LQ_VSCUT cut (mask 'none' = the cut alone). Each patched module is written next to the table, validated with spirv-val when
that is found, and the table goes to <out>/vsmask-counts.txt.
"""
import argparse
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CUTS = ['tdepth', 'scale', 'rot', 'flags', 'pos', 'offset', 'center']   # runtime/spirv_opt.cpp vscut_table order
HARNESS = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "spirv_opt.h"
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    static uint32_t w[1 << 20];
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    size_t n = fread(w, 4, 1 << 20, f);
    fclose(f);
    char err[512] = "";
    uint64_t out_n = 0;
    uint32_t *o = nullptr;
    unsigned applied = 0;
    int r = spirv_lq_patch(w, n, (unsigned)strtoul(argv[3], nullptr, 10), &o, &out_n, &applied, err, sizeof err);
    if (r < 0) return fprintf(stderr, "%s\n", err), 3;
    if (r == 0) return 4;
    f = fopen(argv[2], "wb");
    fwrite(o, 4, out_n, f);
    fclose(f);
    printf("%s\n", err);
    return 0;
}
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spv', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--masks', default='1,2,4,8,10,20,40,80,100,200,400,10000', help='comma list of hex masks')
    ap.add_argument('--cxx', type=Path, default=Path('C:/msys64/ucrt64/bin/clang++.exe'))
    ap.add_argument('--spirv-val', type=Path, default=Path('C:/msys64/ucrt64/bin/spirv-val.exe'))
    ap.add_argument('--caps', action='store_true', help='also apply the caps patch')
    ap.add_argument('--cut', default='', help='DKB_LQ_VSCUT name (tdepth, scale, rot, flags, pos, offset, center) applied with every mask; use mask "none" for the cut alone')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    harness = a.out / 'vsmask_harness.cpp'
    harness.write_text(HARNESS, encoding='utf-8')
    exe = a.out / 'vsmask_harness.exe'
    libs = a.cxx.parent.parent / 'lib'
    subprocess.run([str(a.cxx), '-std=c++20', '-O1', f'-I{ROOT / "runtime"}', str(harness), str(ROOT / 'runtime/spirv_opt.cpp'),
                    f'-L{libs}', '-l:libSPIRV-Tools-opt.a', '-l:libSPIRV-Tools.a', '-o', str(exe)],
                   check=True, capture_output=True, text=True, timeout=600)
    lines = []
    for m in a.masks.split(','):
        mask = None if m == 'none' else int(m, 16)
        tag = (f'{mask:05x}' if mask is not None else 'none') + (f'-cut-{a.cut}' if a.cut else '')
        dst = a.out / f'vertex-mask-{tag}.spv'
        kinds = (128 | mask << 8 if mask is not None else 0) | (1 if a.caps else 0)
        if a.cut:
            kinds |= (CUTS.index(a.cut) + 1) << 25
        run = subprocess.run([str(exe), str(a.spv), str(dst), str(kinds)], capture_output=True, text=True, timeout=300)
        if run.returncode == 4:
            lines.append(f'{tag}: not a matched vertex module (Vertex entry point and an Input at Location 9; for a cut, also every cut id with its opcode)')
            continue
        if run.returncode != 0:
            lines.append(f'{tag}: failed: {run.stderr.strip()}')
            continue
        valid = 'spirv-val not found'
        if a.spirv_val.exists():
            v = subprocess.run([str(a.spirv_val), '--target-env', 'vulkan1.3', str(dst)], capture_output=True, text=True)
            valid = 'valid' if v.returncode == 0 else 'INVALID: ' + (v.stdout + v.stderr).strip().splitlines()[0]
        words = dst.stat().st_size // 4
        lines.append(f'{run.stdout.strip()}; module {words} words; {valid}')
    text = '\n'.join(lines) + '\n'
    (a.out / 'vsmask-counts.txt').write_text(text, encoding='utf-8')
    sys.stdout.write(text)


if __name__ == '__main__':
    main()
