"""Offline proof for DKB_LQ_PATCH=and3exact: the patched vertex shader computes bit-identical results to the original.

    python tools/lq_and3exact_check.py --out build/quad-replay/analysis/and3exact-check build/quad-replay-captures/ground-67a23e6f*-0* ...

For every capture directory it patches vertex.spv with spirv_lq_patch kind and3exact (bit 9, the patcher compiled from
runtime/spirv_opt.cpp), validates both modules with spirv-val, and runs the original and the patched module with tools/lq_vseval.py on
the captured data and on variants of it: the float in constant buffer c9 row 22 (byte 352, the count the index arithmetic starts from)
is replaced by each value of --sweep, which moves the integer the `& 3` tests take through all four residues. The outputs of every
(instance, vertex) and the value of every test-class AND result are compared bit for bit. Writes <out>/and3exact-check.txt. CPU only.
"""
import argparse
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lq_and3_detail   # noqa: E402
import lq_vseval        # noqa: E402
from lq_vsmask_analyze import HARNESS   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KIND_AND3EXACT = 512


def canon(v):
    if isinstance(v, dict):
        return tuple(sorted((k, canon(x)) for k, x in v.items()))
    if isinstance(v, float):
        return ('f', struct.pack('<f', v)) if v == v else ('nan',)
    if isinstance(v, list):
        return tuple(canon(x) for x in v)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('captures', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--kind', type=int, default=KIND_AND3EXACT, help='spirv_lq_patch kinds: 512 and3exact, 1024 rotexact, 1536 both')
    ap.add_argument('--random', type=int, default=0, help='also run this many variants with the whole c9 block replaced by random 32-bit words (NaN and Inf included)')
    ap.add_argument('--sweep', default='0,1,2,3,4,5,6,7,8,9,10,11,13,17,21,100,2.5,-1,-3')
    ap.add_argument('--cxx', type=Path, default=Path('C:/msys64/ucrt64/bin/clang++.exe'))
    ap.add_argument('--spirv-val', type=Path, default=Path('C:/msys64/ucrt64/bin/spirv-val.exe'))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'harness.cpp').write_text(HARNESS, encoding='utf-8')
    exe = a.out / 'harness.exe'
    libs = a.cxx.parent.parent / 'lib'
    subprocess.run([str(a.cxx), '-std=c++20', '-O1', f'-I{ROOT / "runtime"}', str(a.out / 'harness.cpp'), str(ROOT / 'runtime/spirv_opt.cpp'),
                    f'-L{libs}', '-l:libSPIRV-Tools-opt.a', '-l:libSPIRV-Tools.a', '-o', str(exe)], check=True, capture_output=True, text=True, timeout=600)
    sweep = [float(x) for x in a.sweep.split(',')]
    lines, bad = [], 0
    for cap in a.captures:
        src = cap / 'vertex.spv'
        dst = a.out / f'{cap.name}.k{a.kind}.spv'
        run = subprocess.run([str(exe), str(src), str(dst), str(a.kind)], capture_output=True, text=True, timeout=300)
        if run.returncode != 0:
            lines.append(f'{cap.name}: patch failed or not applicable (exit {run.returncode}) {run.stderr.strip()[:100]}')
            bad += 1
            continue
        vals = [subprocess.run([str(a.spirv_val), '--target-env', 'vulkan1.3', str(p)], capture_output=True, text=True).returncode == 0 for p in (src, dst)]
        import re
        words = [int(x) for x in re.findall(r'%(\d+)', run.stdout.split('words', 1)[1])] if 'words' in run.stdout else []
        ids = sorted({int(r['src'][1:]) for r in lq_and3_detail.analyze(src)})
        nonfinite = 0
        word_seen = 0
        seen = {i: set() for i in ids}
        same = total = 0
        first_diff = None
        import random
        variants = [{9: [(22 * 16, struct.pack('<f', v))]} for v in sweep]
        for seed in range(a.random):
            rnd = random.Random(seed)
            size = (cap / 'ubo-9.bin').stat().st_size
            blob = bytearray(rnd.getrandbits(32).to_bytes(4, 'little') for _ in range(0)) or bytearray(b''.join(rnd.getrandbits(32).to_bytes(4, 'little') for _ in range(size // 4)))
            for k in range(0, size, 4):   # a share of NaN / Inf / zero words
                if rnd.random() < 0.08: blob[k:k + 4] = struct.pack('<I', rnd.choice([0x7fc00000, 0x7f800000, 0xff800000, 0]))
            variants.append({9: [(0, bytes(blob))]})
        for v, patch in enumerate(variants):
            mod0, rep0, res0, _ = lq_vseval.evaluate(cap, ubo_patch=patch, probe=ids + words)
            mod1, rep1, res1, _ = lq_vseval.evaluate(cap, spv=dst, ubo_patch=patch, probe=ids + words)
            for (l0, o0), (l1, o1), (pl0, p0), (pl1, p1) in zip(res0, res1, rep0.probes, rep1.probes):
                total += 1
                ok = canon(o0) == canon(o1) and l0 == l1 and all(v is None or canon(v) == canon(p0.get(i)) for i, v in p1.items())   # a dead value the patch removed reads as None
                same += ok
                if not ok and first_diff is None:
                    why = 'outputs' if canon(o0) != canon(o1) else 'probes ' + str([i for i, x in p1.items() if x is not None and canon(x) != canon(p0.get(i))])
                    first_diff = (v, f'{l0} ({why})')
                for wv in words:
                    x = p0.get(wv)
                    if isinstance(x, float):
                        word_seen += 1
                        nonfinite += x != x or x in (float('inf'), float('-inf'))
                for i in ids:
                    if p0.get(i) is not None:
                        seen[i].add(p0[i] & 3 if isinstance(p0[i], int) else p0[i])
            if len(res0) != len(res1):
                first_diff = first_diff or (v, 'different number of invocations')
        lines.append(f'{cap.name}: {run.stdout.strip()}; spirv-val original {"ok" if vals[0] else "FAILED"}, patched {"ok" if vals[1] else "FAILED"}; '
                     f'{same} of {total} invocations bit-identical over {len(variants)} variants ({len(sweep)} sweep values, {a.random} random c9 blocks)'
                     + (f'; FIRST DIFFERENCE at c9[22].x={first_diff[0]} {first_diff[1]}' if first_diff else ''))
        if words:
            lines.append(f'    word locals {words}: {word_seen} values seen, {nonfinite} NaN or Inf in the unrandomised captured runs and the random ones together')
        for i in ids:
            lines.append(f'    test-class AND %{i}: residues reached {sorted(seen[i])}')
        bad += (same != total) + (not all(vals))
    text = chr(10).join(lines) + chr(10) + ('ALL IDENTICAL' if not bad else f'{bad} PROBLEM(S)') + chr(10)
    (a.out / 'and3exact-check.txt').write_text(text, encoding='utf-8')
    sys.stdout.write(text)
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
