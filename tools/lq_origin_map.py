"""Offline: which shader-cache modules belong to an NVN 2 stage origin (the hash in the runtime's lq-trace vs=/fs= fields
and the DKB_SHADER_DUMP file names).

    # 1. lower the dumped stages (tools/dkb_lower_test.cpp), one <origin>.code each:
    clang++ -std=c++20 -O1 -I runtime/third_party/eglantine tools/dkb_lower_test.cpp -o build/dkb_lower_test.exe
    build/dkb_lower_test.exe <dump folder> <lowered folder>
    # 2. map them:
    python tools/lq_origin_map.py --cache build/dkb100-lqpatch/shader-cache --lowered <lowered folder> --out <out folder> \
        --xlate build/dkb100-lqpatch/totk_xlate.exe

The cache stores, per input id, the translator's exact input (cache/in/<iid>.in: the 0x880-byte control, the code with its
0x30-byte header, the constants and the answers). A lowered stage is the same program as an .in file when the .in code is
the stage's lowered code behind that header. The .in's translation is repeated here (the job the runtime builds: code
from +0x30, c1, attribute types) and compared word for word with the modules in the cache; a match is that
variant's cache module. Writes <out>/origin-map.txt and origin-map.json ({origin: [cache key, ...]}). The cache key is not derivable from the
.in name offline (the runtime hashes its own state in), so the translation is compared instead. CPU only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ATTR = {0: 'float', 1: 'sint', 2: 'uint'}
TEX = [None, '1d', '1darray', 'buffer', '2d', '2darray', '2dms', '2dmsarray', '3d', 'cube', 'cubearray']


def read_in(path):
    b = path.read_bytes()
    magic, ver, stage, length, c1len, ans = struct.unpack_from('<6I', b, 0)
    base = 24 + ans
    return {'stage': stage, 'length': length, 'c1len': c1len, 'ans': b[24:24 + ans],
            'code': b[base + 0x880:base + 0x880 + length],
            'c1': b[base + 0x880 + length:base + 0x880 + length + c1len]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--lowered', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--xlate', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    low = {p.stem: p.read_bytes() for p in a.lowered.glob('*.code')}
    cand = {k: [] for k in low}
    for f in sorted((a.cache / 'in').glob('*.in')):
        info = read_in(f)
        if info['stage'] != 0:
            continue
        body = info['code'][0x30:]
        for k, v in low.items():
            n = min(len(v), len(body))
            if n >= 256 and body[:n] == v[:n] and abs(len(body) - len(v)) <= 0x200:
                cand[k].append((f.stem, info))
    # cache modules by content hash
    spv_by_hash = {}
    for f in a.cache.glob('*.spv'):
        spv_by_hash[hashlib.sha1(f.read_bytes()).hexdigest()] = f.stem
    jobs, meta = [], {}
    for k, lst in cand.items():
        for iid, info in lst:
            code = a.out / f'{iid}.code'
            code.write_bytes(info['code'][0x30:])
            ans = info['ans']
            attr = {i: ATTR.get(ans[96 + i], 'float') for i in range(32) if ans[96 + i]}
            tex = {i: TEX[ans[i]] for i in range(32) if 0 < ans[i] < len(TEX)}
            job = {'key': iid, 'stage': 'vs', 'debug': False, 'code': str(code), 'c1': info['c1'].hex(),
                   'tex': {str(i): t for i, t in tex.items()}, 'img': {}, 'attr': {str(i): t for i, t in attr.items()},
                   'fout': {}, 'shared': 0}
            jobs.append(job)
            meta[iid] = k
    (a.out / 'jobs.json').write_text(json.dumps(jobs), encoding='utf-8')
    outdir = a.out / 'xlated'
    subprocess.run([str(a.xlate), str(a.out / 'jobs.json'), str(outdir)], capture_output=True, timeout=1800)
    lines, found = [], {}
    for k in sorted(cand):
        lines.append(f'{k}: {len(cand[k])} candidate input(s)')
        for iid, _ in cand[k]:
            sp = outdir / f'{iid}.spv'
            if not sp.exists():
                lines.append(f'    {iid[:12]}: translation failed')
                continue
            h = hashlib.sha1(sp.read_bytes()).hexdigest()
            key = spv_by_hash.get(h)
            if key: found.setdefault(k, []).append(key)
            lines.append(f'    in {iid[:12]} -> cache module {key[:12] if key else "(no identical module)"}'
                         + (f' ({key})' if key else ''))
    text = '\n'.join(lines) + '\n'
    (a.out / 'origin-map.txt').write_text(text, encoding='utf-8')
    (a.out / 'origin-map.json').write_text(json.dumps(found, indent=1), encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
