"""The bindless-texture translator (tools/build_xlate_bindless.py): descriptor-indexed sampling, spirv-val, and the unchanged default.

Two hand-assembled Maxwell fragment shaders sample bound textures (TEX): one 2D texture, and a 2D plus a 3D texture whose results go
to different colour channels. Each goes through the bindless translator three ways and through the default translator:
  - the default translator and the bindless one without a "bindless_tex" field give identical SPIR-V and reflection (the field is the
    only switch, so the default path is untouched);
  - with "bindless_tex": {"set": 2, "push_offset": 64} the module passes spirv-val for Vulkan 1.3, declares RuntimeDescriptorArray
    and SPV_EXT_descriptor_indexing, has one runtime-array variable per texture class in set 2 (binding = class: 2D 1, 3D 2) and no
    texture variable in set 0, one push constant block whose array member sits at offset 64, and loads the slot of each texture
    from it before indexing the array; the reflection gives every texture its bindless_slot, ranked by binding;
  - a texture buffer, a separate texture or an image stays out of the table (not exercised without game shaders: the predicate is
    the same one in Declarations.cs and Program.cs; see the patches).
No game files and no GPU.

  python tests/xlate_bindless_test.py --xlate build/xlate/totk_xlate.exe [--bindless build/xlate-bindless/totk_xlate.exe] --spirv-val spirv-val
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PT = 7 << 16
SCHED = 0x001F8000FC0007E0
EXIT = 0xE30000000000000F | PT
OP_CAPABILITY, OP_EXTENSION, OP_DECORATE, OP_MEMBER_DECORATE, OP_VARIABLE, OP_ACCESS_CHAIN = 17, 10, 71, 72, 59, 65
CAP_RUNTIME_DESCRIPTOR_ARRAY, DEC_BLOCK, DEC_DESCRIPTOR_SET, DEC_BINDING, DEC_OFFSET = 5302, 2, 34, 33, 35
SC_UNIFORM_CONSTANT, SC_PUSH_CONSTANT = 0, 9


def mov32i(dest, value):
    imm = struct.unpack('<I', struct.pack('<f', value))[0]
    return 0x0100000000000000 | (imm << 20) | (0xF << 12) | PT | dest


def tex(dest, coords, dim, wmask, tid):
    """TEX (bound texture tid, implicit LOD): dim 2 = 2D, 4 = 3D; the coordinates start at register coords."""
    return (0xC000000000000000 | dest | coords << 8 | PT | 0xFF << 20 | dim << 28 | wmask << 31 | tid << 36 | 1 << 49 | 7 << 51)


def fragment(ops):
    header = [0] * 20
    header[0] = 2 | (3 << 5) | (5 << 10)   # SPH type 2, version 3, stage fragment
    header[18] = 0xF                       # OMAP: render target 0, xyzw
    words = []
    for i in range(0, len(ops), 3):
        batch = ops[i:i + 3]
        words += [SCHED] + batch + [0] * (3 - len(batch))
    return struct.pack('<20I', *header) + struct.pack(f'<{len(words)}Q', *words)


def instructions(data):
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    assert words[0] == 0x07230203
    at = 5
    while at < len(words):
        count, op = words[at] >> 16, words[at] & 0xffff
        assert count and at + count <= len(words)
        yield op, words[at + 1:at + count]
        at += count


def translate(exe, job, out):
    line = json.dumps({'job': job, 'out': str(out)})
    r = subprocess.run([str(exe), '--serve'], input=line + '\n', capture_output=True, text=True, timeout=120)
    if r.stdout.strip() != 'ok ' + job['key']:
        raise SystemExit(f'{exe.name}: {job["key"]}: {r.stdout.strip()} {r.stderr[-300:]}')
    return (out / (job['key'] + '.spv')).read_bytes(), json.loads((out / (job['key'] + '.res.json')).read_text())


def validate(val, path, label):
    r = subprocess.run([str(val), '--target-env', 'vulkan1.3', str(path)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f'spirv-val rejects {label}:\n{r.stdout}{r.stderr}')


def check(cond, what):
    if not cond:
        raise SystemExit('FAIL: ' + what)


def analyse(spv, set_index, push_offset):
    """The checks of a bindless module: capabilities, table variables, the push constant block."""
    caps, exts = set(), []
    decor = {}      # id -> {decoration: [operands]}
    member_offsets = {}
    variables = []  # (id, storage class, pointer type)
    access_chains = 0
    pointee = {}
    for op, a in instructions(spv):
        if op == OP_CAPABILITY:
            caps.add(a[0])
        elif op == OP_EXTENSION:
            exts.append(struct.pack('<%dI' % len(a), *a).split(b'\0')[0].decode())
        elif op == OP_DECORATE:
            decor.setdefault(a[0], {})[a[1]] = a[2:]
        elif op == OP_MEMBER_DECORATE and a[2] == DEC_OFFSET:
            member_offsets[a[0]] = a[3]
        elif op == OP_VARIABLE:
            variables.append((a[1], a[2], a[0]))
        elif op == 32:
            pointee[a[0]] = a[2]
        elif op == OP_ACCESS_CHAIN:
            access_chains += 1
    check(CAP_RUNTIME_DESCRIPTOR_ARRAY in caps, 'RuntimeDescriptorArray capability')
    check('SPV_EXT_descriptor_indexing' in exts, 'SPV_EXT_descriptor_indexing extension')
    table = {d[DEC_BINDING][0]: v for v, sc, _ in variables if sc == SC_UNIFORM_CONSTANT
             for d in [decor.get(v, {})] if DEC_DESCRIPTOR_SET in d and d[DEC_DESCRIPTOR_SET][0] == set_index and DEC_BINDING in d}
    set0 = [v for v, sc, _ in variables if sc == SC_UNIFORM_CONSTANT and decor.get(v, {}).get(DEC_DESCRIPTOR_SET, [None])[0] == 0
            and DEC_BINDING in decor.get(v, {})]
    pushes = [(v, t) for v, sc, t in variables if sc == SC_PUSH_CONSTANT]
    check(len(pushes) == 1, 'exactly one push constant block variable')
    check(not set0, 'no texture variable stays in set 0')
    check(member_offsets.get(pointee.get(pushes[0][1])) == push_offset, f'the slot array sits at offset {push_offset}: {member_offsets}')
    check(access_chains >= len(table), 'a slot is read per texture')
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xlate', type=Path, required=True, help='the default translator')
    ap.add_argument('--bindless', type=Path, default=ROOT / 'build' / 'xlate-bindless' / ('totk_xlate.exe' if os.name == 'nt' else 'totk_xlate'))
    ap.add_argument('--spirv-val', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if not args.bindless.is_file():
        print(f'skipped: no bindless translator at {args.bindless} (python tools/build_xlate_bindless.py)')
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shaders = {
            'tex2d': (fragment([mov32i(4, 0.5), mov32i(5, 0.5), tex(0, 4, 2, 0xF, 90), EXIT, 0]), {'0': '2d'}, {1: 1}),
            # two textures: a 2D one into the red channel and a 3D one into the others; slot 0 is the 2D texture (lower binding)
            'tex2d3d': (fragment([mov32i(4, 0.5), mov32i(5, 0.5), mov32i(6, 0.25), mov32i(7, 0.25), mov32i(8, 0.25),
                                  tex(0, 4, 2, 0x1, 90), tex(1, 6, 4, 0xE, 92), EXIT, 0]), {'0': '2d', '1': '3d'}, {1: 1, 2: 2}),
        }
        for name, (code, types, classes) in shaders.items():
            path = tmp / (name + '.bin')
            path.write_bytes(code)
            job = {'key': name, 'stage': 'ps', 'code': str(path), 'debug': False, 'fout': {'0': 'float'}, 'tex': types}
            d_out, b_out, t_out = tmp / 'd', tmp / 'b', tmp / 't'
            default_spv, default_res = translate(args.xlate, job, d_out)
            plain_spv, plain_res = translate(args.bindless, job, b_out)
            check(default_spv == plain_spv, f'{name}: the bindless translator without the option makes the default module')
            check(default_res == plain_res, f'{name}: and the default reflection')
            check(all('bindless_slot' not in t for t in default_res['textures']), f'{name}: no bindless_slot without the option')
            spv, res = translate(args.bindless, dict(job, bindless_tex={'set': 2, 'push_offset': 64}), t_out)
            out = tmp / (name + '.bindless.spv')
            out.write_bytes(spv)
            validate(args.spirv_val, out, name)
            table = analyse(spv, 2, 64)
            check(set(table) == set(classes.values()), f'{name}: one table variable per texture class, got bindings {sorted(table)}')
            slots = sorted(t['bindless_slot'] for t in res['textures'])
            check(slots == list(range(len(res['textures']))), f'{name}: bindless slots {slots}')
            ranked = [t['bindless_slot'] for t in sorted(res['textures'], key=lambda t: t['binding'])]
            check(ranked == sorted(ranked), f'{name}: slots rank by binding')
            # bindless buffers: the record in the push constants (PhysicalStorageBuffer64), texture slots read from it
            spv2, res2 = translate(args.bindless, dict(job, bindless_tex={'set': 2, 'push_offset': 64}, bindless_buf={'record_offset': 8}), tmp / 'r')
            out2 = tmp / (name + '.record.spv')
            out2.write_bytes(spv2)
            validate(args.spirv_val, out2, name + ' (record)')
            caps2 = {a[0] for op, a in instructions(spv2) if op == OP_CAPABILITY}
            check(5347 in caps2, f'{name}: PhysicalStorageBufferAddresses capability with a record')
            check('bindless_buffers' in res2, f'{name}: the reflection gives the buffer count')
            check(sorted(t['bindless_slot'] for t in res2['textures']) == list(range(len(res2['textures']))), f'{name}: record texture slots')
            print(f'{name}: ok ({len(res["textures"])} textures, classes {sorted(table)}, {len(spv)} bytes, default {len(default_spv)} bytes)')
            if args.out:
                args.out.mkdir(parents=True, exist_ok=True)
                (args.out / (name + '.bindless.spv')).write_bytes(spv)
                (args.out / (name + '.default.spv')).write_bytes(default_spv)
    print('xlate-bindless: passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
