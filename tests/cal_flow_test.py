"""Regression fixture for CAL's fall-through edge.

The dynamic c[0] predicate selects either the fast write of 2 or CAL's slow write
of 1.  The unconditional BRA after CAL must skip fast; the old decoder loses that
block and lets slow fall through to fast.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PT = 7 << 16
SCHED = 0x001f8000fc0007e0
EXIT = 0xe30000000000000f | PT


def mov32i(dest, value):
    return 0x0100000000000000 | (struct.unpack('<I', struct.pack('<f', value))[0] << 20) | (0xf << 12) | PT | dest


def framed(ops):
    words = []
    for index in range(0, len(ops), 3):
        words += [SCHED, *ops[index:index + 3]]
    return words


def shader():
    # The relative targets use physical instruction addresses, which include the scheduling words.
    ldc = 0xef90000000000000 | (4 << 48) | (255 << 8) | PT | 4
    # ISETP.EQ P0, R4, 0, PT: predicate comes from a runtime c[0] load.
    isetp = (0x366 << 50) | (2 << 49) | (7 << 39) | PT | (4 << 8)
    ops = [ldc, isetp, 'fast_bra', 'cal', 'end_bra', mov32i(0, 2.0),
           mov32i(1, 0.0), mov32i(2, 0.0), mov32i(3, 1.0), EXIT,
           mov32i(0, 1.0), 0xe32000000000000f | PT]
    address = lambda index: 8 * (index + index // 3 + 1)
    # @!P0 BRA fast; CAL slow; BRA end; fast: write 2; end: exit; slow: write 1; RET.
    ops[2] = 0xe24000000000000f | (0 << 16) | (1 << 19) | (((address(5) - address(2) - 8) & 0xffffff) << 20)
    ops[3] = 0xe260000000000000 | PT | (((address(10) - address(3) - 8) & 0xffffff) << 20)
    ops[4] = 0xe24000000000000f | PT | (((address(9) - address(4) - 8) & 0xffffff) << 20)
    header = [0] * 20
    header[0] = 2 | (3 << 5) | (5 << 10)
    header[18] = 0xf
    return struct.pack('<20I', *header) + struct.pack('<' + 'Q' * len(framed(ops)), *framed(ops))


def words(path):
    data = path.read_bytes()
    return struct.unpack('<' + 'I' * (len(data) // 4), data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--translator', required=True, type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'build/test/cal-flow')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    code = args.out / 'cal-flow.bin'
    code.write_bytes(shader())
    job = {'key': 'cal-flow', 'stage': 'ps', 'code': str(code.resolve()), 'debug': False, 'fout': {'0': 'float'}}
    run = subprocess.run([str(args.translator.resolve()), '--serve'], input=json.dumps({'job': job, 'out': str(args.out)}) + '\n',
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0 and run.stdout.strip() == 'ok cal-flow', (run.stdout, run.stderr)
    spv = words(args.out / 'cal-flow.spv')
    assert spv[0] == 0x07230203
    # Both results must remain because c[0] controls P0. A fixed CFG contains two conditional
    # conditional branch for P0 and an unconditional post-CAL edge to end. The old decoder loses
    # the latter and emits only the P0 selection, making slow overwrite with 2.0.
    conditional = sum((word & 0xffff) == 250 for word in spv[5:])
    unconditional = sum((word & 0xffff) == 249 for word in spv[5:])
    assert 0x40000000 in spv and 0x3f800000 in spv
    assert conditional >= 1 and unconditional >= 2, f'CAL fall-through lost: {conditional} conditional, {unconditional} unconditional branches'
    print(f'CAL flow: {conditional} conditional + {unconditional} unconditional branches retain the dynamic fast/slow split')


if __name__ == '__main__':
    main()
