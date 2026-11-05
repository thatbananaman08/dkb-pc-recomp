"""Regression fixture for an unconditional CAL followed by an unconditional BRA.

The helper writes 1 to R0.  The BRA after the return skips the fast-path write of 2,
so a decoder that loses CAL's fall-through translates the wrong final colour.
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
    ops = ['cal', 'bra', mov32i(0, 2.0), mov32i(1, 0.0), mov32i(2, 0.0), mov32i(3, 1.0), EXIT,
           mov32i(0, 1.0), 0xe32000000000000f | PT]
    address = lambda index: 8 * (index + index // 3 + 1)
    ops[0] = 0xe260000000000000 | (((address(7) - address(0) - 8) & 0xffffff) << 20)
    ops[1] = 0xe24000000000000f | PT | (((address(3) - address(1) - 8) & 0xffffff) << 20)
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
    # A fixed translator removes the unreachable fast-path literal (2.0 / 0x40000000)
    # and retains the helper's 1.0 result as the colour output.
    assert 0x40000000 not in spv, 'CAL fall-through lost: the post-CAL BRA did not skip the fast path'
    assert 0x3f800000 in spv
    print('CAL flow: fast-path 2.0 absent; post-CAL BRA retained')


if __name__ == '__main__':
    main()
