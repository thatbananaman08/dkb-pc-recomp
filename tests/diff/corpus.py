"""Replay a fixed, game-free instruction corpus against Unicorn and the DKB lifter.

The corpus supplies instruction words, not machine states. This runner uses deterministic generated states and
reports shapes outside its A57 single-instruction scope separately. Atomics need the paired/MAX-model harness;
instructions excluded from the source corpus are not tested. A skipped Unicorn case is never counted as a pass.
"""
import argparse
import gzip
import hashlib
import json
import os
import platform
import random
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path

import unicorn as uc
import capstone

from common import BUF, BUF_SIZE, ROOT, Emu, build_harness, diff_state, dis, run_harness
from lifter import Lifter, Unsupported
from lift import BUF_MID, classify, is_mem, make_case, to_test_body

IMAGE = 0x7000000000
PC = IMAGE + 0x200000
IMAGE_SIZE = 0x400000  # literal-load targets may be up to 1 MiB either side of the instruction
DEFAULT_CORPUS = ROOT / 'tests' / 'data' / 'pokopia-corpus.json.gz'
CORPUS_SHA256 = '6b9b43ebdbbb85a302db51417fb09c822da56ce97391eae068eed541ab6381cf'


def load_corpus(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as source:
            data = source.read()
    else:
        data = path.read_bytes()
    if path.resolve() == DEFAULT_CORPUS.resolve() and hashlib.sha256(data).hexdigest() != CORPUS_SHA256:
        raise ValueError('the pinned corpus does not match its upstream SHA-256')
    return json.loads(data)


def make_safe_case(rng, word, pc, decoder):
    fpcr, x, sp, nzcv, v = make_case(rng, word)
    # A register-offset instruction may use the same X register as both base and index. Choose a value whose
    # resulting effective address is inside BUF; an ordinary buffer pointer in both roles is not valid.
    rn, rm = (word >> 5) & 31, (word >> 16) & 31
    if is_mem(word) and (word & 0x3b200c00) == 0x38200800 and rn == rm and rn != 31:
        option = (word >> 13) & 7
        if option in (3, 7):
            decoded = next(decoder.disasm(struct.pack('<I', word), pc))
            memory = next(op for op in decoded.operands if op.type == capstone.arm64.ARM64_OP_MEM)
            x[rn] = BUF_MID // (1 + (1 << memory.shift.value))
        elif option in (2, 6):
            x[rn] = BUF + 0x1000
    return fpcr, x, sp, nzcv, v


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'test' / 'corpus')
    ap.add_argument('--limit', type=int, default=0, help='maximum shapes to examine, 0 for all')
    ap.add_argument('--words-per-shape', type=int, default=1, help='words per shape, 0 for all')
    ap.add_argument('--cases', type=int, default=2, help='deterministic states per accepted word')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--cc', help='Clang executable (defaults to the differential test compiler)')
    ap.add_argument('--cflags', default='-march=x86-64-v3' if platform.machine().lower() in ('amd64', 'x86_64') else '')
    args = ap.parse_args()
    if args.cc:
        os.environ['CC'] = args.cc
    if args.cases < 1 or args.limit < 0 or args.words_per_shape < 0:
        ap.error('--cases must be positive; --limit and --words-per-shape must be nonnegative')
    data = load_corpus(args.corpus)
    shapes = sorted(data['shapes'].items())
    if args.limit:
        # Spread a small run across the sorted corpus, rather than testing only its first few mnemonics.
        shapes = [shapes[i * len(shapes) // min(args.limit, len(shapes))] for i in range(min(args.limit, len(shapes)))]
    r = random.Random(args.seed)
    lifter = Lifter(IMAGE, 'c')
    counts, picks = Counter(), []
    for shape, record in shapes:
        if record['class'] != 'a57':
            counts['non-a57-atomic'] += classify(record['words'][0]) == 'skip'
            counts['non-a57-other'] += classify(record['words'][0]) != 'skip'
            continue
        words = record['words'][:args.words_per_shape] if args.words_per_shape else record['words']
        for w in words:
            if classify(w) == 'skip':
                counts['system-or-atomic-words'] += 1
                continue
            pc = PC + 4 * len(picks)
            try:
                ins = lifter.lift(w, pc)
                body = to_test_body(ins, pc)
            except Unsupported:
                counts['unsupported-words'] += 1
                continue
            picks.append((shape, w, pc, body, ins.kind))
    if not picks:
        raise RuntimeError('no supported A57 instructions selected')
    if 4 * len(picks) > IMAGE_SIZE // 4:
        raise RuntimeError('selected code would exceed the mapped literal-load range')
    print(f'shapes: {len(shapes)}, selected {len(picks)} words; excluded {dict(counts)}', flush=True)
    code = bytearray(IMAGE_SIZE)
    for _, w, pc, _, _ in picks:
        struct.pack_into('<I', code, pc - IMAGE, w)
    args.out.mkdir(parents=True, exist_ok=True)
    image = args.out / 'code.bin'
    image.write_bytes(code)
    exe = build_harness(args.out, [p[3] for p in picks], args.jobs, args.cflags)
    emu = Emu([(IMAGE, code)])
    decoder = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    decoder.detail = True
    buf = bytes(r.getrandbits(8) for _ in range(4096)) * (BUF_SIZE // 4096)
    emu.mu.mem_write(BUF, buf)
    cases, expected, skipped, tested = [], [], Counter(), set()
    for k, (shape, w, pc, _, kind) in enumerate(picks):
        for _ in range(args.cases):
            fpcr, x, sp, nzcv, v = make_safe_case(r, w, pc, decoder)
            try:
                ox, osp, onz, opc, ov = emu.run(pc, fpcr, x, sp, nzcv, v)
            except uc.UcError as exc:
                skipped[(shape, w, str(exc))] += 1
                if is_mem(w):
                    emu.mu.mem_write(BUF, buf)
                continue
            mem = is_mem(w)
            crc = zlib.crc32(bytes(emu.mu.mem_read(BUF, BUF_SIZE))) if mem else 0
            if mem:
                emu.mu.mem_write(BUF, buf)
            cases.append((k, fpcr, x, sp, nzcv, v, mem))
            expected.append((ox, osp, onz, opc, ov, crc))
            tested.add(shape)
    if not cases:
        raise RuntimeError('Unicorn executed no cases')
    actual, error = run_harness(exe, args.out, buf, cases, [(IMAGE, image)])
    if error or len(actual) != len(cases):
        raise RuntimeError(f'harness failed: {error}; got {len(actual)} of {len(cases)} cases')
    failures = 0
    for case, want, got in zip(cases, expected, actual):
        k = case[0]
        shape, w, pc, _, kind = picks[k]
        difference = diff_state(got, want, check_pc=kind != 'op', check_mem=case[6])
        if difference:
            failures += 1
            if failures <= 30:
                print(f'FAIL {shape}: {w:08x} {dis(w, pc)}: {"; ".join(difference[:4])}')
    for (shape, word, why), count in skipped.most_common(20):
        print(f'SKIP {shape}: {word:08x} {why} ({count} cases)')
    if len(skipped) > 20:
        print(f'  and {len(skipped) - 20} more skipped word/reason combinations')
    print(f'{len(tested)} shapes tested, {len(cases)} executed cases, {sum(skipped.values())} skipped by Unicorn, '
          f'{failures} mismatches')
    # Missing coverage is not a pass: the selected A57 sample must both lift and execute in Unicorn.
    return int(bool(failures or skipped or counts['unsupported-words']))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        sys.exit(f'corpus: {exc}')
