"""Compile game-free AArch64 programs, lift their functions, and compare execution with Unicorn.

The fixtures have no OS or external library dependencies. Calls between their functions execute real lifted
functions, rather than the call stubs used by diff/func.py. This does not cover TLS, exceptions or host services.
"""
import argparse
import os
import platform
import random
import subprocess
import sys
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
from elftools.elf.elffile import ELFFile
from unicorn import arm64_const as A

from common import BUF, BUF_SIZE, MSYS_BIN, ROOT, Emu, compiler, diff_state, run_harness
import gen
from lifter import Lifter
from module import Module

BASE = 0x7000000000
SENTINEL = 0x7210000000
SOURCE = ROOT / 'tests' / 'diff' / 'e2e_programs.c'


def run(cmd, env):
    subprocess.run([str(x) for x in cmd], check=True, env=env)


def read_program(path):
    with path.open('rb') as f:
        elf = ELFFile(f)
        section = elf.get_section_by_name('.text')
        symbols = elf.get_section_by_name('.symtab')
        if section is None or symbols is None:
            raise ValueError('linked program lacks .text or symbols')
        code = section.data()
        lo = int(section['sh_addr'])
        functions = {s.name: int(s['st_value']) for s in symbols.iter_symbols()
                     if s['st_info']['type'] == 'STT_FUNC' and lo <= int(s['st_value']) < lo + len(code)}
    entries = {name: addr for name, addr in functions.items() if name.startswith('entry_')}
    required = {'entry_calls', 'entry_loop', 'entry_switch', 'entry_tail'}
    if lo != BASE or len(code) % 4 or set(entries) != required:
        raise ValueError(f'unexpected AArch64 text layout or entry functions: {sorted(entries)}')
    return code, functions, entries


def synthetic_module(code, function_starts):
    # The normal Module methods and generator operate on a memory image; only NSO loading is replaced here.
    mod = Module.__new__(Module)
    mod.name = 'e2e'
    mod.base = mod.lo = BASE
    mod.hi = BASE + len(code)
    mod.image = code
    mod.words = np.frombuffer(code, dtype='<u4')
    mod.lifter = Lifter(BASE, 'c')
    mod._cache = OrderedDict()
    mod.seeds = lambda: sorted(set(function_starts))
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'test' / 'e2e')
    ap.add_argument('--cc', default=None, help='Clang executable (defaults to the differential test compiler)')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cc = args.cc or compiler()
    env = dict(os.environ)
    if os.name == 'nt' and MSYS_BIN.exists():
        env['PATH'] = str(MSYS_BIN) + os.pathsep + env.get('PATH', '')
    elf = out / 'program.elf'
    run([cc, '--target=aarch64-none-elf', '-O1', '-ffreestanding', '-fno-stack-protector',
         '-fno-jump-tables', '-fuse-ld=lld', '-nostdlib', '-Wl,-Ttext=0x7000000000',
         '-Wl,-e,entry_calls', '-o', elf, SOURCE], env)
    code, functions, entries = read_program(elf)
    image = out / 'program.bin'
    image.write_bytes(code)
    mod = synthetic_module(code, functions.values())
    indices, _ = gen.discover(mod)
    starts = [BASE + 4 * int(i) for i in indices]
    if not set(entries.values()) <= set(starts):
        raise RuntimeError('generator did not discover all program entries')
    emitter = gen.Emitter(mod, set(starts))
    ranges = list(zip(starts, starts[1:] + [mod.hi]))
    calls = tails = 0
    for entry, end in ranges:
        for address in range(entry, end, 4):
            ins = mod.lift(address)
            calls += ins.kind == 'bl' and ins.target in starts
            tails += ins.kind == 'b' and ins.target in starts and ins.target != entry
    if not calls or not tails:
        raise RuntimeError('fixture compilation did not produce both direct calls and cross-function tail branches')
    generated = out / 'program.c'
    generated.write_text(gen.chunk_text('e2e', 0, emitter, ranges), encoding='utf-8')
    if emitter.stats['traps'] or emitter.stats['external_calls'] or emitter.stats['external_branches']:
        raise RuntimeError(f'unsupported or external generated code: {dict(emitter.stats)}')
    names = sorted(entries)
    table = out / 'table.c'
    table.write_text('#include "cpu.h"\n' + ''.join(f'void {gen.fname(entries[n])}(Cpu *c);\n' for n in names) +
                     'GuestFn tests[] = {' + ','.join(gen.fname(entries[n]) for n in names) + '};\n' +
                     f'int ntests = {len(names)};\n', encoding='utf-8')
    exe = out / ('e2e.exe' if os.name == 'nt' else 'e2e')
    arch = ['-march=x86-64-v3'] if platform.machine().lower() in ('amd64', 'x86_64') else []
    run([cc, '-O1', '-fno-strict-aliasing', '-Wno-return-type', '-Wno-unused-label', *arch, '-I', ROOT / 'runtime',
         '-fuse-ld=lld', '-o', exe, generated, table, ROOT / 'tests' / 'diff' / 'harness.c'], env)

    emu = Emu([(BASE, code)])
    mu = emu.mu
    buf = bytes(BUF_SIZE)
    rng = random.Random(1)
    cases, expected, labels = [], [], []
    for k, name in enumerate(names):
        for arg in (0, 1, 2, 3, 15, 42, 0xffffffff, 0xffffffffffffffff):
            x = [rng.getrandbits(64) for _ in range(31)]
            x[0], x[1], x[30] = BUF + 0x1000, arg, SENTINEL
            sp, nzcv, v, fpcr = BUF + BUF_SIZE - 0x100, 0, [0] * 32, 0
            mu.mem_write(BUF, buf)
            mu.reg_write_batch(list(zip(emu.xregs, x)) + list(zip(emu.qregs, v)) +
                               [(A.UC_ARM64_REG_SP, sp), (A.UC_ARM64_REG_NZCV, nzcv), (A.UC_ARM64_REG_FPCR, fpcr)])
            mu.emu_start(entries[name], SENTINEL, count=100000)
            state = mu.reg_read_batch(emu.out_regs)
            if state[33] != SENTINEL:
                raise RuntimeError(f'{name}: Unicorn did not return for argument {arg:#x}')
            expected.append((list(state[:31]), state[31], state[32] & 0xf0000000, state[33], list(state[34:]),
                             zlib.crc32(bytes(mu.mem_read(BUF, BUF_SIZE)))))
            cases.append((k, fpcr, x, sp, nzcv, v, True))
            labels.append((name, arg))
    actual, error = run_harness(exe, out, buf, cases, [(BASE, image)])
    if error or len(actual) != len(cases):
        raise RuntimeError(f'harness failed: {error}; got {len(actual)} of {len(cases)} results')
    failures = 0
    for label, want, got in zip(labels, expected, actual):
        differences = diff_state(got, want, check_pc=False, check_mem=True, check_nzcv=False)
        if differences:
            failures += 1
            print(f'{label[0]}({label[1]:#x}): {"; ".join(differences[:4])}')
    print(f'{len(names)} programs, {len(cases)} cases, {failures} failures')
    return int(bool(failures))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f'e2e: {exc}')
