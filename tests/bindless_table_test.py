"""Compile and run tests/bindless_table_test.c (runtime/bindless_table.h, the slot allocator of TOTK_BINDLESS_TEX). No Vulkan, no game."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with the project\'s Clang executable')
    out = ROOT / 'build' / 'test' / 'bindless-table'
    out.mkdir(parents=True, exist_ok=True)
    exe = out / ('bindless_table.exe' if os.name == 'nt' else 'bindless_table')
    subprocess.run([args.cc, '-std=gnu11', '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined' if os.name != 'nt' else '-O1',
                    '-I', str(ROOT / 'runtime'), str(ROOT / 'tests' / 'bindless_table_test.c'), '-o', str(exe)], check=True)
    return subprocess.run([str(exe)], timeout=60).returncode


if __name__ == '__main__':
    raise SystemExit(main())
