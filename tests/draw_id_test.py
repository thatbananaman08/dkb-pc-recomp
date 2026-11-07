"""Compile and run the game-free ordered-draw identifier checks."""
import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    exe = args.out / ('draw_id_test.exe' if os.name == 'nt' else 'draw_id_test')
    subprocess.run([args.cc, '-std=c11', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                    str(ROOT / 'tests' / 'draw_id_check.c'), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()
