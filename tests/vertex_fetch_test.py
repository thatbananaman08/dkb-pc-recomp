import argparse
import subprocess
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--cc', required=True)
p.add_argument('--deps', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
exe = out / 'vertex_fetch_check.exe'
subprocess.run([a.cc, '-std=c11', '-Wall', '-Wextra', '-Werror', '-I', str(root / 'runtime'),
                '-I', str(Path(a.deps) / 'include'), str(root / 'tests' / 'vertex_fetch_check.c'), '-o', str(exe)], check=True)
subprocess.run([str(exe)], check=True)
