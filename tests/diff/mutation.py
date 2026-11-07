"""Check whether diff/lift.py detects deliberate changes to the lifter.

Each mutation is applied to an in-memory copy of lifter.py in a separate Python process. The checkout is never
modified. The unmodified control must pass, and a mutant counts as caught only if the differential test reports
actual mismatches; a crash, compiler error, or empty sample is an error rather than evidence of detection.

Like diff/lift.py, this needs the user's ExeFS and writes game-derived data only under build/.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIFTER = ROOT / 'recomp' / 'lifter.py'

# (name, unique source fragment, replacement): deliberately plausible mistakes in independent instruction families.
MUTATIONS = [
    ('add-sub-immediate', '{"-" if opc else "+"}', '{"+" if opc else "-"}'),
    ('add-sub-flags', "('subs' if opc else 'adds')", "('adds' if opc else 'subs')"),
    ('logical-immediate-and', "f'{a} & {hx(imm)}', sp=True", "f'{a} | {hx(imm)}', sp=True"),
    ('move-wide-keep', 'mask = ~(0xFFFF << (hw * 16)) & ones(64)',
     'mask = ~(0x0FFF << (hw * 16)) & ones(64)'),
]

# Preload the changed module before lift.py imports it. The child gets one mutation, then exits, so the real module
# and subsequent mutants cannot see each other's changes.
CHILD = r'''
import runpy, sys, types
from pathlib import Path
root, name = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root / 'tests' / 'diff'))
sys.path.insert(0, str(root / 'recomp'))
import mutation
path = root / 'recomp' / 'lifter.py'
source = path.read_text(encoding='utf-8')
if name != 'control':
    _, old, new = next(m for m in mutation.MUTATIONS if m[0] == name)
    if source.count(old) != 1:
        raise SystemExit('mutation anchor is missing or ambiguous: ' + name)
    source = source.replace(old, new, 1)
module = types.ModuleType('lifter')
module.__file__ = str(path)
sys.modules['lifter'] = module
exec(compile(source, str(path), 'exec'), module.__dict__)
sys.argv = [str(root / 'tests' / 'diff' / 'lift.py')] + sys.argv[3:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--exefs', type=Path, required=True)
    ap.add_argument('--modules', default='main,sdk')
    ap.add_argument('--encodings', type=int, default=20000, help='encodings per module, as in diff/lift.py')
    ap.add_argument('--cases', type=int, default=3)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'test' / 'mutation')
    args = ap.parse_args()
    if args.encodings < 1 or args.cases < 1:
        ap.error('--encodings and --cases must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    extra = ['--exefs', str(args.exefs.resolve()), '--modules', args.modules, '--encodings', str(args.encodings),
             '--cases', str(args.cases), '--seed', str(args.seed)]
    failed = False
    for name in ['control'] + [m[0] for m in MUTATIONS]:
        p = subprocess.run([sys.executable, '-c', CHILD, str(ROOT), name, *extra], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
        (args.out / f'{name}.log').write_text(p.stdout, encoding='utf-8')
        match = re.search(r'\b(\d+) encodings, (\d+) cases: (\d+) failing cases, (\d+) failing encodings', p.stdout)
        valid = match is not None and int(match[1]) > 0 and int(match[2]) > 0
        if name == 'control':
            ok = valid and p.returncode == 0 and int(match[3]) == 0
            print(f'{"PASS" if ok else "ERROR"} control (log: {args.out / (name + ".log")})', flush=True)
            if not ok:
                return 1
        else:
            caught = valid and p.returncode == 1 and int(match[3]) > 0 and 'harness failed:' not in p.stdout
            survived = valid and p.returncode == 0 and int(match[3]) == 0
            status = 'CAUGHT' if caught else 'SURVIVED' if survived else 'ERROR'
            print(f'{status} {name} (log: {args.out / (name + ".log")})', flush=True)
            failed |= not caught
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
