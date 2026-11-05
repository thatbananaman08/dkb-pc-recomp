"""Check a change at one of three depths, with one command: the tests, then the game on its scenarios.

    python tools/check.py --quick                     the tests your change can affect, and the cave    (~5 min)
    python tools/check.py --standard                  every test that needs no dump, a short tier 3, the cave in
                                                      three stages at once and DK's face                (~20-30 min)
    python tools/check.py --full                      everything: tier 3 in full, the 20 layers twice, the face
                                                      twice, the cave with native GS off and on        (hours)
    python tools/check.py --standard --build build/dkb100-face      check that build instead of build/dkb100
    python tools/check.py --quick --dry-run           print the steps, run nothing

Use --quick while working, --standard before a commit, --full before a pull request or a release.

Each step is a tools/test.py or tools/scenario.py command; its output goes to the console and to
build/check/<stamp>-<level>/<step>.log, and the summary to summary.json there. A step exits 0 (PASS), 77 (SKIP: a
save, a tool or a GPU is missing; the step says which) or anything else (FAIL). The exit code is 1 when a step failed.

Before the game steps it compares the build with the source: when runtime/, recomp/ or the translator changed after
the build's dkb.exe was written, the game steps would test old code, so it says so (and stops, unless
--allow-stale-build). Timings from these runs are not judged: the scenarios that check speed say so themselves, and the
parallel ones are marked unreliable.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SKIP = 77
# Sources that make a build stale when they are newer than its dkb.exe (generated C comes from recomp/).
BUILD_SOURCES = ('runtime', 'recomp', 'tools/xlate', 'tools/vendor/ryujinx', 'tools/build_game.py',
                 'tools/build_xlate.py')
THREE_STAGES = '^(GroundLevelStage|HillLevelStage|DesertLevelStage)$'


def steps(level, build):
    """[(name, argv, needs_build)] of a level."""
    test, scen = [PY, 'tools/test.py'], [PY, 'tools/scenario.py']
    b = ['--build', str(build)]
    # The tests use the build's own translator when it has one (tools/build_xlate.py --out <build>), not build/xlate,
    # which goes stale after a merge or a translator patch.
    xlate = (ROOT / build if not Path(build).is_absolute() else Path(build)) / 'totk_xlate.exe'
    if xlate.is_file():
        test = test + ['--set', f'xlate={xlate}']
    if level == 'quick':
        return [
            ('tests your change can affect', test + ['--changed', '--tier', '0-2'], False),
            ('cave (terrain, 75 s)', scen + ['tutorial-cave'] + b, True),
        ]
    if level == 'standard':
        return [
            ('tests, tiers 0-2', test + ['--tier', '0-2'], False),
            ('tests on the dump, short', test + ['--tier', '3', 'diff-lift', 'diff-func-quick', 'jumptables'], False),
            ('cave in three stages at once', scen + ['tutorial-cave', '--sweep', THREE_STAGES, '--jobs', '3'] + b, True),
            ("DK's face (no NaN)", scen + ['dk-face'] + b, True),
        ]
    if level == 'full':
        return [
            ('tests, tiers 0-3', test + ['--tier', '0-3'], False),
            ('the 20 layers, twice', scen + ['tutorial-cave', '--sweep', 'LevelStage$', '--repeat', '2', '--jobs', '3',
                                              '--seconds', '90'] + b, True),
            ("DK's face, twice", scen + ['dk-face', '--repeat', '2', '--jobs', '2'] + b, True),
            ('cave with native GS off and on', scen + ['tutorial-cave', '--ab', 'native_gs', '--repeat', '2',
                                                        '--jobs', '2'] + b, True),
        ]
    raise ValueError(level)


def newest_source(root=ROOT):
    """(mtime, path) of the newest file under BUILD_SOURCES, or (0, None)."""
    best = (0.0, None)
    for rel in BUILD_SOURCES:
        p = root / rel
        files = [p] if p.is_file() else (f for f in p.rglob('*') if f.is_file()) if p.is_dir() else []
        for f in files:
            if '__pycache__' in f.parts:
                continue
            t = f.stat().st_mtime
            if t > best[0]:
                best = (t, f)
    return best


def stale(build, root=ROOT):
    """Why the build in folder build would test old code, or None. Missing exe: 'no ...'."""
    exe = next((build / n for n in ('dkb.exe', 'dkb', 'totk.exe') if (build / n).is_file()), None)
    if exe is None:
        return f'no dkb.exe in {build} (build it with tools/build_game.py --out {build})'
    t, f = newest_source(root)
    if f is not None and t > exe.stat().st_mtime + 2:
        when = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M')
        built = datetime.datetime.fromtimestamp(exe.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        return (f'{f.relative_to(root)} changed at {when}, after {exe.relative_to(root) if exe.is_relative_to(root) else exe} '
                f'was built at {built}: rebuild (tools/build_game.py --out {build}, and tools/build_xlate.py --out {build} '
                'after a translator change)')
    return None


def run_step(name, argv, log):
    """Run one step, streaming its output to the console and to log. Returns (status, seconds)."""
    start = time.monotonic()
    env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    with open(log, 'w', encoding='utf-8', errors='replace') as f:
        f.write('$ ' + subprocess.list2cmdline(argv) + '\n')
        p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding='utf-8', errors='replace')
        for line in p.stdout:
            sys.stdout.write('    ' + line)
            f.write(line)
        code = p.wait()
    status = 'PASS' if code == 0 else 'SKIP' if code == SKIP else 'FAIL'
    return status, time.monotonic() - start, code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--quick', action='store_const', const='quick', dest='level')
    g.add_argument('--standard', action='store_const', const='standard', dest='level')
    g.add_argument('--full', action='store_const', const='full', dest='level')
    ap.add_argument('--build', type=Path, default=ROOT / 'build' / 'dkb100', help='game build folder (default build/dkb100)')
    ap.add_argument('--allow-stale-build', action='store_true', help='run the game steps on a build older than the source')
    ap.add_argument('--dry-run', action='store_true', help='print the steps, run nothing')
    a = ap.parse_args(argv)
    build = a.build if a.build.is_absolute() else ROOT / a.build
    plan = steps(a.level, build.relative_to(ROOT) if build.is_relative_to(ROOT) else build)
    why_stale = stale(build)
    print(f'check --{a.level}: {len(plan)} steps, build {build.relative_to(ROOT) if build.is_relative_to(ROOT) else build}')
    if why_stale:
        print(f'  build: {why_stale}')
    if a.dry_run:
        for name, cmd, needs_build in plan:
            print(f'  {name}: {subprocess.list2cmdline(cmd)}')
        return 0
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out = ROOT / 'build' / 'check' / f'{stamp}-{a.level}'
    out.mkdir(parents=True, exist_ok=True)
    results, started = [], time.monotonic()
    for i, (name, cmd, needs_build) in enumerate(plan, 1):
        if needs_build and why_stale and (why_stale.startswith('no ') or not a.allow_stale_build):
            note = why_stale if why_stale.startswith('no ') else 'build older than the source (--allow-stale-build runs it)'
            print(f'\n[{i}/{len(plan)}] {name}: SKIP, {note}')
            results.append(dict(step=name, status='SKIP' if why_stale.startswith('no ') else 'FAIL', seconds=0, note=note))
            continue
        print(f'\n[{i}/{len(plan)}] {name}', flush=True)
        status, secs, code = run_step(name, cmd, out / f'{i}.log')
        results.append(dict(step=name, status=status, seconds=round(secs), exit_code=code, log=str(out / f'{i}.log')))
    total = time.monotonic() - started
    print(f'\ncheck --{a.level} in {total / 60:.1f} min:')
    for r in results:
        print(f'  {r["status"]:4}  {r["seconds"] / 60:5.1f} min  {r["step"]}' + (f'  ({r["note"]})' if r.get('note') else ''))
    ok = not any(r['status'] == 'FAIL' for r in results)
    print('PASS' if ok else 'FAIL' + ' (each step\'s log is in ' + str(out.relative_to(ROOT)) + ')')
    (out / 'summary.json').write_text(json.dumps(dict(level=a.level, build=str(build), stale=why_stale, minutes=round(total / 60, 1),
                                                      steps=results), indent=2) + '\n', encoding='utf-8')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
