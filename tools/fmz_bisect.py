"""Find the shaders whose translation decides a GPU fault, by mixing two builds' shader caches and running the game.

    python tools/fmz_bisect.py --bad build/dkb100-cal --good build/dkb100-fmz --stage GroundLevelStage \
        --env TOTK_FIX_VS_FAULT=0

The two builds have the same game and different translators: with --bad the stage ends `device lost` (or crashes),
with --good it completes (here: FMZ on every shader, which keeps the #9 fault away). Every shader both caches hold
and translate differently is a candidate. A test copies the bad build into build/<bad>-bisect with the bad cache, puts
the good translation of a subset of the candidates under the bad translator's keys, and runs tools/scenario.py on the
stage once (--warp, so any save reaches it). The subset that turns the fault off is narrowed down (ddmin: halves,
then the complement of a half, then finer splits), until removing any one shader brings the fault back. Before that it
checks that the bad cache faults and the full swap does not; if either fails, the result would mean nothing, so it
stops and says why.

How the keys work (runtime/gfx_shader.c, "Keys"): a shader's input id does not depend on the translator; its cache
key is sha256("TKSK", KEY_VERSION, identity, input id), where identity is the build's totk_xlate.id mixed with the
host options the runtime logs ("gfx_shader: host options ...", read here from the newest log of the bad build). Every
key is checked against the files before a run, so a wrong identity stops the tool instead of testing nothing.

Each test is about 80 s, plus --cooldown (90 s) after each one that loses the GPU. Run one test at a time: half
of the tests lose the GPU on purpose, and three at once crashed Windows (bugcheck 0x119, VIDEO_SCHEDULER_INTERNAL_ERROR)
on an RTX 5060, so --jobs > 1 needs --allow-parallel-faults. The result names the shaders by input id, with their stored inputs (in/<id>.in) and both
SPIR-V files, and writes everything to build/bisect/<stamp>/result.json. Standard library only.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGGER_S = 20    # seconds between the starts of parallel tests
KEY_VERSION = 5   # runtime/gfx_shader.c KEY_VERSION; checked against the source below
GAME_FILES = ('dkb.exe', 'SDL3.dll', 'libiconv-2.dll', 'libopus-0.dll', 'romfs-index.bin', 'totk.map', 'totk.ini',
              'totk_xlate.exe', 'totk_xlate.id')


def key_version():
    m = re.search(r'#define KEY_VERSION (\d+)', (ROOT / 'runtime' / 'gfx_shader.c').read_text(encoding='utf-8'))
    return int(m.group(1)) if m else KEY_VERSION


def host_options(build):
    """The host options string the runtime mixes into the identity, from the newest log of a run of this build."""
    # A log's first line is the command, with the exe as it was given (relative or absolute): match on its folder name.
    exe = f'\\{build.name.lower()}\\dkb.exe'
    logs = sorted((ROOT / 'build' / 'scenarios').glob('*/*/game.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs:
        text = log.read_text(encoding='utf-8', errors='replace')
        first = '\\' + text.split('\n', 1)[0].lower().replace('/', '\\')
        if exe not in first:
            continue
        m = re.search(r'gfx_shader: host options (.*?): own shader cache keys', text)
        return ('TKHO ' + m.group(1)) if m else ''
    raise SystemExit(f'fmz_bisect: no scenario log of {build}; run any scenario with --build {build} first')


class Cache:
    def __init__(self, build, host):
        self.dir = build / 'shader-cache'
        xid = bytes.fromhex((build / 'totk_xlate.id').read_text().split()[0])
        self.ident = hashlib.sha256(xid + host.encode()).digest() if host else xid
        self.kv = key_version()
        self.inputs = {p.stem for p in (self.dir / 'in').glob('*.in')}
        missing = [i for i in list(self.inputs)[:200] if not (self.dir / (self.key(i) + '.spv')).is_file()]
        if missing:
            raise SystemExit(f'fmz_bisect: {len(missing)} of the first 200 inputs of {self.dir} have no file under the '
                             'computed key: the identity or KEY_VERSION is wrong for this build')

    def key(self, iid):
        return hashlib.sha256(b'TKSK' + struct.pack('<I', self.kv) + self.ident + bytes.fromhex(iid)).hexdigest()

    def files(self, iid):
        k = self.key(iid)
        return self.dir / f'{k}.spv', self.dir / f'{k}.res.json'


def candidates(bad, good):
    out = []
    for iid in sorted(bad.inputs & good.inputs):
        bs, gs = bad.files(iid)[0], good.files(iid)[0]
        if bs.is_file() and gs.is_file() and bs.read_bytes() != gs.read_bytes():
            out.append(iid)
    return out


def make_build(bad_build, bad, good, swap, dest):
    """dest: the bad build's game and translator, its cache, and the good translation of the shaders in swap."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in GAME_FILES:
        if (bad_build / f).is_file():
            shutil.copy2(bad_build / f, dest / f)
    shutil.copytree(bad.dir, dest / 'shader-cache', ignore=shutil.ignore_patterns('*.pack'))
    if any(bad.dir.glob('*.pack')):
        raise SystemExit('fmz_bisect: the bad cache is packed (tools/pack_shader_cache.py); use an unpacked one')
    for iid in swap:
        for src, dst in zip(good.files(iid), bad.files(iid)):
            shutil.copy2(src, dest / 'shader-cache' / dst.name)


def safe_signal(r, after):
    """The reset-free fault signal of a --safe-gpu run: accesses GPU-AV skipped, plus the shader sanitizer's
    undefined results (NaN into min/max and the like) from shaders whose first report came at or after `after`
    seconds of game time. Reports from before that (the title screen and the load) belong to every run."""
    g = r.get('gpuav') or {}
    late = sum(v['count'] for v in (g.get('undefined_results') or {}).values()
               if v.get('from_world_s') is not None and v['from_world_s'] >= after)
    return (g.get('errors') or 0) + late


def run(build, stage, env, seconds, log, force=False, safe_after=None):
    """'fault', 'ok' or 'unclear' for one run of the stage, and the run's result.json. With safe_after, the run
    goes through GPU-AV safe mode (tools/scenario.py --safe-gpu) and 'fault' means safe_signal() found reports."""
    cmd = [sys.executable, 'tools/scenario.py', 'tutorial-cave', '--build', str(build), '--warp', stage,
           '--seconds', str(seconds)] + [a for e in env for a in ('--env', e)] + (['--force'] if force else [])
    if safe_after is not None:
        cmd.append('--safe-gpu')
    with open(log, 'a', encoding='utf-8') as f:
        f.write('$ ' + subprocess.list2cmdline(cmd) + '\n')
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        f.write(out.stdout + out.stderr)
    m = re.search(r'results in (\S+)', out.stdout)
    if not m:
        return 'unclear', None
    results = sorted((ROOT / m.group(1)).glob('*/result.json'))
    r = json.loads(results[-1].read_text(encoding='utf-8')) if results else None
    if r is None:
        return 'unclear', None
    if r['ending'] in ('device lost', 'crashed', 'crashed, no report'):
        return 'fault', r
    if r['ending'] == 'completed' and r.get('in_scene') is not False:
        if safe_after is not None and safe_signal(r, safe_after):
            return 'fault', r
        return 'ok', r
    return 'unclear', r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bad', type=Path, required=True, help='build whose run faults')
    ap.add_argument('--good', type=Path, required=True, help='build whose run completes (same game, other translator)')
    ap.add_argument('--stage', required=True, help='the stage to warp to (tools/save_warp.py --stages)')
    ap.add_argument('--env', action='append', default=[], metavar='KEY=VALUE', help='for every run (repeatable)')
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--max-tests', type=int, default=40, help='stop after this many runs (default 40, about an hour)')
    ap.add_argument('--dry-run', action='store_true', help='check the keys and count the candidates, run nothing')
    ap.add_argument('--jobs', '-j', type=int, default=1, help='tests at once (1-3): each round splits the candidates '
                    'into at least this many parts and runs them together; about a third of the time with 3')
    ap.add_argument('--candidates', metavar='RESULT_JSON', help='start from the remaining candidates of an earlier run')
    ap.add_argument('--cooldown', type=float, default=90, help='seconds to wait after a test that lost the GPU (default 90)')
    ap.add_argument('--safe-gpu', dest='safe_after', type=float, nargs='?', const=30.0, metavar='SECONDS',
                    help='reset-free: run every test under GPU-AV safe mode (tools/scenario.py --safe-gpu), which '
                    'skips the faulting access, and count a test as faulting when the shader sanitizer reports '
                    'undefined results from shaders first seen at or after SECONDS of game time (default 30, after '
                    'the warp has loaded). About 4 min a test once the shaders are cached')
    ap.add_argument('--allow-parallel-faults', action='store_true', help='allow --jobs > 1 (risks a driver crash)')
    a = ap.parse_args(argv)
    if not 1 <= a.jobs <= 3:
        ap.error('--jobs takes 1 to 3')
    if a.jobs > 1 and not a.allow_parallel_faults:
        # Half of a bisection's tests lose the GPU on purpose. Three at once reset the NVIDIA driver three times in seven
        # minutes and then stopped Windows with bugcheck 0x119 (VIDEO_SCHEDULER_INTERNAL_ERROR) on an RTX 5060.
        ap.error('--jobs > 1 runs GPU faults in parallel, which crashed Windows (bugcheck 0x119) on an RTX 5060; '
                 'use --jobs 1, or add --allow-parallel-faults if you accept that risk')
    bad_build, good_build = (ROOT / a.bad).resolve(), (ROOT / a.good).resolve()
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out = ROOT / 'build' / 'bisect' / f'{stamp}-{a.stage}'
    out.mkdir(parents=True)
    log = out / 'runs.log'
    bad, good = Cache(bad_build, host_options(bad_build)), Cache(good_build, host_options(good_build))
    cand = candidates(bad, good)
    print(f'fmz_bisect: {len(cand)} shaders translate differently ({len(bad.inputs)} in {a.bad}, {len(good.inputs)} '
          f'in {a.good}); stage {a.stage}; log {out.relative_to(ROOT)}', flush=True)
    if a.dry_run:
        shutil.rmtree(out)
        return 0
    test_dir = ROOT / 'build' / f'{bad_build.name}-bisect'
    tests = []
    lock = threading.Lock()

    def one(swap, why, slot):
        # Each parallel test has its own build folder; the starts are staggered so loads do not collide, and
        # scenario.py gets --force because the other games are this tool's own tests.
        if slot:
            time.sleep(slot * STAGGER_S)
        d = test_dir if a.jobs == 1 else test_dir.with_name(f'{test_dir.name}-{slot}')
        make_build(bad_build, bad, good, swap, d)
        verdict, r = run(d, a.stage, a.env, a.seconds, log, a.jobs > 1, a.safe_after)
        if verdict == 'unclear':   # one retry: a run that ended neither way says nothing
            verdict, r = run(d, a.stage, a.env, a.seconds, log, a.jobs > 1, a.safe_after)
        if r and r['ending'] == 'device lost':   # let the driver settle after a GPU reset before the next test
            time.sleep(a.cooldown)
        with lock:
            tests.append(dict(why=why, swapped=len(swap), verdict=verdict, ending=r and r['ending'],
                              detail=r and r['ending_detail']))
            print(f'  test {len(tests):2}: {len(swap):5} of {len(cand)} good  -> {verdict:7} ({why})', flush=True)
        return verdict

    def test_all(jobs_):
        """[(swap, why)] -> verdicts, up to a.jobs at once."""
        if len(tests) + len(jobs_) > a.max_tests:
            raise TimeoutError
        if a.jobs == 1 or len(jobs_) == 1:
            return [one(s, w, 0) for s, w in jobs_]
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futs = [pool.submit(one, s, w, i % a.jobs) for i, (s, w) in enumerate(jobs_)]
            return [f.result() for f in futs]

    def test(swap, why):
        return test_all([(swap, why)])[0]

    def remaining(cur):   # written after every narrowing, so a stopped run can be resumed with --candidates
        (out / 'remaining.json').write_text(json.dumps(dict(stage=a.stage, remaining=cur), indent=1) + '\n',
                                            encoding='utf-8')

    def finish(status, culprits=None):
        res = dict(status=status, remaining=culprits or [], stage=a.stage, bad=str(a.bad), good=str(a.good), env=a.env, candidates=len(cand),
                   tests=tests, culprits=[dict(input_id=i, stored_input=str(bad.dir / 'in' / f'{i}.in'),
                                               bad_spirv=str(bad.files(i)[0]), good_spirv=str(good.files(i)[0]))
                                          for i in (culprits or [])])
        (out / 'result.json').write_text(json.dumps(res, indent=2) + '\n', encoding='utf-8')
        print(f'fmz_bisect: {status}' + (f': {len(culprits)} shader(s)' if culprits else ''))
        for c in res['culprits']:
            print(f'  {c["input_id"]}\n    stored input {c["stored_input"]}\n    bad  {c["bad_spirv"]}\n    good {c["good_spirv"]}')
        print(f'  details: {out.relative_to(ROOT) / "result.json"}')
        return 0 if culprits else 1

    try:
        if a.candidates:   # resume a narrowed run: start from the candidates a finished or stopped run left
            keep = set(json.loads(Path(a.candidates).read_text(encoding='utf-8')).get('remaining', []))
            cand = [i for i in cand if i in keep]
            print(f'  starting from {len(cand)} candidates of {a.candidates}', flush=True)
        sanity = test_all([([], 'sanity: the bad cache must fault'),
                           (cand, 'sanity: every candidate good must not fault')])
        if sanity[0] != 'fault':
            return finish('the bad build did not fault on this stage: nothing to bisect')
        if sanity[1] != 'ok':
            return finish('swapping every candidate does not stop the fault: the cause is outside these shaders '
                          '(not in both caches, or not shader translation)')
        # ddmin over the swapped set: the smallest set of good translations that still keeps the fault away. With
        # --jobs N each round splits into at least N parts and tests them at once.
        cur, base = list(cand), max(2, a.jobs)
        n = base
        while len(cur) >= 2:
            n = min(n, len(cur))
            chunks = [cur[i * len(cur) // n:(i + 1) * len(cur) // n] for i in range(n)]
            verdicts = test_all([(part, f'one of {n} parts of {len(cur)}') for part in chunks])
            hit = next((part for part, v in zip(chunks, verdicts) if v == 'ok'), None)
            if hit is not None:
                cur, n = hit, base
                remaining(cur)
                continue
            if n > 2:
                rests = [[i for i in cur if i not in set(part)] for part in chunks]
                verdicts = test_all([(rest, f'all but one of {n} parts of {len(cur)}') for rest in rests])
                hit = next((rest for rest, v in zip(rests, verdicts) if v == 'ok'), None)
                if hit is not None:
                    cur, n = hit, max(n - 1, base)
                    remaining(cur)
                    continue
            if n >= len(cur):
                break
            n = min(len(cur), n * 2)
        return finish('found', cur)
    except TimeoutError:
        return finish(f'stopped after {a.max_tests} tests (raise --max-tests to go on)')


if __name__ == '__main__':
    sys.exit(main())
