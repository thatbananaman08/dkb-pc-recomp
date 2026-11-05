"""Run the game on a saved scenario without anyone playing, and say whether it worked.

    python tools/scenario.py --list                         the scenarios and whether their private save is present
    python tools/scenario.py tutorial-cave                  one run with the default build (build/dkb100)
    python tools/scenario.py tutorial-cave --repeat 3       three runs
    python tools/scenario.py tutorial-cave --ab native_gs   variants of the scenario, alternating: off, on, off, on, ...
    python tools/scenario.py tutorial-cave --build build/dkb100-pr67 --env TOTK_VK_VALIDATION=1
    python tools/scenario.py tutorial-cave --warp HillLevelStage    start in another stage (tools/save_warp.py)
    python tools/scenario.py tutorial-cave --sweep 'LevelStage$'    once per matching stage (the 20 layers)
    python tools/scenario.py tutorial-cave --repeat 2 --env TOTK_DUMP=4300 --env TOTK_DUMP_DIR={run}/dump
                                                         {run} in a value: that run's own folder

A scenario is tests/scenarios/<name>.toml: which private save to start from, the scripted input that gets from the title
screen into it, how long to run, and what must hold afterwards. The save itself is never in the repository: it is the
folder build/saves/<save>/ (account.uid and save/ of a game profile from your own console or an earlier run). The input
scripts are timings only (tools/input/), not game data.

Each run gets its own folder build/scenarios/<stamp>-<name>/<variant>-<n>/ with the game's whole output (game.log), the
last frame (last.png), the perf analysis (perf.json, tools/perf_check.py) and result.json. The profile is a fresh copy of
the save and is deleted after the run (--keep-profile keeps it), so a run never changes the save.

How a run ended is one of:
  completed        the game reached --seconds of game time and exited 0
  crashed          the crash reporter wrote FATAL (exit 71): the log has the host address, the lifted function, the
                   guest registers and frames; crash.txt has that part
  crashed, no report   the process died with no FATAL line (stack overflow, a fault in a driver or a validation layer,
                   or the reporter itself failing); the exit code says how (0xC0000005 access violation, 0xC00000FD stack
                   overflow, 0xC0000409 fast fail)
  device lost      Vulkan lost the device; the log has the VK_EXT_device_fault report when the driver gives one
  hung             no exit within the time limit (seconds x 6 + 180 s); the process was killed
  did not start    the exe could not be started (missing build, wrong path)
Then the checks: a perf verdict (OK, HITCHY or SLOW, tools/perf_check.py); that the last frame is in the scene (its
HUD band has bright pixels; a loading screen or fade has none, and then nothing about the picture is judged); and
the last frame's share of near-black pixels below the HUD band (the black destructible terrain of #45 shows as a
high share).

Before each run the machine is checked: another game process running stops the run (two games share the GPU); other
programs using more than 1 GiB of memory or less than 4 GiB of free commit memory mark the run as on a busy machine, and
its timings as unreliable. The exit code is 0 when every run completed and every check held, 1 otherwise, 2 for a usage
or setup error, 77 when the scenario's private save is missing (tools/test.py reports that as a skip).
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / 'tests' / 'scenarios'
SAVES = ROOT / 'build' / 'saves'
RUNS = ROOT / 'build' / 'scenarios'
WIN = os.name == 'nt'
GAME_EXES = ('dkb.exe', 'totk.exe') if WIN else ('dkb', 'totk')
sys.path.insert(0, str(ROOT / 'tools'))
import perf_check  # noqa: E402
import save_warp  # noqa: E402

SKIP = 77
BUSY_PRIVATE_MIB = 1024      # another program with this much private memory makes the machine busy
MIN_FREE_COMMIT_GIB = 4.0    # less free commit memory than this makes it busy


# ---------------------------------------------------------------- scenario files

def load(name):
    path = SCENARIOS / f'{name}.toml'
    if not path.is_file():
        known = ', '.join(sorted(p.stem for p in SCENARIOS.glob('*.toml'))) or 'none'
        raise SystemExit(f'scenario: no {_rel(path)} (known: {known})')
    s = tomllib.loads(path.read_text(encoding='utf-8'))
    for key in ('description', 'save', 'input', 'seconds'):
        if key not in s:
            raise SystemExit(f'scenario: {path.name} lacks "{key}"')
    s['name'] = name
    s.setdefault('env', {})
    s.setdefault('checks', {})
    s.setdefault('variants', {})
    return s


def save_dir(s):
    return SAVES / s['save']


def has_save(s):
    d = save_dir(s)
    return (d / 'save').is_dir() and (d / 'account.uid').is_file()


# ---------------------------------------------------------------- machine state

def machine_state(exe_names=GAME_EXES):
    """Other game processes, programs with over BUSY_PRIVATE_MIB of private memory, and free commit memory (Windows)."""
    if not WIN:
        return dict(games=[], heavy=[], free_commit_gib=None, checked=False)
    ps = ('$o = Get-CimInstance Win32_OperatingSystem; '
          '$p = Get-Process | Where-Object { $_.PrivateMemorySize64 -gt %d } | '
          'ForEach-Object { @{name = $_.ProcessName; id = $_.Id; mib = [int]($_.PrivateMemorySize64 / 1MB)} }; '
          '@{free_commit_kib = $o.FreeVirtualMemory; heavy = @($p)} | ConvertTo-Json -Depth 3 -Compress'
          % (BUSY_PRIVATE_MIB << 20))
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True, timeout=60).stdout
        data = json.loads(out)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return dict(games=[], heavy=[], free_commit_gib=None, checked=False)
    heavy = [h for h in data.get('heavy') or [] if h]
    games = [h for h in heavy if f'{h["name"].lower()}.exe' in exe_names]
    try:   # a game process with little memory still holds the GPU
        tl = subprocess.run(['tasklist', '/FO', 'CSV', '/NH'], capture_output=True, text=True, timeout=30).stdout
        for line in tl.splitlines():
            parts = line.strip('"').split('","')
            if parts and parts[0].lower() in exe_names and not any(g['id'] == int(parts[1]) for g in games):
                games.append(dict(name=parts[0], id=int(parts[1]), mib=None))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    others = [h for h in heavy if f'{h["name"].lower()}.exe' not in exe_names]
    free = data.get('free_commit_kib')
    return dict(games=games, heavy=others, free_commit_gib=free / 1048576 if free else None, checked=True)


def busy_reasons(m):
    why = [f'{h["name"]} ({h["mib"]} MiB)' for h in m['heavy']]
    if m['free_commit_gib'] is not None and m['free_commit_gib'] < MIN_FREE_COMMIT_GIB:
        why.append(f'only {m["free_commit_gib"]:.1f} GiB of commit memory free')
    return why


# ---------------------------------------------------------------- one run

FATAL = re.compile(r'^\[?\w*\]?\s*FATAL: (.*)$', re.M)
NTSTATUS = {0xC0000005: 'access violation', 0xC00000FD: 'stack overflow', 0xC0000409: 'fast fail (stack buffer overrun '
            'or __fastfail)', 0xC000001D: 'illegal instruction', 0xC0000094: 'integer divide by zero',
            0xC0000374: 'heap corruption', 0xC0000017: 'out of memory', 0x40010004: 'killed (debugger or Task Manager)'}


def ending(code, text, timed_out):
    """(kind, detail) for how a run ended, from its exit code and output."""
    if code is None and not timed_out:
        return 'did not start', text.strip()[-300:]
    lost = re.search(r'.*(device lost|VK_ERROR_DEVICE_LOST|upload wait failed \(-4\)).*', text)
    m = FATAL.search(text)
    if timed_out:
        return 'hung', 'no exit within the time limit; killed'
    if m:
        return 'crashed', m.group(1).strip()
    if lost:
        return 'device lost', lost.group(0).strip()
    if code == 0:
        if 'frametime: exit' not in text and 'perf: frame' not in text:
            return 'completed', 'exited 0 without perf output (TOTK_TRACE=perf missing?)'
        return 'completed', ''
    u = code & 0xFFFFFFFF
    name = NTSTATUS.get(u)
    return 'crashed, no report', f'exit {u:#x}' + (f' ({name})' if name else '') + '; no FATAL line in the log'


def crash_excerpt(text):
    """The crash reporter's block: the FATAL line and the indented lines after it."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if 'FATAL:' in line:
            block = [line]
            for nxt in lines[i + 1:i + 80]:
                if not nxt.startswith(' '):
                    break
                block.append(nxt)
            return '\n'.join(block) + '\n'
    return ''


def frame_stats(png, hud_band, threshold=8, bright=96):
    """(near-black share below the top hud_band, bright share inside it) of a frame; (None, None) without Pillow or
    the image. In gameplay the HUD (hearts, counters) puts a few percent of bright pixels in the band, also when the
    terrain is black; a loading screen or a fade has none (measured: 0.00% against 2.4-3.6% in the tutorial cave)."""
    try:
        from PIL import Image
    except ImportError:
        return None, None
    if not png.is_file():
        return None, None
    with Image.open(png) as im:
        g = im.convert('L')
        w, h = g.size
        below = g.crop((0, int(h * hud_band), w, h)).histogram()
        top = g.crop((0, 0, w, max(1, int(h * hud_band)))).histogram()
    nb, nt = sum(below), sum(top)
    return (sum(below[:threshold]) / nb if nb else None, sum(top[bright:]) / nt if nt else None)


def black_share(png, hud_band, threshold=8):
    """Share of pixels with luma below threshold in the frame below its top hud_band (None without Pillow or image)."""
    return frame_stats(png, hud_band, threshold)[0]


def run_env(s, env_extra, out, base):
    """The environment of one run: base, the scenario's [env], then env_extra (--env and the variant). {run} in a value
    is the run's folder, so repeated runs do not overwrite each other's outputs (TOTK_DUMP_DIR={run}/dump). perf is
    always traced, because the perf verdict needs it; other TOTK_TRACE categories asked for are added to it."""
    env = dict(base)
    env['TOTK_INPUT_SCRIPT'] = str(ROOT / s['input'])
    env.update({k: str(v) for k, v in s['env'].items()})
    env.update({k: str(v) for k, v in env_extra.items()})
    for k in set(s['env']) | set(env_extra):
        env[k] = env[k].replace('{run}', str(out))
    env['TOTK_TRACE'] = ','.join(sorted(set(filter(None, env.get('TOTK_TRACE', '').split(',') + ['perf']))))
    return env


def perf_fails(checks, verdict):
    """Speed is judged only when the scenario asks: perf = "SLOW" fails a SLOW run, "HITCHY" a SLOW or HITCHY one.
    Without it the verdict is reported and nothing fails on it."""
    want = checks.get('perf')
    return want == 'SLOW' and verdict == 'SLOW' or want == 'HITCHY' and verdict in ('SLOW', 'HITCHY')


def run_once(s, exe, out, env_extra, args, warp=None):
    out.mkdir(parents=True)
    prof = out / 'profile'
    shutil.copytree(save_dir(s), prof)
    warp = warp or args.warp or s.get('warp')
    if warp:   # this run's profile copy starts in another stage (tools/save_warp.py); the private save is unchanged
        save_warp.warp_profile(prof, warp)
    env = run_env(s, env_extra, out, os.environ)
    seconds = args.seconds or s['seconds']
    cmd = [str(exe), '--profile', str(prof), '--seconds', str(seconds), '--shot', str(out / 'last.png')]
    limit = seconds * 6 + 180
    m = machine_state()
    start = time.monotonic()
    code, timed_out, text = None, False, ''
    with open(out / 'game.log', 'w', encoding='utf-8', errors='replace') as log:
        log.write('$ ' + ' '.join(f'{k}={v}' for k, v in sorted(env_extra.items())) + ' ' + subprocess.list2cmdline(cmd)
                  + '\n')
        log.flush()
        try:
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            try:
                code = p.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                p.kill()
                p.wait()
        except OSError as e:
            log.write(f'\nscenario: could not start: {e}\n')
    wall = time.monotonic() - start
    text = (out / 'game.log').read_text(encoding='utf-8', errors='replace')
    kind, detail = ending(code, text, timed_out)
    if kind == 'crashed':
        (out / 'crash.txt').write_text(crash_excerpt(text), encoding='utf-8')
    checks = s['checks']
    perf = perf_check.analyse(out / 'game.log', checks.get('min_fps', 50), checks.get('hitch_ms', 100),
                              checks.get('skip_world', 5))
    (out / 'perf.json').write_text(json.dumps(perf, indent=2) + '\n', encoding='utf-8')
    black, hud = frame_stats(out / 'last.png', checks.get('hud_band', 0.15))
    in_scene = None if hud is None else hud >= checks.get('min_hud_bright', 0.005)
    failed = []
    if kind != 'completed':
        failed.append(kind)
    if perf_fails(checks, perf['verdict']):
        failed.append(f'perf {perf["verdict"]}')
    if kind == 'completed' and perf['verdict'] == 'no data':
        failed.append('no perf data')
    if kind == 'completed' and in_scene is False:
        # Judging a loading screen's black would report black terrain that is not there: say what happened instead.
        failed.append(f'not in the scene at the end (no HUD in the last frame: a loading screen, menu or fade?); '
                      f'give it more time with --seconds')
    elif 'max_black' in checks and kind == 'completed':
        if black is None:
            failed.append('no last frame to check')
        elif black > checks['max_black']:
            failed.append(f'near-black {black * 100:.1f}% > {checks["max_black"] * 100:.0f}%')
    if not args.keep_profile:
        shutil.rmtree(prof, ignore_errors=True)
    busy = busy_reasons(m)
    r = dict(scenario=s['name'], variant=out.name, warp=warp, env=env_extra, exe=str(exe), seconds=seconds, wall_s=round(wall, 1),
             exit_code=code, ending=kind, ending_detail=detail, perf=perf['verdict'], fps=perf['fps'],
             slow_windows=perf['slow_windows'], worst_frame_ms=perf['worst_ms'],
             major_episode_causes=sorted({e['cause'] for e in perf['episodes'] if e['major']}),
             black_share=black, hud_bright=hud, in_scene=in_scene, failed=failed, ok=not failed, busy_machine=busy,
             timings_reliable=not busy, machine=m)
    (out / 'result.json').write_text(json.dumps(r, indent=2) + '\n', encoding='utf-8')
    return r


# ---------------------------------------------------------------- command line

def _pct(v):
    return '-' if v is None else f'{v * 100:.1f}%'


def _rel(p):
    """p relative to the repository when it is inside it, else as it is."""
    p = Path(p)
    return p.relative_to(ROOT) if p.is_relative_to(ROOT) else p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scenario', nargs='?')
    ap.add_argument('--list', action='store_true', help='list the scenarios and whether their save is present')
    ap.add_argument('--build', type=Path, default=ROOT / 'build' / 'dkb100', help='build folder with dkb.exe')
    ap.add_argument('--repeat', type=int, default=1, help='runs per variant (default 1)')
    ap.add_argument('--ab', metavar='VARIANTS', help='the scenario\'s [variants.<name>] table to alternate')
    ap.add_argument('--env', action='append', default=[], metavar='KEY=VALUE', help='extra environment for every run')
    ap.add_argument('--seconds', type=float, help="override the scenario's game time")
    ap.add_argument('--keep-profile', action='store_true', help='keep the run\'s profile copy (for a save to inspect)')
    ap.add_argument('--force', action='store_true', help='run even while another game process is running')
    ap.add_argument('--sweep', metavar='REGEX', help='run once per stage whose name matches, e.g. LevelStage$ for the '
                    '20 layers (combines with --ab and --repeat)')
    ap.add_argument('--warp', metavar='STAGE', help="start each run in this stage instead of the save's "
                    '(tools/save_warp.py; `python tools/save_warp.py --stages` lists them)')
    a = ap.parse_args(argv)

    if a.list:
        for p in sorted(SCENARIOS.glob('*.toml')):
            s = load(p.stem)
            state = 'save present' if has_save(s) else f'missing save build/saves/{s["save"]}/'
            print(f'{p.stem:22} {state:40} {s["description"]}')
            for v, envs in s['variants'].items():
                print(f'{"":22}   variants {v}: ' + ', '.join(f'{k}: {e}' for k, e in envs.items()))
        return 0
    if not a.scenario:
        ap.error('name a scenario (see --list)')
    s = load(a.scenario)
    if not has_save(s):
        print(f'scenario {s["name"]}: skipped, no private save at {_rel(save_dir(s))}/ (account.uid and '
              f'save/ of a profile; tests/scenarios/README.md says how to make one)')
        return SKIP
    exe = next((a.build / n for n in GAME_EXES if (a.build / n).is_file()), None)
    if exe is None:
        print(f'scenario: no dkb.exe in {a.build} (build it with tools/build_game.py)')
        return 2
    extra = {}
    for item in a.env:
        k, sep, v = item.partition('=')
        if not sep:
            ap.error(f'--env needs KEY=VALUE, got {item!r}')
        extra[k] = v
    if a.ab:
        table = s['variants'].get(a.ab)
        if not table:
            ap.error(f'scenario {s["name"]} has no variants "{a.ab}" (has: {", ".join(s["variants"]) or "none"})')
        variants = [(name, {**extra, **{k: str(v) for k, v in env.items()}}, None) for name, env in table.items()]
    else:
        variants = [('run', extra, None)]
    warp = a.warp or s.get('warp')
    known = save_warp.stages() if warp or a.sweep else None
    if warp and known is not None and warp not in known:
        close = [k for k in known if warp.lower() in k.lower()][:8]
        print(f'scenario: no stage "{warp}" in the RomFS' + (f' (did you mean {", ".join(close)}?)' if close else ''))
        return 2
    if a.sweep:   # one variant per matching stage, each with every --ab variant
        if known is None:
            print('scenario: --sweep needs the RomFS index (build/extracted/dkb100/romfs/index.json) for the stage list')
            return 2
        try:
            pick = [k for k in known if re.search(a.sweep, k)]
        except re.error as e:
            ap.error(f'--sweep: {e}')
        if not pick:
            print(f'scenario: no stage matches {a.sweep!r} (python tools/save_warp.py --stages lists them)')
            return 2
        variants = [(stage if name == 'run' else f'{stage}-{name}', env, stage)
                    for stage in pick for name, env, _ in variants]
    m = machine_state()
    if m['games'] and not a.force:
        print('scenario: another game process is running (' + ', '.join(f'{g["name"]} {g["id"]}' for g in m['games'])
              + '); close it or pass --force')
        return 2
    busy = busy_reasons(m)
    if busy:
        print('scenario: busy machine, timings will be marked unreliable: ' + '; '.join(busy))

    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    base = RUNS / f'{stamp}-{s["name"]}'
    print(f'scenario {s["name"]}: {s["description"]}\n  {len(variants) * a.repeat} runs of {a.seconds or s["seconds"]} s '
          f'game time with {_rel(exe)}; results in '
          f'{_rel(base)}')
    results = []
    width = max(12, max(len(v[0]) for v in variants) + 2)
    for n in range(1, a.repeat + 1):
        for name, env, stage in variants:
            r = run_once(s, exe, base / f'{name}-{n}', env, a, stage)
            results.append(r)
            causes = f' ({", ".join(r["major_episode_causes"])})' if r['major_episode_causes'] else ''
            fps = '-' if r['fps'] is None else f'{r["fps"]:.1f}'
            print(f'  {r["variant"]:{width}} {"PASS" if r["ok"] else "FAIL":4}  {r["ending"]:18} perf {r["perf"]:7}{causes:16}'
                  f' {fps:>5} fps  near-black {_pct(r["black_share"]):>6}  {r["wall_s"]:.0f} s'
                  + (f'  [{r["ending_detail"]}]' if r['ending_detail'] else '')
                  + ('' if r['timings_reliable'] else '  [busy machine]'))
            if r['failed']:
                print(f'    failed: {", ".join(r["failed"])}')
    summary = dict(scenario=s['name'], build=str(a.build), runs=results,
                   ok=all(r['ok'] for r in results), timings_reliable=all(r['timings_reliable'] for r in results))
    (base / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if len(variants) > 1:
        print(f'\n  {"variant":{width - 2}} runs  pass  median fps  median near-black')
        for name, _, _ in variants:
            rs = [r for r in results if r['variant'].rsplit('-', 1)[0] == name]
            fps = sorted(r['fps'] for r in rs if r['fps'] is not None)
            blk = sorted(r['black_share'] for r in rs if r['black_share'] is not None)
            print(f'  {name:{width - 2}} {len(rs):4} {sum(r["ok"] for r in rs):5} '
                  f'{fps[len(fps) // 2] if fps else float("nan"):11.1f} {_pct(blk[len(blk) // 2] if blk else None):>18}')
    if not summary['timings_reliable']:
        print('  timings unreliable: other programs were using the machine (result.json "busy_machine")')
    print(f'{"PASS" if summary["ok"] else "FAIL"}: {sum(r["ok"] for r in results)} of {len(results)} runs passed')
    return 0 if summary['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
