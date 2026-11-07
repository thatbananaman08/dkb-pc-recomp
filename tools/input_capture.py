"""Record a playable scenario session, or validate a recording into a replay script.

    python tools/input_capture.py record tutorial-cave --build build/dkb100-drawid-before --warp UnderLevelStage
    python tools/input_capture.py convert build/input-captures/SESSION/recording.txt --output build/input-captures/replay.txt

Record uses a copy of the scenario save. Play normally, then close the game. Recordings, scripts, profiles and logs
stay under build/. Replaying depends on the original save, timing and game state; it is not a state-based macro.
"""
import argparse
import datetime
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import scenario
import save_warp

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / 'build'
CAPTURES = BUILD / 'input-captures'
BUTTONS = frozenset('A B X Y LCLICK RCLICK L R ZL ZR PLUS MINUS LEFT UP RIGHT DOWN'.split())
STICKS = frozenset(('LSTICK', 'RSTICK'))


def build_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(BUILD.resolve()):
        raise ValueError('recordings and output must stay under build/')
    return path


def number(value, where, *, positive=False):
    try:
        n = float(value)
    except ValueError as e:
        raise ValueError(f'{where}: invalid number {value!r}') from e
    if not math.isfinite(n) or (n <= 0 if positive else n < 0):
        raise ValueError(f'{where}: number must be finite and {"positive" if positive else "nonnegative"}')
    return n


def validate(text):
    """Validate the runtime's recording format without reordering overlapping input events."""
    events, clocks = 0, set()
    for line_no, line in enumerate(text.splitlines(), 1):
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        where = f'line {line_no}'
        if len(fields) < 2:
            raise ValueError(f'{where}: expected time and input')
        timed = fields[0].startswith('@')
        number(fields[0][1:] if timed else fields[0], where)
        clocks.add('world' if timed else 'ticks')
        name = fields[1]
        if name in BUTTONS:
            if len(fields) not in (2, 3):
                raise ValueError(f'{where}: button needs an optional hold')
            if len(fields) == 3:
                number(fields[2], where, positive=True)
        elif name in STICKS:
            if len(fields) not in (4, 5):
                raise ValueError(f'{where}: stick needs x, y and an optional hold')
            for axis in fields[2:4]:
                if not re.fullmatch(r'[+-]?\d+', axis) or not -32767 <= int(axis) <= 32767:
                    raise ValueError(f'{where}: stick axes must be integers from -32767 to 32767')
            if len(fields) == 5:
                number(fields[4], where, positive=True)
        else:
            raise ValueError(f'{where}: unknown input {name!r}')
        events += 1
    if not events:
        raise ValueError('recording contains no input events')
    return events, clocks


def convert(source, output):
    source, output = build_path(source), build_path(output)
    text = source.read_text(encoding='utf-8-sig')
    events, clocks = validate(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects previous captures, including when source and output are the same file.
    with output.open('x', encoding='utf-8', newline='\n') as f:
        f.write('# Validated recording; event order and timing preserved.\n')
        f.write(text.rstrip() + '\n')
    return events, clocks


def print_replay(name, build, script, warp):
    args = [sys.executable, str(ROOT / 'tools/scenario.py'), name, '--build', str(build),
            '--env', 'TOTK_INPUT=none', '--env', f'TOTK_INPUT_SCRIPT={script}']
    if warp:
        args += ['--warp', warp]
    print('Replay with:', subprocess.list2cmdline(args) if os.name == 'nt' else shlex.join(args))
    print('Inspect the result and final frame; the scenario verdict may not establish gameplay.')


def record(args):
    s = scenario.load(args.scenario)
    if not scenario.has_save(s):
        raise ValueError(f'no scenario save at {scenario.save_dir(s)}')
    build = args.build.resolve()
    exe = next((build / n for n in scenario.GAME_EXES if (build / n).is_file()), None)
    if exe is None:
        raise ValueError(f'no game executable in {build}')
    running = scenario.machine_state()['games']
    if running:
        raise ValueError('another game process is running; close it before recording')
    warp = args.warp or s.get('warp')
    if warp:
        known = save_warp.stages()
        if known is not None and warp not in known:
            raise ValueError(f'unknown stage {warp!r}')
    CAPTURES.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix=datetime.datetime.now().strftime('%Y%m%d-%H%M%S-'), dir=CAPTURES)).resolve()
    profile = session / 'profile'
    shutil.copytree(scenario.save_dir(s), profile)
    if warp:
        save_warp.warp_profile(profile, warp)
    raw = session / 'recording.txt'
    script = session / 'replay.txt'
    env = dict(os.environ)
    env.update({k: str(v).replace('{run}', str(session)) for k, v in s['env'].items()})
    for key in ('TOTK_INPUT', 'TOTK_INPUT_SCRIPT', 'TOTK_BACKGROUND_WINDOW', 'TOTK_ASSUME_FOCUS'):
        env.pop(key, None)
    env['TOTK_INPUT_RECORD'] = str(raw)
    print(f'Capture folder: {session}\nPlay in the game window, then close the game to finish.', flush=True)
    with (session / 'game.log').open('w', encoding='utf-8', errors='replace') as log:
        result = subprocess.run([str(exe), '--profile', str(profile)], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if not raw.is_file():
        raise ValueError(f'no recording produced (game exit {result.returncode}); see {session / "game.log"}')
    events, clocks = convert(raw, script)
    print(f'Saved {events} events to {script} (game exit {result.returncode}).')
    if len(clocks) > 1:
        print('Warning: mixed tick and world times stop tick events when the world clock begins.')
    if result.returncode:
        print('The game exited with an error; inspect game.log and check for incomplete held inputs.')
    print_replay(s['name'], build, script, warp)
    return 0 if result.returncode == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    rec = sub.add_parser('record', help='play a copy of a scenario save and record input')
    rec.add_argument('scenario')
    rec.add_argument('--build', type=Path, default=BUILD / 'dkb100')
    rec.add_argument('--warp', help='start the copied save in this stage')
    conv = sub.add_parser('convert', help='validate an existing recording into a replay script')
    conv.add_argument('recording', type=Path)
    conv.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == 'record':
            return record(args)
        events, clocks = convert(args.recording, args.output)
        print(f'Saved {events} events to {args.output}.')
        if len(clocks) > 1:
            print('Warning: mixed tick and world times stop tick events when the world clock begins.')
        return 0
    except (OSError, ValueError) as e:
        print(f'input_capture: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
