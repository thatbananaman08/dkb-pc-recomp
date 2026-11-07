"""Compare shader groups in two DKB_DUMP_ORDERED draws.txt files.

Groups use shader hashes and operation type, not pipeline handles or draw numbers. A group can contain multiple
draws, so equal groups do not prove that individual draws or their bindings match. Inputs and output can contain
game-derived shader identifiers; keep them under build/ and out of public reports.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

MATCH_KEY = re.compile(r'^v1:([0-9a-f]{16}):([0-9a-f]{16}):([0-9a-f]{16}):(\d+):([1-9]\d*)$')


def read_trace(path):
    draws = []
    truncated = False
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.startswith('truncated:'):
            truncated = True
        if not line.startswith('draw '):
            continue
        fields = dict(item.split('=', 1) for item in line.split()[1:] if '=' in item)
        if not all(fields.get(k) for k in ('vs', 'fs', 'op')):
            raise ValueError(f'{path}:{number}: missing vs, fs or op')
        draws.append(fields)
    if not draws:
        raise ValueError(f'{path}: no draws found')
    return draws, truncated


def read_groups(path):
    draws, truncated = read_trace(path)
    return Counter((d['vs'], d['fs'], d['op']) for d in draws), truncated


def by_match_key(draws, path):
    keys = {}
    for draw in draws:
        key = draw.get('match_key')
        if not key:
            raise ValueError(f'{path}: draw lacks match_key; capture with a build that supports draw IDs')
        if key == 'unavailable':
            raise ValueError(f'{path}: a draw could not be assigned a match_key')
        match = MATCH_KEY.fullmatch(key)
        if not match or (draw.get('vs'), draw.get('gs'), draw.get('fs'), draw.get('op')) != match.groups()[:4]:
            raise ValueError(f'{path}: invalid match_key or fields do not agree: {key}')
        if key in keys:
            raise ValueError(f'{path}: duplicate match_key {key}')
        keys[key] = draw
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('before', type=Path)
    ap.add_argument('after', type=Path)
    ap.add_argument('--ids', action='store_true', help='list candidate individual draw matches by match_key')
    args = ap.parse_args()
    try:
        before_draws, before_truncated = read_trace(args.before)
        after_draws, after_truncated = read_trace(args.after)
        before = Counter((d['vs'], d['fs'], d['op']) for d in before_draws)
        after = Counter((d['vs'], d['fs'], d['op']) for d in after_draws)
        if args.ids:
            before_ids = by_match_key(before_draws, args.before)
            after_ids = by_match_key(after_draws, args.after)
    except (OSError, ValueError) as exc:
        ap.exit(2, f'compare_draws: {exc}\n')
    if before_truncated or after_truncated:
        print('WARNING: at least one trace was truncated; counts are incomplete')
    print('Shader groups by vertex hash, fragment hash and operation; counts are not individual draw matches.')
    print(f'{"vertex":16} {"fragment":16} {"op":>4} {"before":>8} {"after":>8} {"change":>8}')
    for vs, fs, op in sorted(before.keys() | after.keys()):
        a, b = before[(vs, fs, op)], after[(vs, fs, op)]
        if a != b:
            print(f'{vs:16} {fs:16} {op:>4} {a:8} {b:8} {b-a:+8}')
    print(f'{sum(before.values())} before, {sum(after.values())} after; '
          f'{sum(before[k] != after[k] for k in before.keys() | after.keys())} groups changed')
    if args.ids:
        print('Candidate matches only: changes to repeated draws can shift occurrences within a shader group.')
        before_groups = Counter(key.rsplit(':', 1)[0] for key in before_ids)
        after_groups = Counter(key.rsplit(':', 1)[0] for key in after_ids)
        for key in sorted(before_ids.keys() & after_ids.keys()):
            a, b = before_ids[key], after_ids[key]
            group = key.rsplit(':', 1)[0]
            confidence = 'ambiguous' if before_truncated or after_truncated or before_groups[group] != after_groups[group] else 'candidate'
            print(f'{confidence} {key} before={a.get("id", "?")}/{a.get("pass", "?")}/{a.get("ordinal", "?")} '
                  f'after={b.get("id", "?")}/{b.get("pass", "?")}/{b.get("ordinal", "?")}')
        print(f'{len(before_ids.keys() & after_ids.keys())} candidates; '
              f'{len(before_ids.keys() - after_ids.keys())} before-only, '
              f'{len(after_ids.keys() - before_ids.keys())} after-only')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
