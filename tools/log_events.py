"""Extract selected diagnostic events from a runtime text log as JSON lines.

This is an offline, best-effort adapter for existing logs, not a complete structured runtime log. Text and event
details can contain game data; the output must stay under build/ and must not be published.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build'
PATTERNS = (
    ('crash', 'error', re.compile(r'\bFATAL:')),
    ('device_lost', 'error', re.compile(r'\bvk: device lost\b', re.I)),
    ('shader_translation', 'error', re.compile(r'\bgfx_shader: translation failed\b', re.I)),
    ('vulkan_validation', 'error', re.compile(r'\bvk validation error:', re.I)),
    ('vulkan_validation', 'warning', re.compile(r'\bvk validation warning:', re.I)),
)


def events(lines):
    for number, line in enumerate(lines, 1):
        for category, level, pattern in PATTERNS:
            if pattern.search(line):
                yield {'line': number, 'category': category, 'level': level, 'message': line.rstrip('\r\n')}
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('log', type=Path)
    ap.add_argument('--out', type=Path, required=True, help='JSONL output under build/')
    args = ap.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(BUILD.resolve()):
        ap.error('--out must be under build/; events can contain game data')
    if out == args.log.resolve():
        ap.error('output must differ from the input log')
    try:
        with args.log.open(encoding='utf-8', errors='replace') as src:
            found = list(events(src))
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', encoding='utf-8', newline='\n') as dst:
            for event in found:
                dst.write(json.dumps(event, ensure_ascii=False) + '\n')
    except OSError as exc:
        ap.exit(2, f'log_events: {exc}\n')
    print(f'{len(found)} selected events written to {out}; unrecognized messages are omitted')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
