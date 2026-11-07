"""Compare corpus and local module instruction mnemonics; keep the game-derived report under build/.

This is a coarse mnemonic comparison, not an instruction-shape or feature coverage measurement. A mnemonic present
in both sources may have operand forms the corpus omits. It does not prove that an instruction was executed in play.
"""
import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import capstone
import numpy as np

from common import ROOT
from corpus import DEFAULT_CORPUS, load_corpus
from nso import Nso


def classify_counts(counts, corpus_shapes):
    a57 = {shape.split()[0] for shape, rec in corpus_shapes.items() if rec['class'] == 'a57'}
    other = {shape.split()[0] for shape, rec in corpus_shapes.items() if rec['class'] != 'a57'}
    return {name: {'occurrences': count, 'corpus': 'a57' if name in a57 else
                   'non-a57' if name in other else 'absent'} for name, count in sorted(counts.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--exefs', type=Path, required=True)
    ap.add_argument('--modules', default='main,sdk')
    ap.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    ap.add_argument('--out', type=Path, default=ROOT / 'build' / 'test' / 'corpus-coverage' / 'report.json')
    args = ap.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / 'build').resolve()):
        ap.error('--out must be under build/; this report is game-derived')
    shapes = load_corpus(args.corpus)['shapes']
    decoder = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    report = {'limitation': 'Mnemonic comparison only; matching names do not establish operand-form coverage.',
              'modules': {}}
    for name in args.modules.split(','):
        if not name or Path(name).name != name:
            ap.error('module names must be filenames')
        nso = Nso(str(args.exefs / name))
        lo, size = nso.text
        words = np.frombuffer(bytes(nso.image[lo:lo + size]), dtype='<u4')
        unique, frequencies = np.unique(words, return_counts=True)
        counts = Counter()
        for word, count in zip(unique, frequencies):
            instruction = next(decoder.disasm(struct.pack('<I', int(word)), 0), None)
            counts[instruction.mnemonic if instruction else '<undecodable>'] += int(count)
        report['modules'][name] = classify_counts(counts, shapes)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote private mnemonic comparison to {out}; matching mnemonics do not establish shape coverage.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f'corpus_coverage: {exc}')
