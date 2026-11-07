"""Game-free checks for corpus state construction and coverage reporting."""
import random
import struct
import sys
import unittest
from pathlib import Path

import capstone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests' / 'diff'))
import corpus
import corpus_coverage
from common import BUF, BUF_SIZE, Emu


class CorpusHelpers(unittest.TestCase):
    def test_same_base_and_index_stays_in_the_scratch_buffer(self):
        words = [0x3c71fa36, 0xf87c6b83, 0x38f8fb03, 0xb82c798b, 0xfc75dab3, 0x3cfedbd1, 0xb8735a62]
        code = b''.join(struct.pack('<I', w) for w in words)
        emu = Emu([(corpus.PC, code)])
        decoder = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
        decoder.detail = True
        rng = random.Random(1)
        for i, word in enumerate(words):
            with self.subTest(word=hex(word)):
                emu.mu.mem_write(BUF, bytes(BUF_SIZE))
                pc = corpus.PC + 4 * i
                emu.run(pc, *corpus.make_safe_case(rng, word, pc, decoder))

    def test_mnemonic_coverage_does_not_claim_operand_form_coverage(self):
        shapes = {'add x, x, x': {'class': 'a57'}, 'add w, w, w': {'class': 'lse'},
                  'ldapur x, [x]': {'class': 'lse'}}
        counts = {'add': 2, 'ldapur': 3, 'sdot': 4}
        self.assertEqual(corpus_coverage.classify_counts(counts, shapes),
                         {'add': {'occurrences': 2, 'corpus': 'a57'},
                          'ldapur': {'occurrences': 3, 'corpus': 'non-a57'},
                          'sdot': {'occurrences': 4, 'corpus': 'absent'}})


if __name__ == '__main__':
    unittest.main()
