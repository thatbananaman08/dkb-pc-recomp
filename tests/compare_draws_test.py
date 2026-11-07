"""Tests for comparing ordered draw traces without game data."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import compare_draws


class CompareDraws(unittest.TestCase):
    def test_groups_ignore_unstable_handles_and_keep_duplicate_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / 'draws.txt'
            trace.write_text('draw id=1 pass=2 op=3 pipeline=0x12 vs=abc fs=def\n'
                             '  binding draw=1 handle=0x123\n'
                             'draw id=8 pass=9 op=3 pipeline=0x99 vs=abc fs=def\n'
                             'draw id=9 op=4 vs=abc fs=def\n', encoding='utf-8')
            groups, truncated = compare_draws.read_groups(trace)
            self.assertEqual(groups[('abc', 'def', '3')], 2)
            self.assertEqual(groups[('abc', 'def', '4')], 1)
            self.assertFalse(truncated)

    def test_missing_fields_and_empty_trace_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / 'draws.txt'
            for data in ('draw vs=abc op=1\n', '  binding draw=1\n'):
                trace.write_text(data, encoding='utf-8')
                with self.assertRaises(ValueError):
                    compare_draws.read_groups(trace)

    def test_truncated_trace_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / 'draws.txt'
            trace.write_text('draw op=1 vs=a fs=b\ntruncated: draw limit reached\n', encoding='utf-8')
            self.assertTrue(compare_draws.read_groups(trace)[1])
            with mock.patch.object(sys, 'argv', ['compare_draws.py', str(trace), str(trace)]):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(compare_draws.main(), 0)
            self.assertIn('WARNING', output.getvalue())

    def test_match_keys_match_despite_changed_frame_positions(self):
        def draw(vs, occurrence, draw_id, pass_id):
            gs, fs = '0000000000000000', '000000000000000b'
            return {'vs': vs, 'gs': gs, 'fs': fs, 'op': '1', 'match_key': f'v1:{vs}:{gs}:{fs}:1:{occurrence}',
                    'id': draw_id, 'pass': pass_id}
        a, z = '000000000000000a', '000000000000000c'
        before = [draw(a, 1, '1', '2'), draw(a, 2, '2', '2')]
        after = [draw(z, 1, '1', '0'), draw(a, 1, '8', '5'), draw(a, 2, '9', '5')]
        old = compare_draws.by_match_key(before, 'before')
        new = compare_draws.by_match_key(after, 'after')
        self.assertEqual(len(old.keys() & new.keys()), 2)
        self.assertEqual(new[before[0]['match_key']]['id'], '8')

    def test_match_keys_reject_missing_and_duplicate_ids(self):
        key = 'v1:000000000000000a:0000000000000000:000000000000000b:1:1'
        valid = {'vs': '000000000000000a', 'gs': '0000000000000000', 'fs': '000000000000000b',
                 'op': '1', 'match_key': key}
        for draws in ([{'vs': 'a'}], [{'match_key': 'unavailable'}], [valid, valid],
                      [{**valid, 'op': '2'}], [{**valid, 'match_key': 'invalid'}]):
            with self.assertRaises(ValueError):
                compare_draws.by_match_key(draws, 'trace')

    def test_changed_group_size_marks_matches_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, after = Path(tmp) / 'before.txt', Path(tmp) / 'after.txt'
            prefix = ('draw id={n} pass=1 ordinal={n} op=1 vs=000000000000000a gs=0000000000000000 '
                      'fs=000000000000000b match_key=v1:000000000000000a:0000000000000000:'
                      '000000000000000b:1:{n}\n')
            before.write_text(prefix.format(n=1), encoding='utf-8')
            after.write_text(prefix.format(n=1) + prefix.format(n=2), encoding='utf-8')
            with mock.patch.object(sys, 'argv', ['compare_draws.py', str(before), str(after), '--ids']):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(compare_draws.main(), 0)
            self.assertIn('ambiguous v1:', output.getvalue())


if __name__ == '__main__':
    unittest.main()
