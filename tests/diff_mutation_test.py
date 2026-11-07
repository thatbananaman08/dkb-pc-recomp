"""Check that mutation failures and broken differential runs are distinguished."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests' / 'diff'))
import mutation


def result(code, failures=0, output=None):
    if output is None:
        output = f'10 encodings, 30 cases: {failures} failing cases, {int(failures > 0)} failing encodings\n'
    return SimpleNamespace(returncode=code, stdout=output)


class MutationRunner(unittest.TestCase):
    def run_with(self, results):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(sys, 'argv', ['mutation.py', '--exefs', tmp, '--out', tmp]), \
             mock.patch.object(mutation.subprocess, 'run', side_effect=results) as run, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            status = mutation.main()
        return status, output.getvalue(), run.call_count

    def test_mismatches_are_caught(self):
        status, output, count = self.run_with([result(0)] + [result(1, 2)] * len(mutation.MUTATIONS))
        self.assertEqual(status, 0)
        self.assertEqual(count, len(mutation.MUTATIONS) + 1)
        self.assertEqual(output.count('CAUGHT'), len(mutation.MUTATIONS))

    def test_surviving_mutation_fails(self):
        status, output, _ = self.run_with([result(0), result(0)] + [result(1, 2)] * (len(mutation.MUTATIONS) - 1))
        self.assertEqual(status, 1)
        self.assertIn('SURVIVED', output)

    def test_compiler_error_does_not_count_as_detection(self):
        status, output, _ = self.run_with([result(0), result(1, output='compile failed\n')] +
                                          [result(1, 2)] * (len(mutation.MUTATIONS) - 1))
        self.assertEqual(status, 1)
        self.assertIn('ERROR', output)

    def test_harness_error_is_not_a_detection_even_with_mismatches(self):
        broken = result(1, output='harness failed: exit 1\n10 encodings, 3 cases: 1 failing cases, 1 failing encodings\n')
        status, output, _ = self.run_with([result(0), broken] + [result(1, 2)] * (len(mutation.MUTATIONS) - 1))
        self.assertEqual(status, 1)
        self.assertIn('ERROR', output)

    def test_empty_control_does_not_pass(self):
        empty = result(0, output='10 encodings, 0 cases: 0 failing cases, 0 failing encodings\n')
        status, _, count = self.run_with([empty])
        self.assertEqual(status, 1)
        self.assertEqual(count, 1)

    def test_bad_control_stops_before_mutations(self):
        status, output, count = self.run_with([result(1, 1)])
        self.assertEqual(status, 1)
        self.assertIn('ERROR control', output)
        self.assertEqual(count, 1)


if __name__ == '__main__':
    unittest.main()
