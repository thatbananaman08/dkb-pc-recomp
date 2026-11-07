"""Check that the end-to-end comparison rejects a faulty emitted program."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests' / 'diff'))
import e2e

CC = None
if '--cc' in sys.argv:
    index = sys.argv.index('--cc')
    CC = sys.argv[index + 1]
    del sys.argv[index:index + 2]


class EndToEndOracle(unittest.TestCase):
    def test_corrupting_an_argument_in_emitted_functions_is_detected(self):
        original = e2e.gen.chunk_text

        def corrupt(*args):
            generated = original(*args)
            self.assertIn('    FN_ENTER(', generated)
            return generated.replace('    FN_ENTER(', '    c->x[1] ^= 1; FN_ENTER(')

        build = ROOT / 'build' / 'test'
        build.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build) as tmp, \
             mock.patch.object(sys, 'argv', ['e2e.py', '--out', tmp] + (['--cc', CC] if CC else [])), \
             mock.patch.object(e2e.gen, 'chunk_text', side_effect=corrupt), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            status = e2e.main()
        self.assertEqual(status, 1)
        self.assertIn('failures', output.getvalue())


if __name__ == '__main__':
    unittest.main()
