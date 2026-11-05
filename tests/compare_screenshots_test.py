import importlib.util
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare_screenshots', ROOT / 'tools/compare_screenshots.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompareTest(unittest.TestCase):
    def test_cli_warns_without_matching_state(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, diff = (Path(d) / x for x in ('a.png', 'b.png', 'diff.png'))
            Image.new('RGB', (1, 1)).save(a)
            Image.new('RGB', (1, 1)).save(b)
            run = subprocess.run([sys.executable, str(ROOT / 'tools/compare_screenshots.py'), str(a), str(b),
                                  '--diff', str(diff)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0)
            self.assertIn('metadata are missing or differ', run.stderr)
            self.assertIn('"matching_state_metadata": false', run.stdout)

    def test_threshold_and_difference(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / 'a.png', Path(d) / 'b.png'
            Image.new('RGB', (2, 1), (10, 10, 10)).save(a)
            im = Image.new('RGB', (2, 1), (10, 10, 10))
            im.putpixel((1, 0), (12, 15, 10))
            im.save(b)
            diff, result = module.compare(a, b, 2)
            self.assertEqual(diff.getpixel((1, 0)), (2, 5, 0))
            self.assertEqual(result['different_pixels'], 1)
            self.assertEqual(result['different_channels'], 1)
            self.assertEqual(result['different_fraction'], .5)
            self.assertEqual(module.compare(a, b, 5)[1]['different_pixels'], 0)
            Image.new('RGB', (1, 1)).save(b)
            with self.assertRaises(ValueError):
                module.compare(a, b)


if __name__ == '__main__':
    unittest.main()
