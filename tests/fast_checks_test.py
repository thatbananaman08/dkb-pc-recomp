"""The fast-check pieces without the game: tools/test.py --changed selection, tools/scenario.py's frame-dump NaN count,
and tools/check.py's levels and stale-build check. Synthetic inputs only; no game files.
Run: python tests/fast_checks_test.py
"""
import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / 'tools'))
    spec.loader.exec_module(mod)
    return mod


T, SC, CK = load('test'), load('scenario'), load('check')


def test(name, area, cmd, **kw):
    return dict(name=name, area=area, cmd=cmd, tier=kw.pop('tier', 1), **kw)


TESTS = [
    test('lifter', 'cpu', ['{python}', 'tests/diff/simd.py']),
    test('mixer', 'audio', ['{python}', 'tests/audio_splitter_order_test.py', '--cc', '{cc}']),
    test('xlate', 'shaders', ['{python}', 'tests/float_minmax_test.py', '--translator', '{xlate}']),
    test('extra', 'tools', ['{python}', 'tests/something_test.py'], paths=['data/special/*']),
]


class Changed(unittest.TestCase):
    def pick(self, *files):
        sel, why = T.select_changed(TESTS, list(files))
        return sorted(t['name'] for t in sel), why

    def test_area_paths(self):
        self.assertEqual(self.pick('recomp/lifter.py')[0], ['lifter'])
        self.assertEqual(self.pick('runtime/cpu.h')[0], ['lifter'])
        self.assertEqual(self.pick('tools/xlate/patches/0019-x.patch')[0], ['xlate'])

    def test_own_script_and_fixture(self):
        self.assertEqual(self.pick('tests/audio_splitter_order.c')[0], ['mixer'])   # tests/<stem>* beside the script
        self.assertEqual(self.pick('tests/float_minmax_test.py')[0], ['xlate'])

    def test_manifest_paths(self):
        self.assertEqual(self.pick('data/special/table.bin')[0], ['extra'])

    def test_docs_select_nothing(self):
        names, why = self.pick('DEBUGGING.md', 'docs/building.md', '.github/workflows/tests.yml')
        self.assertEqual(names, [])
        self.assertIn('only documents', why)

    def test_runner_or_unmapped_runs_everything(self):
        self.assertEqual(len(self.pick('tools/test.py')[0]), len(TESTS))
        names, why = self.pick('weird/thing.bin')
        self.assertEqual(len(names), len(TESTS))
        self.assertIn('weird/thing.bin', why)

    def test_every_manifest_area_is_mapped(self):
        for t in T.load_manifest():
            with self.subTest(t['name']):
                self.assertIn(t['area'], T.AREA_PATHS)


class DumpNan(unittest.TestCase):
    def test_sums_every_images_file(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'a').mkdir()
            (d / 'images.txt').write_text('img1 2880x1620x1 min 0 max 1 distinct 5 top-quarter 1 nan 7 inf 0\n'
                                          'img2 1440x810x1 min 0 max 1 distinct 5 top-quarter 1 nan 3 inf 2\n')
            (d / 'a' / 'images.txt').write_text('img3 10x10x1 min 0 max 1 distinct 1 top-quarter 1 nan 0 inf 0\n')
            self.assertEqual(SC.dump_nan(d), 10)
            self.assertIsNone(SC.dump_nan(d / 'missing'))

    def test_dump_env(self):
        self.assertEqual(SC.dump_env({}), {})
        self.assertEqual(SC.dump_env({'dump_frame': 4300, 'dump_passes': '30-45'}),
                         {'TOTK_DUMP': '4300', 'TOTK_DUMP_DIR': '{run}/dump', 'DKB_DUMP_PASS': '30-45'})


class Check(unittest.TestCase):
    def test_levels(self):
        for level in ('quick', 'standard', 'full'):
            plan = CK.steps(level, Path('build/x'))
            self.assertTrue(plan)
            for name, argv, needs_build in plan:
                self.assertTrue(name)
                self.assertTrue((ROOT / argv[1]).is_file(), argv[1])
                self.assertEqual(needs_build, 'tools/scenario.py' in argv)
                if needs_build:
                    self.assertIn('build' + os.sep + 'x', ' '.join(map(str, argv)) + str(Path('build/x')))
        quick = CK.steps('quick', Path('b'))
        self.assertIn('--changed', quick[0][1])

    def test_named_tests_exist(self):
        names = {t['name'] for t in T.load_manifest()}
        for level in ('quick', 'standard', 'full'):
            for _, argv, _ in CK.steps(level, Path('b')):
                if argv[1] == 'tools/test.py':
                    for a in argv[2:]:
                        if not a.startswith('-') and not a[0].isdigit():
                            self.assertIn(a, names)

    def test_stale_build(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'runtime').mkdir()
            build = root / 'build' / 'b'
            build.mkdir(parents=True)
            self.assertTrue(CK.stale(build, root).startswith('no dkb.exe'))
            src = root / 'runtime' / 'main.c'
            src.write_text('x')
            exe = build / 'dkb.exe'
            exe.write_text('x')
            old = time.time() - 3600
            os.utime(src, (old, old))
            self.assertIsNone(CK.stale(build, root))
            os.utime(exe, (old - 3600, old - 3600))
            self.assertIn('runtime', CK.stale(build, root))


if __name__ == '__main__':
    unittest.main()
