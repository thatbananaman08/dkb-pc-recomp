"""tools/scenario.py without the game: how a run ended, the crash excerpt, the near-black share, scenario files and the
missing-save skip. Synthetic logs and images only; no game files.
Run: python tests/scenario_test.py
"""
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('scenario', ROOT / 'tools' / 'scenario.py')
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

PERF = '[host] perf: frame 60: interval avg 16.70 ms max 20.00 ms (59.9 fps); render thread blocked: pacing 40% ...\n'


class Ending(unittest.TestCase):
    def test_completed(self):
        self.assertEqual(sc.ending(0, PERF + '[host] frametime: exit: 900 frames\n', False), ('completed', ''))

    def test_completed_without_perf_output(self):
        kind, detail = sc.ending(0, 'nothing\n', False)
        self.assertEqual(kind, 'completed')
        self.assertIn('TOTK_TRACE=perf', detail)

    def test_reported_crash(self):
        text = PERF + '[host] FATAL: access violation (0xc0000005) at host 00007FF6 (totk+0x91c9630), reading address 0x38\n'
        self.assertEqual(sc.ending(71, text, False)[0], 'crashed')
        self.assertIn('access violation', sc.ending(71, text, False)[1])

    def test_crash_without_report(self):
        kind, detail = sc.ending(0xC0000005, PERF, False)
        self.assertEqual(kind, 'crashed, no report')
        self.assertIn('0xc0000005 (access violation)', detail)
        self.assertIn('stack overflow', sc.ending(-1073741571, PERF, False)[1])   # 0xC00000FD as a signed exit code

    def test_device_lost(self):
        text = PERF + '[host] vk: device lost; fault: instruction pointer (invalid)\n'
        self.assertEqual(sc.ending(70, text, False)[0], 'device lost')

    def test_hung_and_not_started(self):
        self.assertEqual(sc.ending(None, PERF, True)[0], 'hung')
        self.assertEqual(sc.ending(None, 'scenario: could not start: [WinError 2]\n', False)[0], 'did not start')

    def test_crash_excerpt_takes_the_reporter_block(self):
        text = ('before\n[host] FATAL: access violation\n  in lifted function f_7000123 = main+0x123\n  x0  0000\n'
                '[host] after\n')
        self.assertEqual(sc.crash_excerpt(text),
                         '[host] FATAL: access violation\n  in lifted function f_7000123 = main+0x123\n  x0  0000\n')


class Image(unittest.TestCase):
    def test_black_share_below_the_hud(self):
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.skipTest('no Pillow')
        with tempfile.TemporaryDirectory() as d:
            png = Path(d) / 'f.png'
            im = PILImage.new('RGB', (100, 100), (200, 150, 100))
            for y in range(0, 20):           # the HUD band: black, left out
                for x in range(100):
                    im.putpixel((x, y), (0, 0, 0))
            for y in range(60, 100):         # 40 of the 80 rows below the band
                for x in range(100):
                    im.putpixel((x, y), (2, 2, 2))
            im.save(png)
            self.assertAlmostEqual(sc.black_share(png, 0.20), 0.5)
            self.assertIsNone(sc.black_share(Path(d) / 'missing.png', 0.2))

    def test_hud_band_tells_gameplay_from_a_loading_screen(self):
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.skipTest('no Pillow')
        with tempfile.TemporaryDirectory() as d:
            loading, game = Path(d) / 'loading.png', Path(d) / 'game.png'
            PILImage.new('RGB', (100, 100), (0, 0, 0)).save(loading)
            im = PILImage.new('RGB', (100, 100), (0, 0, 0))
            for x in range(10):              # a heart in the HUD band: 10 of its 1500 pixels bright
                im.putpixel((x, 5), (255, 60, 120))
            im.save(game)
            self.assertEqual(sc.frame_stats(loading, 0.15), (1.0, 0.0))
            black, hud = sc.frame_stats(game, 0.15)
            self.assertEqual(black, 1.0)
            self.assertAlmostEqual(hud, 10 / 1500)


class Files(unittest.TestCase):
    def test_every_scenario_parses_and_names_its_input(self):
        for p in sorted(sc.SCENARIOS.glob('*.toml')):
            with self.subTest(p.stem):
                s = sc.load(p.stem)
                self.assertTrue((ROOT / s['input']).is_file(), s['input'])
                self.assertGreater(s['seconds'], 0)
                for envs in s['variants'].values():
                    self.assertGreaterEqual(len(envs), 2)

    def test_missing_save_skips_with_77(self):
        s = sc.load('tutorial-cave')
        saved = sc.SAVES
        with tempfile.TemporaryDirectory() as d:
            sc.SAVES = Path(d)
            try:
                out = io.StringIO()
                with redirect_stdout(out):
                    code = sc.main(['tutorial-cave'])
            finally:
                sc.SAVES = saved
            self.assertEqual(code, sc.SKIP)
            self.assertIn(str(Path(d) / s['save']), out.getvalue())   # the message names the folder to create

    def test_run_env_placeholder_and_trace(self):
        s = dict(input='tools/input/dkb-title-to-save.txt', env={'TOTK_FLIGHT_DIR': '{run}/flight'})
        out = Path('build') / 'scenarios' / 'x' / 'run-1'
        env = sc.run_env(s, {'TOTK_DUMP_DIR': '{run}/dump', 'TOTK_TRACE': 'fs'}, out, {'TOTK_TRACE': 'gfx', 'KEEP': '1'})
        self.assertEqual(env['TOTK_DUMP_DIR'], f'{out}/dump')
        self.assertEqual(env['TOTK_FLIGHT_DIR'], f'{out}/flight')
        self.assertEqual(env['TOTK_TRACE'], 'fs,perf')   # --env replaces the caller's value, perf is added
        self.assertEqual(env['KEEP'], '1')
        self.assertTrue(env['TOTK_INPUT_SCRIPT'].endswith('dkb-title-to-save.txt'))

    def test_perf_is_judged_only_when_asked(self):
        self.assertFalse(sc.perf_fails({}, 'SLOW'))
        self.assertTrue(sc.perf_fails({'perf': 'SLOW'}, 'SLOW'))
        self.assertFalse(sc.perf_fails({'perf': 'SLOW'}, 'HITCHY'))
        self.assertTrue(sc.perf_fails({'perf': 'HITCHY'}, 'HITCHY'))
        self.assertFalse(sc.perf_fails({'perf': 'HITCHY'}, 'OK'))

    def test_busy_reasons(self):
        m = dict(games=[], heavy=[dict(name='godot', id=1, mib=6191)], free_commit_gib=2.5, checked=True)
        self.assertEqual(sc.busy_reasons(m), ['godot (6191 MiB)', 'only 2.5 GiB of commit memory free'])
        self.assertEqual(sc.busy_reasons(dict(games=[], heavy=[], free_commit_gib=20.0, checked=True)), [])


if __name__ == '__main__':
    unittest.main()
