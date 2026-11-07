"""Exercise input conversion and the recording launch without game data."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import input_capture as capture


class InputCapture(unittest.TestCase):
    def test_preserves_overlapping_and_out_of_order_events(self):
        recording = '# recording\n@100 X 150\n@50 LSTICK -32767 32767 200\n@100 A 15\n'
        with tempfile.TemporaryDirectory(dir=capture.BUILD) as folder:
            source, target = Path(folder) / 'in.txt', Path(folder) / 'out.txt'
            source.write_text(recording, encoding='utf-8')
            self.assertEqual(capture.convert(source, target), (3, {'world'}))
            self.assertTrue(target.read_text().endswith(recording))
            with self.assertRaises(FileExistsError):
                capture.convert(source, target)
            self.assertTrue(target.read_text().endswith(recording))

    def test_invalid_input_and_empty_recording(self):
        for text in ('# no events', '@nan X 5', '@10 X 0', '@10 Z 5', '@1 LSTICK 1 32768 5',
                     '@1 LSTICK 1 2 3 4', '@1 A -5', '-1 A 1'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                capture.validate(text)

    def test_mixed_clocks_and_default_holds(self):
        self.assertEqual(capture.validate('1 A\n@1 RSTICK 0 -1\n'), (2, {'ticks', 'world'}))

    def test_private_paths_only(self):
        with self.assertRaises(ValueError):
            capture.build_path(ROOT / 'tools' / 'outside.txt')

    def test_record_uses_copied_save_and_enables_real_input(self):
        with tempfile.TemporaryDirectory(dir=capture.BUILD) as folder:
            root = Path(folder)
            save = root / 'save-source'
            (save / 'save').mkdir(parents=True)
            (save / 'account.uid').write_text('fake')
            build = root / 'game'
            build.mkdir()
            (build / capture.scenario.GAME_EXES[0]).touch()
            s = {'name': 'fixture', 'env': {'TOTK_INPUT': 'none', 'DKB_NATIVE_GS': '1'}}
            def run(cmd, **kw):
                env = kw['env']
                self.assertNotIn('TOTK_INPUT', env)
                self.assertNotIn('TOTK_INPUT_SCRIPT', env)
                self.assertEqual(env['DKB_NATIVE_GS'], '1')
                profile = Path(cmd[cmd.index('--profile') + 1])
                self.assertNotEqual(profile, save)
                self.assertTrue((profile / 'account.uid').exists())
                Path(env['TOTK_INPUT_RECORD']).write_text('@10 X 50\n')
                return type('Result', (), {'returncode': 0})()
            args = type('Args', (), {'scenario': 'fixture', 'build': build, 'warp': None})()
            with patch.object(capture, 'CAPTURES', root / 'captures'), \
                 patch.object(capture.scenario, 'load', return_value=s), \
                 patch.object(capture.scenario, 'has_save', return_value=True), \
                 patch.object(capture.scenario, 'save_dir', return_value=save), \
                 patch.object(capture.scenario, 'machine_state', return_value={'games': []}), \
                 patch.object(capture.subprocess, 'run', side_effect=run), \
                 patch.dict(capture.os.environ, {'TOTK_INPUT': 'none', 'TOTK_INPUT_SCRIPT': 'stale'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(capture.record(args), 0)
            self.assertEqual(len(list((root / 'captures').glob('*/replay.txt'))), 1)


if __name__ == '__main__':
    unittest.main()
