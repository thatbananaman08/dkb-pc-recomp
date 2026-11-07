"""Exercise the runtime JSONL writer, platform append, concurrent events and an actual host exception."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from shader_failure_replay_test import function
CC = None
OUT = None
DEPS = None


class EventLog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True, exist_ok=True)
        shader = (ROOT / 'runtime' / 'gfx_shader.c').read_text(encoding='utf-8')
        (OUT / 'event_failure.inc').write_text(function(shader, 'fail'), encoding='utf-8')
        dispatch = (ROOT / 'runtime' / 'dispatch.c').read_text(encoding='utf-8')
        (OUT / 'event_fatal.inc').write_text(function(dispatch, 'rt_fatal'), encoding='utf-8')
        device = (ROOT / 'runtime' / 'vk_device.c').read_text(encoding='utf-8')
        (OUT / 'event_device.inc').write_text(function(device, 'fault_address_type') + '\n' +
                                               function(device, 'vk_device_fault_report'), encoding='utf-8')
        cls.exe = OUT / ('event_fixture.exe' if os.name == 'nt' else 'event_fixture')
        platform = ROOT / 'runtime' / 'platform' / ('win32.c' if os.name == 'nt' else 'linux.c')
        libs = ['-lpsapi', '-lbcrypt'] if os.name == 'nt' else ['-lSDL3', '-ldl', '-lpthread', '-rdynamic']
        subprocess.run([CC, '-O2', '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections',
                        '-I', str(ROOT / 'runtime'), '-I', str(DEPS / 'include'), '-I', str(DEPS / 'include' / 'vma'),
                        '-I', str(OUT), str(ROOT / 'tests' / 'event_log_fixture.c'),
                        str(ROOT / 'runtime' / 'event_log.c'), str(platform), *libs, '-o', str(cls.exe)],
                       check=True, timeout=60)

    def invoke(self, path, mode):
        return subprocess.run([str(self.exe), str(path), mode], capture_output=True, text=True, timeout=60)

    def test_json_fields_escaping_and_truncation(self):
        with tempfile.TemporaryDirectory(dir=OUT) as d:
            path = Path(d) / 'events.jsonl'
            run = self.invoke(path, 'events')
            self.assertEqual(run.returncode, 0, run.stderr)
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['category'] for r in records],
                             ['shader_translation', 'shader_translation', 'device_lost', 'crash', 'crash',
                              'shader_translation'])
            self.assertTrue(all(r['version'] == 1 and r['level'] == 'error' and r['time_unix_ms'] > 0 for r in records))
            self.assertEqual(records[0]['reason'], 'quote " slash \\ newline\n utf8 é byte ÿ')
            self.assertEqual(records[0]['shader'], 'abc123')
            self.assertIsNone(records[1]['reason'])
            self.assertIsNone(records[1]['shader'])
            self.assertEqual(records[2]['result'], -4)
            self.assertEqual(records[3]['code'], 0xc0000005)
            self.assertEqual(records[3]['host_address'], '0x0000000000001234')
            self.assertEqual(records[3]['kind'], 'reported')
            self.assertEqual(records[3]['module_offset'], '0x0000000000000034')
            self.assertEqual(records[3]['access'], 1)
            self.assertEqual(records[3]['fault_address'], '0x0000000000000028')
            self.assertEqual(records[3]['module'], 'test.dll')
            self.assertEqual(records[3]['minidump'], 'local.dmp')
            self.assertEqual(len(records[4]['message']), 384)
            self.assertTrue(records[4]['truncated'])
            self.assertEqual(records[5]['reason'], '\x01' * 384)
            self.assertEqual(records[5]['shader'], '\x01' * 128)
            self.assertTrue(records[5]['truncated'])

    def test_existing_file_is_not_replaced(self):
        with tempfile.TemporaryDirectory(dir=OUT) as d:
            path = Path(d) / 'events.jsonl'
            path.write_bytes(b'keep this file\n')
            self.assertEqual(self.invoke(path, 'events').returncode, 3)
            self.assertEqual(path.read_bytes(), b'keep this file\n')

    def test_concurrent_records_are_complete(self):
        with tempfile.TemporaryDirectory(dir=OUT) as d:
            path = Path(d) / 'events.jsonl'
            run = self.invoke(path, 'threads')
            self.assertEqual(run.returncode, 0, run.stderr)
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(records), 1600)
            self.assertEqual({r['reason'] for r in records}, {f'{thread}:{i}' for thread in range(8) for i in range(200)})

    def test_real_exception_survives_immediate_exit(self):
        with tempfile.TemporaryDirectory(dir=OUT) as d:
            path = Path(d) / 'events.jsonl'
            run = self.invoke(path, 'crash')
            self.assertEqual(run.returncode, 71, run.stderr)
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual((records[0]['category'], records[0]['kind']), ('crash', 'reported'))
            self.assertEqual(records[0]['fault_address'], '0x0000000000000000')

    def test_fatal_exit_writes_the_formatted_message(self):
        with tempfile.TemporaryDirectory(dir=OUT) as d:
            path = Path(d) / 'events.jsonl'
            run = self.invoke(path, 'fatal')
            self.assertEqual(run.returncode, 70, run.stderr)
            self.assertIn('FATAL: fixture fatal value 42\n', run.stderr)
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual((records[0]['category'], records[0]['kind']), ('crash', 'fatal'))
            self.assertEqual(records[0]['message'], 'fixture fatal value 42')

    @unittest.skipUnless(os.name == 'nt', 'the Windows crash hook reports C runtime aborts')
    def test_c_runtime_abort_is_a_reported_crash(self):
        with tempfile.TemporaryDirectory(dir=OUT) as d:
            path = Path(d) / 'events.jsonl'
            run = self.invoke(path, 'abort')
            self.assertEqual(run.returncode, 71, run.stderr)
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual((records[0]['category'], records[0]['kind'], records[0]['message']),
                             ('crash', 'reported', 'abort'))
            self.assertEqual(records[0]['access'], -1)
            self.assertIsNone(records[0]['fault_address'])

    def test_device_loss_is_reported_once_without_a_gpu(self):
        with tempfile.TemporaryDirectory(dir=OUT) as d:
            path = Path(d) / 'events.jsonl'
            run = self.invoke(path, 'device')
            self.assertEqual(run.returncode, 0, run.stderr)
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual((records[0]['category'], records[0]['kind'], records[0]['result']),
                             ('device_lost', 'vulkan', -4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cc', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--deps', type=Path, required=True)
    args, rest = ap.parse_known_args()
    CC, OUT, DEPS = args.cc, args.out.resolve(), args.deps.resolve()
    sys.argv = [sys.argv[0], *rest]
    unittest.main()
