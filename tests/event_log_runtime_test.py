"""Check structured fatal events in the built runtime without loading game data or using the GPU."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXE = None
OUT = None


class RuntimeEventLog(unittest.TestCase):
    def run_probe(self, path):
        env = dict(os.environ, TOTK_EVENT_LOG=str(path), TOTK_PROBE_NVN_CHILD='event-log-fixture')
        env.pop('DKB_EVENT_LOG', None)
        return subprocess.run([str(EXE), '--probe-nvn'], env=env, capture_output=True, text=True, timeout=30)

    def test_actual_fatal_and_existing_file_protection(self):
        OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as d:
            path = Path(d) / 'events.jsonl'
            run = self.run_probe(path)
            self.assertEqual(run.returncode, 70, run.stderr)
            self.assertTrue(path.is_file(), 'the runtime did not create an event log; rebuild the executable')
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual((records[0]['category'], records[0]['kind']), ('crash', 'fatal'))
            self.assertEqual(records[0]['message'], 'probe-nvn: no proc event-log-fixture')
            self.assertIn('FATAL: probe-nvn: no proc event-log-fixture', run.stderr)
            path.write_bytes(b'existing file\n')
            again = self.run_probe(path)
            self.assertEqual(again.returncode, 70, again.stderr)
            self.assertEqual(path.read_bytes(), b'existing file\n')
            self.assertIn('event log: cannot create', again.stderr)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args, rest = ap.parse_known_args()
    EXE, OUT = args.exe.resolve(), args.out.resolve()
    sys.argv = [sys.argv[0], *rest]
    unittest.main()
