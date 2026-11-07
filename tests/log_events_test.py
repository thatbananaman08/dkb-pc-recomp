"""Tests for the offline diagnostic log adapter."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import log_events


class LogEvents(unittest.TestCase):
    def test_selected_events_keep_line_and_severity(self):
        lines = ['[host] ordinary message\n', '[host] vk validation warning: warning detail\n',
                 '[host] vk validation error: error detail\n', '[host] gfx_shader: translation failed (abc): bad\n',
                 '[host] vk: device lost; no fault report\n', 'FATAL: crash\n']
        found = list(log_events.events(lines))
        self.assertEqual([e['line'] for e in found], [2, 3, 4, 5, 6])
        self.assertEqual([(e['category'], e['level']) for e in found],
                         [('vulkan_validation', 'warning'), ('vulkan_validation', 'error'),
                          ('shader_translation', 'error'), ('device_lost', 'error'), ('crash', 'error')])
        self.assertEqual(found[0]['message'], lines[1].rstrip())

    def test_unrelated_text_is_not_promoted_to_an_event(self):
        self.assertEqual(list(log_events.events(['validation errors: 0\n', 'shader translation succeeded\n'])), [])


if __name__ == '__main__':
    unittest.main()
