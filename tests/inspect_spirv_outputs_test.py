import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('inspect_spirv_outputs', ROOT / 'tools/inspect_spirv_outputs.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class InspectTest(unittest.TestCase):
    def test_output_components_and_non_output_stores(self):
        asm = '''
               OpDecorate %out Location 1
        %out = OpVariable %ptr Output
        %local = OpVariable %ptr Function
        %component = OpAccessChain %floatptr %out %int_2
               OpStore %component %result
               OpStore %local %other
               OpStore %out %vector
'''
        result = mod.inspect(asm)
        self.assertEqual(result['outputs'], [{'id': '%out', 'location': 1}])
        self.assertEqual(len(result['stores']), 2)
        self.assertEqual(result['stores'][0]['indices'], ['%int_2'])
        self.assertEqual(result['stores'][1]['indices'], [])
        self.assertEqual(result['stores'][0]['location'], 1)


if __name__ == '__main__':
    unittest.main()
