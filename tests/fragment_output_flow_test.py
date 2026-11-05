import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('flow', ROOT / 'tools/fragment_output_flow.py')
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)

class FlowTest(unittest.TestCase):
    def test_conditional_store_is_not_definite(self):
        source = '''
 OpEntryPoint Fragment %main "main" %out
 OpDecorate %out Location 0
 %out = OpVariable %ptr Output
 %zero = OpConstant %int 0
 %one = OpConstant %int 1
 %main = OpFunction %void None %fn
 %start = OpLabel
 %x = OpAccessChain %outfloat %out %zero
 OpStore %x %value
 OpBranchConditional %cond %yes %no
 %yes = OpLabel
 %y = OpAccessChain %outfloat %out %one
 OpStore %y %value
 OpBranch %join
 %no = OpLabel
 OpBranch %join
 %join = OpLabel
 OpReturn
 OpFunctionEnd
'''
        result = flow.analyze(source)
        self.assertEqual(result['definite_components'], [[0, 0]])
        self.assertEqual(result['possibly_written_components'], [[0, 0], [0, 1]])
    def test_dynamic_component_rejected(self):
        source = '''
 OpEntryPoint Fragment %main "main" %out
 OpDecorate %out Location 0
 %out = OpVariable %ptr Output
 %main = OpFunction %void None %fn
 %start = OpLabel
 %x = OpAccessChain %outfloat %out %dynamic
 OpStore %x %value
 OpReturn
 OpFunctionEnd
'''
        self.assertFalse(flow.analyze(source)['analysis_complete_for_listed_patterns'])

if __name__ == '__main__':
    unittest.main()
