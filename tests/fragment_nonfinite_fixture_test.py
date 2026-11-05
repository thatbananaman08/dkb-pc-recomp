"""Compile and validate standalone fragment-output diagnostic shader fixtures.

This is a SPIR-V fixture check, not a GPU execution test or runtime instrumentation.
"""
import argparse
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests/fixtures/fragment_nonfinite.frag'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glslc', required=True)
    parser.add_argument('--spirv-val', required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        for name, defines in [('original', []), ('diagnostic-set1', ['-DDIAGNOSTIC', '-DREPORT_SET=1']),
                              ('diagnostic-set2', ['-DDIAGNOSTIC', '-DREPORT_SET=2'])]:
            out = pathlib.Path(tmp) / (name + '.spv')
            subprocess.run([args.glslc, '--target-env=vulkan1.3', '-fshader-stage=frag', *defines,
                            str(SOURCE), '-o', str(out)], check=True)
            subprocess.run([args.spirv_val, '--target-env', 'vulkan1.3', str(out)], check=True)
            print(name + ': SPIR-V valid')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
