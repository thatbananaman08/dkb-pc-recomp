"""Build and execute the non-finite fragment fixture on Vulkan with validation.

The fixture is independent of game content. It requires Vulkan 1.3, a Khronos
validation layer, and the named GPU. Artifacts and logs are written under --out.
"""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', required=True)
    ap.add_argument('--glslc', required=True)
    ap.add_argument('--spirv-val', required=True)
    ap.add_argument('--layer-path', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    exe = out / ('fixture.exe' if os.name == 'nt' else 'fixture')
    subprocess.run([a.cc, '-std=c11', '-O2', '-Wall', '-Wextra', str(ROOT / 'tests/fragment_nonfinite_gpu.c'),
                    '-o', str(exe), '-lvulkan-1' if os.name == 'nt' else '-lvulkan'], check=True)
    source = ROOT / 'tests/fixtures'
    def compile_shader(name, filename, options=()):
        target = out / (name + '.spv')
        subprocess.run([a.glslc, '--target-env=vulkan1.3', *options, str(source / filename), '-o', str(target)], check=True)
        subprocess.run([a.spirv_val, '--target-env', 'vulkan1.3', str(target)], check=True)
        return target
    vs = compile_shader('vertex', 'fragment_nonfinite.vert')
    plain = compile_shader('plain', 'fragment_nonfinite.frag')
    env = os.environ.copy()
    env['VK_LAYER_PATH'] = str(Path(a.layer_path).resolve())
    for slot in (0, 1):
        first = compile_shader(f'report-{slot}', 'fragment_nonfinite.frag',
                               ('-DDIAGNOSTIC', f'-DREPORT_SET={slot}'))
        second = compile_shader(f'report-{slot}-second', 'fragment_nonfinite.frag',
                                ('-DDIAGNOSTIC', '-DSECOND_SHADER', f'-DREPORT_SET={slot}'))
        if first.read_bytes() == second.read_bytes():
            raise RuntimeError('fixture shader identities unexpectedly match')
        cmd = [str(exe), str(vs), str(plain), str(first), str(second), str(slot)]
        run = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        (out / f'layout-{slot}.log').write_text(run.stdout + run.stderr, encoding='utf-8')
        print(run.stdout, end='')
        if run.stderr:
            print(run.stderr, end='')
        if run.returncode:
            raise SystemExit(run.returncode)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
