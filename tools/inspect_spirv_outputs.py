"""List direct SPIR-V fragment-output stores without modifying the shader.

Pointer aliases through functions and phi nodes are not resolved; this inventory
is not proof that an output is finite at return.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

ID = r'%[A-Za-z0-9_]+'


def inspect(assembly):
    locations, outputs, pointers, stores = {}, {}, {}, []
    lines = assembly.splitlines()
    for line in lines:
        if m := re.search(rf'OpDecorate\s+({ID})\s+Location\s+(\d+)', line):
            locations[m[1]] = int(m[2])
        if m := re.search(rf'^\s*({ID})\s*=\s*OpVariable\s+{ID}\s+Output\b', line):
            outputs[m[1]] = None
    pointers.update({key: (key, []) for key in outputs})
    for number, line in enumerate(lines, 1):
        if m := re.search(rf'^\s*({ID})\s*=\s*Op(?:InBounds)?AccessChain\s+{ID}\s+({ID})(.*)', line):
            if m[2] in pointers:
                base, path = pointers[m[2]]
                pointers[m[1]] = (base, path + re.findall(ID, m[3]))
        if m := re.search(rf'\bOpStore\s+({ID})\s+({ID})', line):
            if m[1] in pointers:
                base, path = pointers[m[1]]
                stores.append({'line': number, 'output': base, 'location': locations.get(base),
                               'indices': path, 'value': m[2]})
    return {'outputs': [{'id': x, 'location': locations.get(x)} for x in outputs], 'stores': stores,
            'limitation': 'Only direct output variables and access chains are resolved; aliases and dynamic control flow need separate analysis.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shader', type=Path)
    parser.add_argument('--spirv-dis', default='spirv-dis')
    args = parser.parse_args()
    result = subprocess.run([args.spirv_dis, str(args.shader)], check=True, capture_output=True, text=True)
    print(json.dumps(inspect(result.stdout), indent=2))


if __name__ == '__main__':
    main()
