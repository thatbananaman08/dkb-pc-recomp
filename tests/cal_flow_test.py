"""Regression fixture for the CAL fall-through bug: a call must return to the instruction after it.

The vendored decoder took an unpredicated CAL for a branch without fall-through (Decoder.IsUnconditionalBranch), so the
instruction after it had no predecessor. In the compiler's usual square-root expansion (dwk001, issue #45)

    @!P0 BRA fast ; CAL slow ; BRA end ; fast: <fast path> ; end: ...        slow: <slow path> ; RET

the BRA end was lost and the slow path ran into the fast path. This shader has that shape with P0 from a c[0] load
(ISETP.GT R4, 0), so the translator cannot fold the branch away; the slow path writes 3.0 to R0, the fast path 5.0.
The check is on the control flow of the emitted SPIR-V: from the slow path's write (the call, or the 3.0 itself when
inlined) no path may reach a write of 5.0. With the bug one does; fixed, none does. It also checks that both values
and a conditional branch survive, so a translator that folded the shape away cannot pass by accident.

No game data and no GPU. Usage: python tests/cal_flow_test.py --translator PATH [--spirv-val PATH] [--out DIR]
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PT = 7 << 16                      # instruction predicate field: PT (always)
NOT_P0 = (0 << 16) | (1 << 19)    # @!P0
CC_T = 0xf                        # branch condition code: always
SCHED = 0x001f8000fc0007e0        # a scheduling control word (the decoder skips it)
SLOW, FAST, ALPHA = 3.0, 5.0, 1.0


def f32(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def mov32i(dest, value):
    return 0x0100000000000000 | (f32(value) << 20) | (0xf << 12) | PT | dest


def ldc(dest, offset):            # LDC R<dest>, c[0][offset] (32-bit, from RZ)
    return 0xef90000000000000 | (4 << 48) | (offset << 20) | (255 << 8) | PT | dest


def isetp_gt_zero(pred, src):     # ISETP.GT.S32.AND P<pred>, PT, R<src>, 0, PT
    return (0x3660000000000000 | (4 << 49) | (1 << 48) | (7 << 39) | (pred << 3) | 7 | (src << 8) | PT)


def program():
    """Instruction words (scheduling words not yet inserted) and the indices the branches target."""
    ops = [ldc(4, 0), isetp_gt_zero(0, 4),
           ('bra', NOT_P0, 'fast'), ('cal', 'slow'), ('bra', PT, 'end'),
           mov32i(0, FAST),                                                    # fast:
           mov32i(1, 0.0), mov32i(2, 0.0), mov32i(3, ALPHA), 0xe300000000000000 | PT | CC_T,   # end: ... EXIT
           mov32i(0, SLOW), 0xe320000000000000 | PT | CC_T]                    # slow: R0 = 3.0; RET
    labels = {'fast': 5, 'end': 6, 'slow': 10}
    address = lambda i: 8 * (i + i // 3 + 1)   # physical byte address, with one scheduling word per three
    out = []
    for i, op in enumerate(ops):
        if isinstance(op, tuple):
            rel = (address(labels[op[-1]]) - address(i) - 8) & 0xffffff
            # CAL with predicate PT, as the Ampere lowering emits it. Decoder.IsUnconditional reads the predicate field
            # of every instruction: with it at 0 (P0) the call counts as conditional, keeps its fall-through, and the
            # bug cannot show.
            op = (0xe260000000000000 | PT | (rel << 20)) if op[0] == 'cal' else (0xe240000000000000 | op[1] | CC_T | (rel << 20))
        out.append(op)
    return out


def shader():
    ops = program()
    words = []
    for i in range(0, len(ops), 3):
        words += [SCHED, *ops[i:i + 3]]
    header = [0] * 20
    header[0] = 2 | (3 << 5) | (5 << 10)   # fragment stage
    header[18] = 0xf                       # render target 0: R0-R3
    return struct.pack('<20I', *header) + struct.pack('<' + 'Q' * len(words), *words)


# ---------------------------------------------------------------- SPIR-V control flow

OP_CONSTANT, OP_FUNCTION, OP_FUNCTION_END, OP_FUNCTION_CALL, OP_LABEL = 43, 54, 56, 57, 248
OP_BRANCH, OP_BRANCH_CONDITIONAL, OP_SWITCH = 249, 250, 251


def instructions(data):
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    assert words[0] == 0x07230203, 'not a SPIR-V module'
    out, i = [], 5
    while i < len(words):
        n, op = words[i] >> 16, words[i] & 0xffff
        assert n and i + n <= len(words), 'malformed SPIR-V'
        out.append((op, list(words[i + 1:i + n])))
        i += n
    return out


def functions(insts):
    """{function id: [(label, [instructions], [successor labels])]} and {value: [constant ids]}."""
    funcs, consts, cur, block = {}, {}, None, None
    for op, args in insts:
        if op == OP_CONSTANT and len(args) == 3:
            consts.setdefault(args[2], []).append(args[1])
        elif op == OP_FUNCTION:
            cur = args[1]
            funcs[cur] = []
        elif op == OP_FUNCTION_END:
            cur = None
        elif cur is not None and op == OP_LABEL:
            block = (args[0], [], [])
            funcs[cur].append(block)
        elif cur is not None and block is not None:
            block[1].append((op, args))
            if op == OP_BRANCH:
                block[2].append(args[0])
            elif op == OP_BRANCH_CONDITIONAL:
                block[2].extend(args[1:3])
            elif op == OP_SWITCH:
                block[2].append(args[1])
                block[2].extend(args[3::2])
    return funcs, consts


def check(spv):
    """None when the slow path never reaches the fast path's write, else what went wrong."""
    funcs, consts = functions(instructions(spv))
    slow, fast = set(consts.get(f32(SLOW), [])), set(consts.get(f32(FAST), []))
    if not slow or not fast:
        return f'the translation lost a path: constant 3.0 {"present" if slow else "missing"}, 5.0 {"present" if fast else "missing"}'
    if not any(op == OP_BRANCH_CONDITIONAL for f in funcs.values() for _, body, _ in f for op, _ in body):
        return 'no conditional branch: the P0 test was folded away, so the fixture checks nothing'

    def refs(inst, ids):
        return any(a in ids for a in inst[1])

    memo = {}

    def calls_into(fid, ids, seen=()):
        """Whether function fid (or a function it calls) uses one of ids."""
        if (fid, frozenset(ids)) in memo:
            return memo[fid, frozenset(ids)]
        hit = False
        for _, body, _ in funcs.get(fid, []):
            for inst in body:
                if refs(inst, ids) or inst[0] == OP_FUNCTION_CALL and inst[1][2] not in seen \
                        and calls_into(inst[1][2], ids, seen + (fid,)):
                    hit = True
        memo[fid, frozenset(ids)] = hit
        return hit

    def uses(inst, ids):
        return refs(inst, ids) or inst[0] == OP_FUNCTION_CALL and calls_into(inst[1][2], ids)

    for fid, blocks in funcs.items():
        by_label = {b[0]: b for b in blocks}
        for label, body, succ in blocks:
            for k, inst in enumerate(body):
                if not uses(inst, slow):
                    continue
                # Everything control can reach after the slow path's write: the rest of this block, then its successors.
                if any(uses(i, fast) for i in body[k + 1:]):
                    return f'the slow path (block %{label}) runs into the fast path in the same block'
                todo, seen = list(succ), set()
                while todo:
                    nxt = todo.pop()
                    if nxt in seen or nxt not in by_label:
                        continue
                    seen.add(nxt)
                    if any(uses(i, fast) for i in by_label[nxt][1]):
                        return (f'the slow path (block %{label}) reaches the fast path\'s write in block %{nxt}: '
                                'the BRA after the CAL was lost')
                    todo.extend(by_label[nxt][2])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--translator', required=True, type=Path)
    ap.add_argument('--spirv-val', type=Path)
    ap.add_argument('--out', type=Path, default=ROOT / 'build/test/cal-flow')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    code = a.out / 'cal-flow.bin'
    code.write_bytes(shader())
    job = {'key': 'cal-flow', 'stage': 'ps', 'code': str(code.resolve()), 'debug': False, 'fout': {'0': 'float'}}
    run = subprocess.run([str(a.translator.resolve()), '--serve'], input=json.dumps({'job': job, 'out': str(a.out)}) + '\n',
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0 and run.stdout.strip() == 'ok cal-flow', (run.stdout, run.stderr)
    spv = (a.out / 'cal-flow.spv').read_bytes()
    if a.spirv_val:
        v = subprocess.run([str(a.spirv_val), '--target-env', 'vulkan1.3', str(a.out / 'cal-flow.spv')],
                           capture_output=True, text=True)
        assert v.returncode == 0, v.stdout + v.stderr
    problem = check(spv)
    if problem:
        raise SystemExit(f'FAIL: {problem}')
    print('CAL flow: the slow path returns past the fast path (BRA after the CAL kept)')


if __name__ == '__main__':
    main()
