"""Bit-accurate MAC entry at packed operands, without raw conversion/entry RTL.

Reuses independent arithmetic functions, not DUT outputs. Inactive lanes are
masked here because multiply() normally receives already-masked entry payloads.
"""
from dataclasses import dataclass
from .mac_contract import (
    AccKind, ComputeMode, ContractViolation, Entry, FloatAcc36State,
    align, multiply, mx, partial_sum, require_signed, require_unsigned,
    saturate_int32, signed, update_float_acc36,
)


@dataclass(frozen=True)
class PackedStep:
    sequence: int
    step: int
    mode: ComputeMode
    kind: AccKind
    group_valid: int
    acc_first: int
    phase: int
    bias_sum: int
    mx_enable: bool
    scale_a: tuple[int, int]
    scale_b: tuple[int, int]
    tag: tuple[int, int]
    operand_a: tuple[int, ...]
    operand_b: tuple[int, ...]

    def entry(self) -> Entry:
        mode, kind = ComputeMode(self.mode), AccKind(self.kind)
        wide = mode in (ComputeMode.FP19, ComputeMode.INT16)
        if self.group_valid not in ((1,) if wide else (1, 2, 3)):
            raise ContractViolation('invalid group mask')
        if self.phase not in (0, 1) or (mode != ComputeMode.INT16 and self.phase):
            raise ContractViolation('invalid phase')
        if self.mx_enable and mode not in (ComputeMode.FP9, ComputeMode.INT8):
            raise ContractViolation('invalid MX enable')
        expected_kind = int(mode in (ComputeMode.FP19, ComputeMode.FP9) or self.mx_enable)
        if kind != expected_kind:
            raise ContractViolation('kind/mode mismatch')
        require_unsigned(self.acc_first, 2, 'first')
        require_unsigned(self.bias_sum, 8, 'bias_sum')
        if mode == ComputeMode.FP19 and self.bias_sum not in (30, 254):
            raise ContractViolation('FP19 bias sum')
        if mode == ComputeMode.FP9 and self.bias_sum not in (14, 22, 30):
            raise ContractViolation('FP9 bias sum')
        if len(self.tag) != 2 or len(self.scale_a) != 2 or len(self.scale_b) != 2:
            raise ContractViolation('two groups required')
        for g in range(2):
            require_unsigned(self.tag[g], 4, 'tag (TXT v1 fixes TAG_W=4)')
            for scale in (self.scale_a[g], self.scale_b[g]):
                require_unsigned(scale, 8, 'scale')
                if self.mx_enable and self.group_valid & (1 << g) and scale == 255:
                    raise ContractViolation('scale255 undefined')
        arrays = []
        for operands in (self.operand_a, self.operand_b):
            if len(operands) != 16:
                raise ContractViolation('dot16 required')
            result = []
            for value in operands:
                require_unsigned(value, 26, 'packed operand')
                if mode == ComputeMode.FP19:
                    if value & ~0xFF0FFF or abs(signed(value, 12)) > 2047:
                        raise ContractViolation('FP19 payload bits/range')
                    masked = value
                elif mode == ComputeMode.FP9:
                    if value & ~0x3FF03FF:
                        raise ContractViolation('FP9 reserved payload bits')
                    masked = 0
                    for g in range(2):
                        if self.group_valid & (1 << g):
                            if abs(signed(value >> (5*g), 5)) > 15:
                                raise ContractViolation('FP9 coefficient range')
                            masked |= value & ((31 << (5*g)) | (31 << (16+5*g)))
                else:
                    width = {ComputeMode.INT16: 16, ComputeMode.INT8: 8, ComputeMode.INT4: 4}[mode]
                    groups = 1 if wide else 2
                    if value >> (width*groups):
                        raise ContractViolation('integer reserved payload bits')
                    masked = 0
                    for g in range(groups):
                        if self.group_valid & (1 << g):
                            field = (value >> (width*g)) & ((1 << width)-1)
                            if field == 1 << (width-1):
                                raise ContractViolation('reserved integer minimum')
                            masked |= field << (width*g)
                result.append(masked)
            arrays.append(tuple(result))
        return Entry(mode, kind, *arrays, self.bias_sum, self.group_valid)


class PackedAccumulator:
    """Explicit-phase scoreboard including inactive banks and update metadata."""
    def __init__(self):
        self.valid = [False, False]
        self.kind = [0, 0]
        self.mode = [None, None]
        self.integer = [0, 0]
        self.floating = [FloatAcc36State(0, 0), FloatAcc36State(0, 0)]
        self.pending = None
        self.phase0 = None
        self.update_kind = 0
        self.update_tag = (0, 0)

    def evaluate(self, step: PackedStep):
        entry = step.entry()
        m = multiply(entry, step.phase)
        a = align(entry, m)
        p = partial_sum(entry, m, a)
        # mx only consumes group_valid/mx_enable/scale fields of its tx argument.
        x = mx(entry, step, m, a, p)
        is_int16 = step.mode == ComputeMode.INT16
        if self.pending is not None and not (is_int16 and step.phase):
            raise ContractViolation('interleaved INT16 phases')
        if is_int16 and step.phase:
            if self.pending is None or step.acc_first:
                raise ContractViolation('phase1 without phase0 or repeated first')
            prior = self.phase0
            if (step.operand_a, step.operand_b, step.tag, step.sequence) != (
                prior.operand_a, prior.operand_b, prior.tag, prior.sequence
            ):
                raise ContractViolation('phase pairing mismatch')
        else:
            for g in range(2):
                if step.group_valid & (1 << g) and not step.acc_first & (1 << g):
                    if not self.valid[g] or self.kind[g] != step.kind or self.mode[g] != step.mode:
                        raise ContractViolation('first required on initial use or mode change')
        # Compute both groups before committing so a rejected exponent is atomic.
        proposed = list(self.floating)
        if step.kind == AccKind.TC_FLOAT:
            for g in range(2):
                if step.group_valid & (1 << g):
                    proposed[g] = update_float_acc36(self.floating[g], x.contribution[g], x.exp[g], first=bool(step.acc_first & (1 << g)))
        commit = not (is_int16 and step.phase == 0)
        if is_int16:
            if step.phase == 0:
                self.pending = require_signed((0 if step.acc_first & 1 else self.integer[0]) + x.contribution[0], 36, 'INT16 mid')
                self.phase0 = step
            else:
                self.integer[0] = saturate_int32(self.pending + x.contribution[0])
                self.pending = self.phase0 = None
        else:
            self.floating = proposed
            if step.kind == AccKind.INT32:
                for g in range(2):
                    if step.group_valid & (1 << g):
                        self.integer[g] = saturate_int32((0 if step.acc_first & (1 << g) else self.integer[g]) + x.contribution[g])
        if commit:
            self.update_kind = int(step.kind)
            self.update_tag = step.tag
            for g in range(2):
                if step.group_valid & (1 << g):
                    self.valid[g], self.kind[g], self.mode[g] = True, int(step.kind), step.mode
        expected = [int(commit), step.group_valid if commit else 0, self.update_kind,
                    *self.update_tag, int(self.valid[0]) | int(self.valid[1]) << 1]
        for g in range(2):
            f = self.floating[g]
            expected += [self.kind[g], self.integer[g], f.sig, f.exp, f.infinity_sign]
        return expected, (m, a, p, x)
