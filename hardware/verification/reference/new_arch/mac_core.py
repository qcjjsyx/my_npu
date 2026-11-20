"""End-to-end MAC cycle oracle using mathematical contributions and exact RNE40."""

from collections import deque
from dataclasses import dataclass

from .forward import Mode, forward_groups
from .oracle import exact_update


ZERO_FIELDS = (0,) * 16
TWO_GROUPS = (Mode.FP4, Mode.FP8, Mode.INT4, Mode.INT8)
WRITEBACK_DELAY = 11


def pack_float(m: int, h: int) -> int:
    return (h % (1 << 10)) * (1 << 41) + m % (1 << 41)


@dataclass(frozen=True)
class MacInput:
    mode: Mode | int
    ctx: int
    first: bool = True
    last: bool = True
    a: tuple[int, ...] = ZERO_FIELDS
    b: tuple[int, ...] = ZERO_FIELDS
    mx: bool = False
    scale_a: int = 0
    scale_b: int = 0

    @property
    def meta(self) -> int:
        return self.mode * 32 + self.mx * 16 + self.ctx * 4 + self.first * 2 + self.last

    @property
    def is_float(self) -> bool:
        return self.mx or self.mode < Mode.INT4

    @property
    def group_count(self) -> int:
        return 2 if self.mode in TWO_GROUPS else 1


@dataclass
class Context:
    last_input: int
    closing: bool
    done: bool = False
    states: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Writeback:
    txn: MacInput
    states: tuple[int, int]


def _state_words(states: tuple[tuple[int, int], ...], is_float: bool) -> tuple[int, int]:
    words = tuple(pack_float(m, h) if is_float else m % (1 << 51) for m, h in states)
    return words[0], words[1] if len(words) == 2 else 0


class MacCoreModel:
    """Schedule commits 11 edges after acceptance and expose only completed slots."""

    def __init__(self) -> None:
        self.cycle = 0
        self.reset()

    def reset(self) -> None:
        self.pending: deque[MacInput | None] = deque([None] * WRITEBACK_DELAY)
        self.contexts: dict[int, Context] = {}
        self.mode, self.mx = Mode.FP4, False

    def outputs(self, selected: int) -> tuple[int, ...]:
        free = sum(1 << ctx for ctx in range(3) if ctx not in self.contexts)
        done = sum(1 << ctx for ctx, slot in self.contexts.items() if slot.done)
        empty = int(not any(self.pending))
        slot = self.contexts.get(selected)
        if slot is None or not slot.done:
            return free, done, empty, 0, 0, 0, 0, 0
        words = _state_words(slot.states, self.mx or self.mode < Mode.INT4)
        return free, done, empty, 1, *words, int(self.mode), int(self.mx)

    def step(self, txn: MacInput | None, selected: int, ready: bool,
             rst_n: bool = True) -> Writeback | None:
        """Apply all events at one edge. Test stimuli follow the legal scheduling contract."""
        cycle = self.cycle
        self.cycle += 1
        if not rst_n:
            self.reset()
            return None

        release = selected in self.contexts and self.contexts[selected].done and ready
        if txn is not None:
            assert 0 <= txn.ctx < 3, txn
            slot = self.contexts.get(txn.ctx)
            if not self.contexts and not any(self.pending):
                assert txn.first, txn
                self.mode, self.mx = txn.mode, txn.mx
            assert (txn.mode, txn.mx) == (self.mode, self.mx), txn
            if txn.first:
                assert slot is None, (cycle, txn, slot)
                self.contexts[txn.ctx] = Context(cycle, txn.last)
            else:
                assert slot is not None and not slot.closing, (cycle, txn, slot)
                assert cycle - slot.last_input >= 3, (cycle, txn, slot)
                slot.last_input = cycle
                slot.closing = txn.last
        if release:
            del self.contexts[selected]

        commit = self.pending.pop()
        self.pending.appendleft(txn)
        if commit is None:
            return None

        slot = self.contexts[commit.ctx]
        old = ((0, 0),) * commit.group_count if commit.first else slot.states
        groups = forward_groups(commit.mode, commit.mx, commit.a, commit.b,
                                commit.scale_a, commit.scale_b)
        states = []
        for (m_old, h_old), group in zip(old, groups[:commit.group_count], strict=True):
            if commit.is_float:
                m, h, _ = group.mx_record(True)
                states.append(exact_update(m_old, h_old, m, h))
            else:
                states.append((m_old + group.n, 0))
        slot.states = tuple(states)
        slot.done = commit.last
        return Writeback(commit, _state_words(slot.states, commit.is_float))
