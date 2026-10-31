"""Deterministic random streams.

Every stream is derived from ``sha256`` over a stable key built from the
master seed, format, distribution, sample index, block index and purpose, so
that:

* sample ``i`` is unaffected by how many samples are requested;
* block ``i`` of a sample is unaffected by ``Bmax`` (prefix stability);
* amplitude, sign and lane-permutation randomness are independent;
* parallel scheduling cannot change the data.
"""

from __future__ import annotations

import hashlib
import random

from ..version import MX_INPUT_STREAM_VERSION, MX_STREAM_VERSION, STREAM_VERSION

PURPOSES = ("amplitude", "sign", "lane")


def stream_rng(
    master_seed: int,
    format_name: str,
    distribution: str,
    sample_index: int,
    block_index: int,
    purpose: str,
) -> random.Random:
    if purpose not in PURPOSES:
        raise ValueError(f"unknown stream purpose {purpose!r}")
    key = (
        f"mac-stream-v{STREAM_VERSION}|{master_seed}|{format_name}|{distribution}"
        f"|{sample_index}|{block_index}|{purpose}"
    )
    digest = hashlib.sha256(key.encode("ascii")).digest()
    return random.Random(int.from_bytes(digest, "big"))


MX_PURPOSES = ("amplitude", "sign", "scale")


def mx_stream_rng(
    master_seed: int,
    stream_kind: str,
    matrix: str,
    label: str,
    pair_index: int,
    index_a: int,
    index_b: int,
    purpose: str,
) -> random.Random:
    """Deterministic MX matrix stream (``MX_STREAM_VERSION`` = 2).

    The key contains the master seed, the stream kind (element or scale), the
    matrix ("x"/"w"), the strategy label (element distribution or scale
    scenario), the matrix-pair index and the global logical coordinates (row,
    column or group) -- never ``Kmax``/``M``/``N``, the distribution list or the
    execution order.  Therefore:

    * different matrix pairs of one seed use disjoint streams;
    * sample prefixes stay stable when the maximum K grows;
    * scale scenarios and element distributions use disjoint keys, so switching
      a scale scenario cannot change element codes;
    * adding a distribution, scenario or pair cannot perturb an existing
      stream.

    Version 1 omitted ``pair_index``, which made every pair of a seed
    identical; v1 runs stay readable as historical records and are identified
    by the missing ``mx_stream_version`` manifest field.
    """
    if purpose not in MX_PURPOSES:
        raise ValueError(f"unknown MX stream purpose {purpose!r}")
    key = (
        f"mac-mx-stream-v{MX_STREAM_VERSION}|{master_seed}|{stream_kind}|{matrix}"
        f"|{pair_index}|{label}|{index_a}|{index_b}|{purpose}"
    )
    digest = hashlib.sha256(key.encode("ascii")).digest()
    return random.Random(int.from_bytes(digest, "big"))


class CodeFingerprint:
    """Incremental fingerprint over the encoded input sequence."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()
        self.count = 0

    def update(self, code: int) -> None:
        self._hash.update(int(code).to_bytes(4, "little", signed=False))
        self.count += 1

    def hexdigest(self) -> str:
        return "sha256:" + self._hash.hexdigest()


MX_INPUT_PURPOSES = ("value",)


def mx_input_stream_rng(
    master_seed: int,
    matrix: str,
    distribution_label: str,
    pair_index: int,
    row: int,
    purpose: str,
) -> random.Random:
    """Deterministic numeric-domain FP32 input stream (``MX_INPUT_STREAM_VERSION``).

    Key: ``mac-mx-fp32-stream-v1|seed|matrix|distribution_label|pair_index|row|purpose``.

    * ``matrix`` separates activation ("x") from weight ("w") streams;
    * ``distribution_label`` is the canonical distribution + parameter string, so
      changing parameters changes the stream while switching the other matrix's
      distribution does not;
    * ``pair_index`` makes every matrix pair of one seed distinct;
    * only the global row is in the key; columns are drawn sequentially, so a
      smaller requested K (prefix) never changes existing draws;
    * neither ``Kmax``/``M``/``N`` nor the configuration traversal order appear,
      and Python's process-randomised ``hash()`` is never used.
    """
    if purpose not in MX_INPUT_PURPOSES:
        raise ValueError(f"unknown MX input stream purpose {purpose!r}")
    key = (
        f"mac-mx-fp32-stream-v{MX_INPUT_STREAM_VERSION}|{master_seed}|{matrix}"
        f"|{distribution_label}|{pair_index}|{row}|{purpose}"
    )
    digest = hashlib.sha256(key.encode("ascii")).digest()
    return random.Random(int.from_bytes(digest, "big"))
