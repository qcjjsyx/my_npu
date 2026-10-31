"""MXFP8 matrix MAC engine: exact reference, dot16 FP32 accumulation, prefixes.

Inputs are numeric-domain FP32 matrices quantised to MXFP8 with G32 input
scales (see ``matrix/inputs.py``).  The engine itself consumes the MX element /
scale codes exactly like before:

* ``X[m,k] = 2**alpha[m,g] * u[m,k]`` and ``W[k,n] = 2**beta[g,n] * v[k,n]``
  with ``g = k // 32``; scales are applied exactly as dyadic exponent shifts.
* exact reference ``S[m,n] = sum_k X[m,k] * W[k,n]`` over the **decoded MX
  values** (arbitrary precision dyadics only; never the raw FP32 matrices and
  never FP32/FP64/NumPy matmul);
* L1 reference ``sum_k |X*W|`` over the same decoded MX products;
* the 16-term dot is summed exactly; ``r = RN32(r + s_b)`` performs exactly one
  RNE per dot16 update with gradual underflow;
* the two dot16 blocks of one 32-element input scale group share the group
  scale but update the FP32 state separately (never merged into a dot32);
* no finite window loss is modelled (the W-window reducer is not used).  The
  engine produces exact 16-term partial sums, so a future finite-W MX variant
  can feed ``ProductTerm`` values to the existing reducer without changing the
  dot16/scale/rounding order defined here.

The engine is streaming: one pass over Kmax with snapshots at the requested
paired-prefix K values.  ``evaluate_mx_pair`` is pure and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..accumulation.accumulator import Fp32Accumulator
from ..arithmetic.dyadic import Dyadic
from ..formats.e4m3_element import decode_element, is_nan_code
from ..formats.e8m0 import code_to_exponent
from ..version import (
    MX_INPUT_QUANTIZATION_POLICY,
    MX_INPUT_SAMPLING_VERSION,
    MX_INPUT_STREAM_VERSION,
)
from .inputs import Fp32InputSampler, quantize_fp32_bits_to_mx
from .tensor import GROUP_LENGTH, MxTensor

MAC_PARALLELISM = 16


@dataclass
class MxSampleSpec:
    sample_id: str
    seed: int
    pair_index: int
    activation_distribution: str
    weight_distribution: str
    x_fp32_bits: list[int]
    w_fp32_bits: list[int]
    x: MxTensor
    w: MxTensor
    input_events: dict
    input_diagnostics: dict


@dataclass
class MxPairResult:
    snapshots: list[dict]
    status: str
    blocks_completed: int


class MxMatrixEngine:
    def __init__(
        self,
        M: int,
        N: int,
        k_values: list[int],
        activation_distribution: str,
        weight_distribution: str,
        mac_parallelism: int = MAC_PARALLELISM,
        group_length: int = GROUP_LENGTH,
    ):
        if group_length != GROUP_LENGTH:
            raise ValueError(f"MXFP8 group_length must be {GROUP_LENGTH}, got {group_length}")
        if mac_parallelism != MAC_PARALLELISM:
            raise ValueError(f"MAC parallelism must be {MAC_PARALLELISM}, got {mac_parallelism}")
        self.M = int(M)
        self.N = int(N)
        if self.M < 1 or self.N < 1:
            raise ValueError("M and N must be positive")
        if self.N % group_length:
            raise ValueError(f"N must be a multiple of the group length {group_length}")
        self.k_values = sorted({int(k) for k in k_values})
        if not self.k_values:
            raise ValueError("at least one K value is required")
        self.kmax = max(self.k_values)
        for k in self.k_values:
            if k < group_length or k % group_length:
                raise ValueError(f"K={k} is not a positive multiple of the group length {group_length}")
        self.activation_distribution = activation_distribution
        self.weight_distribution = weight_distribution
        self.mac_parallelism = mac_parallelism
        self.group_length = group_length
        self.activation_sampler = Fp32InputSampler(activation_distribution)
        self.weight_sampler = Fp32InputSampler(weight_distribution)

    def sampling_metadata(self) -> dict:
        return {
            "activation": self.activation_sampler.metadata(),
            "weight": self.weight_sampler.metadata(),
            "element_format": "fp8_e4m3",
            "scale_format": "e8m0",
            "group_length": self.group_length,
            "mac_parallelism": self.mac_parallelism,
            "input_sampling_version": MX_INPUT_SAMPLING_VERSION,
            "input_stream_version": MX_INPUT_STREAM_VERSION,
            "input_quantization_policy": MX_INPUT_QUANTIZATION_POLICY,
            "pipeline": "numeric-domain sample -> FP32 bits -> G32 MXFP8 quantization -> dot16/FP32",
            "scope_note": (
                "measures the behaviour of the quantized MX inputs; it does not measure input "
                "quantization loss relative to the raw FP32 matrix product"
            ),
        }

    def build_sample(self, seed: int, pair_index: int) -> MxSampleSpec:
        x_bits = self.activation_sampler.sample_matrix(
            seed, "x", self.M, self.kmax, pair_index
        )
        w_bits = self.weight_sampler.sample_matrix(
            seed, "w", self.kmax, self.N, pair_index
        )
        x_input = quantize_fp32_bits_to_mx("x", x_bits, self.M, self.kmax, group_axis=1)
        w_input = quantize_fp32_bits_to_mx("w", w_bits, self.kmax, self.N, group_axis=0)
        sample_id = (
            f"mx|{self.activation_distribution}|{self.weight_distribution}"
            f"|seed{seed}|{pair_index:05d}"
        )
        return MxSampleSpec(
            sample_id=sample_id,
            seed=seed,
            pair_index=pair_index,
            activation_distribution=self.activation_distribution,
            weight_distribution=self.weight_distribution,
            x_fp32_bits=x_bits,
            w_fp32_bits=w_bits,
            x=x_input.tensor,
            w=w_input.tensor,
            input_events={"x": x_input.events, "w": w_input.events},
            input_diagnostics={"x": x_input.diagnostics, "w": w_input.diagnostics},
        )

    def input_record(self, sample: MxSampleSpec) -> dict:
        return {
            "record_type": "input_pair",
            "schema": "mac-mx-exp-v2",
            "mode": "mx_matrix_sweep",
            "sample_id": sample.sample_id,
            "seed": sample.seed,
            "pair_index": sample.pair_index,
            "activation_distribution": sample.activation_distribution,
            "weight_distribution": sample.weight_distribution,
            "M": self.M,
            "N": self.N,
            "Kmax": self.kmax,
            "K_list": list(self.k_values),
            "B_list": [k // self.mac_parallelism for k in self.k_values],
            "group_length": self.group_length,
            "mac_parallelism": self.mac_parallelism,
            "input_sampling_version": MX_INPUT_SAMPLING_VERSION,
            "input_stream_version": MX_INPUT_STREAM_VERSION,
            "input_quantization_policy": MX_INPUT_QUANTIZATION_POLICY,
            "activation_sampler": self.activation_sampler.metadata(),
            "weight_sampler": self.weight_sampler.metadata(),
            "x_fp32_bits": list(sample.x_fp32_bits),
            "w_fp32_bits": list(sample.w_fp32_bits),
            "x_input_quantization_events": sample.input_events["x"],
            "w_input_quantization_events": sample.input_events["w"],
            "x_input_diagnostics": sample.input_diagnostics["x"],
            "w_input_diagnostics": sample.input_diagnostics["w"],
            "x": sample.x.to_dict(),
            "w": sample.w.to_dict(),
        }

    def evaluate(self, sample: MxSampleSpec, update_log: list | None = None) -> MxPairResult:
        return evaluate_mx_pair(
            sample.x,
            sample.w,
            self.k_values,
            mac_parallelism=self.mac_parallelism,
            group_length=self.group_length,
            update_log=update_log,
        )


def evaluate_mx_pair(
    x: MxTensor,
    w: MxTensor,
    k_values: list[int],
    mac_parallelism: int = MAC_PARALLELISM,
    group_length: int = GROUP_LENGTH,
    update_log: list | None = None,
) -> MxPairResult:
    if x.group_axis != 1 or w.group_axis != 0:
        raise ValueError("MX inputs must group X along K (axis 1) and W along K (axis 0)")
    if x.cols != w.rows:
        raise ValueError(f"contraction mismatch: X is {x.shape}, W is {w.shape}")
    if x.cols % group_length or w.rows % group_length:
        raise ValueError("contraction dimension must be a multiple of the group length")
    snapshots = sorted({int(k) for k in k_values})
    if not snapshots:
        raise ValueError("at least one K value is required")
    kmax = max(snapshots)
    if x.cols != kmax:
        raise ValueError(f"X has K={x.cols} but the maximum requested K is {kmax}")
    if any(k % group_length for k in snapshots):
        raise ValueError("all K values must be multiples of the group length")
    if any(k % mac_parallelism for k in snapshots):
        raise ValueError("all K values must be multiples of the MAC parallelism")

    M, N = x.rows, w.cols
    grid = M * N
    kvalues_set = set(snapshots)
    dot16 = mac_parallelism
    group = group_length
    if group % dot16:
        raise ValueError("group length must be a multiple of the MAC parallelism")

    total_sum = [Dyadic.zero() for _ in range(grid)]
    l1_sum = [Dyadic.zero() for _ in range(grid)]
    block_sum = [Dyadic.zero() for _ in range(grid)]
    block_nonzero = [False] * grid
    accumulators = [Fp32Accumulator() for _ in range(grid)]
    n_nonfinite_updates = 0
    n_overflow_updates = 0
    n_absorbed_updates = 0
    n_all_zero_dot16 = 0

    records: list[dict] = []
    block_index = 0
    for k in range(kmax):
        g = k // group
        w_row = k * N
        dx: list[Dyadic] = []
        for m in range(M):
            code = x.element_codes[m * x.cols + k]
            if is_nan_code(code):
                raise ValueError(f"NaN element code in X at ({m}, {k}); MX inputs must be finite")
            element = decode_element(code)
            if element.is_zero():
                dx.append(element)
            else:
                scale = x.scale_codes[m * (x.cols // group) + g]
                dx.append(Dyadic(element.m, element.e + code_to_exponent(scale)))
        dw: list[Dyadic] = []
        group_col = (k // group) * N
        for n in range(N):
            code = w.element_codes[k * N + n]
            if is_nan_code(code):
                raise ValueError(f"NaN element code in W at ({k}, {n}); MX inputs must be finite")
            element = decode_element(code)
            if element.is_zero():
                dw.append(element)
            else:
                scale = w.scale_codes[group_col + n]
                dw.append(Dyadic(element.m, element.e + code_to_exponent(scale)))

        for m in range(M):
            a = dx[m]
            base = m * N
            a_zero = a.is_zero()
            for n in range(N):
                b = dw[n]
                if a_zero or b.is_zero():
                    continue
                product = a * b
                index = base + n
                total_sum[index] = total_sum[index] + product
                l1_sum[index] = l1_sum[index] + abs(product)
                block_sum[index] = block_sum[index] + product
                block_nonzero[index] = True

        block_index += 1
        if block_index % dot16 == 0:
            all_zero_block = True
            for index in range(grid):
                partial = block_sum[index]
                if block_nonzero[index]:
                    all_zero_block = False
                outcome = accumulators[index].update(partial)
                if update_log is not None:
                    update_log.append(partial)
                if outcome.non_finite:
                    n_nonfinite_updates += 1
                if outcome.overflowed:
                    n_overflow_updates += 1
                if outcome.absorbed:
                    n_absorbed_updates += 1
                block_sum[index] = Dyadic.zero()
                block_nonzero[index] = False
            if all_zero_block:
                n_all_zero_dot16 += 1

        if block_index in kvalues_set:
            r_bits = [acc.state.bits for acc in accumulators]
            nonfinite_count = sum(1 for acc in accumulators if not acc.state.is_finite())
            if nonfinite_count == 0:
                status = "ok"
            elif nonfinite_count == grid:
                status = "all_nonfinite"
            else:
                status = "partial_nonfinite"
            records.append(
                {
                    "record_type": "matrix_snapshot",
                    "M": M,
                    "N": N,
                    "K": block_index,
                    "B": block_index // dot16,
                    "group_length": group,
                    "mac_parallelism": dot16,
                    "S": [value.to_dict() for value in total_sum],
                    "L1": [value.to_dict() for value in l1_sum],
                    "r_bits": r_bits,
                    "nonfinite_count": nonfinite_count,
                    "events_cumulative": {
                        "nonfinite_fp32_updates": n_nonfinite_updates,
                        "overflow_fp32_updates": n_overflow_updates,
                        "absorbed_updates": n_absorbed_updates,
                        "all_zero_dot16": n_all_zero_dot16,
                    },
                    "status": status,
                }
            )

    final_nonfinite = sum(1 for acc in accumulators if not acc.state.is_finite())
    if final_nonfinite == 0:
        status = "ok"
    elif final_nonfinite == grid:
        status = "all_nonfinite"
    else:
        status = "partial_nonfinite"
    return MxPairResult(snapshots=records, status=status, blocks_completed=block_index)
