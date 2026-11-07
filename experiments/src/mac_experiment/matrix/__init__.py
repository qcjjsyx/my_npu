"""MXFP8 matrix experiment core (FP32 inputs, sampling, MAC, quantisation).

Imports are lazy (PEP 562) so that offline analysis can import e.g.
``mac_experiment.matrix.tensor`` / ``matrix.quantize`` without pulling in the
execution engine, the aligned engine or the input sampler.  Public names are
still available as ``mac_experiment.matrix.<name>``.
"""

from __future__ import annotations

import importlib

__all__ = [
    "Fp32InputSampler",
    "GROUP_LENGTH",
    "GroupQuantization",
    "INPUT_DISTRIBUTIONS",
    "MatrixQuantization",
    "MxMatrixEngine",
    "MxPairResult",
    "MxSampleSpec",
    "MxTensor",
    "align_dot16",
    "align_term",
    "aligned_container_bounds",
    "check_container",
    "decode_magnitude",
    "evaluate_mx_pair",
    "evaluate_mx_pair_aligned",
    "fp32_bits_to_dyadics",
    "quantize_fp32_bits_to_mx",
    "quantize_matrix_independent",
    "quantize_matrix_shared",
    "select_scale_exponent",
    "trace_dot16",
]

_EXPORTS = {
    "Fp32InputSampler": (".inputs", "Fp32InputSampler"),
    "GROUP_LENGTH": (".tensor", "GROUP_LENGTH"),
    "GroupQuantization": (".quantize", "GroupQuantization"),
    "INPUT_DISTRIBUTIONS": (".inputs", "INPUT_DISTRIBUTIONS"),
    "MatrixQuantization": (".quantize", "MatrixQuantization"),
    "MxMatrixEngine": (".engine", "MxMatrixEngine"),
    "MxPairResult": (".engine", "MxPairResult"),
    "MxSampleSpec": (".engine", "MxSampleSpec"),
    "MxTensor": (".tensor", "MxTensor"),
    "align_dot16": (".alignment", "align_dot16"),
    "align_term": (".alignment", "align_term"),
    "aligned_container_bounds": (".alignment", "aligned_container_bounds"),
    "check_container": (".alignment", "check_container"),
    "decode_magnitude": (".alignment", "decode_magnitude"),
    "evaluate_mx_pair": (".engine", "evaluate_mx_pair"),
    "evaluate_mx_pair_aligned": (".aligned_engine", "evaluate_mx_pair_aligned"),
    "fp32_bits_to_dyadics": (".inputs", "fp32_bits_to_dyadics"),
    "quantize_fp32_bits_to_mx": (".inputs", "quantize_fp32_bits_to_mx"),
    "quantize_matrix_independent": (".quantize", "quantize_matrix_independent"),
    "quantize_matrix_shared": (".quantize", "quantize_matrix_shared"),
    "select_scale_exponent": (".quantize", "select_scale_exponent"),
    "trace_dot16": (".aligned_engine", "trace_dot16"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
