"""Version and policy identifiers for the MAC precision experiment framework."""

PACKAGE_VERSION = "0.7.0"

SCHEMA_VERSION = "mac-exp-v2"

SUPPORTED_SCHEMAS = ("mac-exp-v1", "mac-exp-v2")

STREAM_VERSION = 1

# MXFP8 matrix experiment (separate schema; never mixed with mac-exp-* dot-product runs)
# Current input scheme: numeric-domain sampling -> FP32 matrices -> G32 MXFP8 input
# quantization -> exact reference / dot16 FP32 accumulation -> MX output comparison.
MX_SCHEMA_VERSION = "mac-mx-exp-v2"
MX_INPUT_SAMPLING_VERSION = "mx-fp32-input-v1"
MX_INPUT_STREAM_VERSION = 1
MX_INPUT_QUANTIZATION_POLICY = "min_binary_power_covering_group_max_from_fp32_bits"
MX_ELEMENT_SEMANTICS = "ocp_e4m3_no_ftz"
MX_SCALE_SEMANTICS = "ocp_e8m0_bias127_reserved_255"
MX_ACCUMULATION_POLICY = "exact_dot16_shared_scale_group_single_fp32_rne"
MX_OUTPUT_SCALE_POLICY = "min_binary_power_covering_group_max"

# Fixed-precision alignment K/B sweep on top of the MXFP8 input pipeline.
MX_ALIGN_SCHEMA_VERSION = "mac-mx-align-v1"
MX_ALIGN_POLICY = "fixed_w_align_dot16_rne_integer_alignment"
MX_ALIGN_DEFAULT = 21
MX_ALIGN_CONTAINER_EXTRA_BITS = 7  # signed container width = w_align + 7 bits

# MAC bit-width selection experiment (W_align / W_acc).  Independent schema,
# independent run IDs and result roots; it never reads historical inputs,
# results or analyses (inputs are regenerated from a new stream namespace).
WIDTHSELECT_SCHEMA_VERSION = "mac-width-v1"
WIDTHSELECT_ANALYSIS_VERSION = "mac-width-analysis-v1"
WIDTHSELECT_INPUT_SAMPLING_VERSION = "width-fp32-input-v1"
WIDTHSELECT_INPUT_STREAM_VERSION = 1
WIDTHSELECT_ALIGN_MODE = "widthselect_align"
WIDTHSELECT_ACC_MODE = "widthselect_acc"
WIDTHSELECT_JOINT_MODE = "widthselect_joint"
WIDTHSELECT_MODES = (WIDTHSELECT_ALIGN_MODE, WIDTHSELECT_ACC_MODE, WIDTHSELECT_JOINT_MODE)
WIDTHSELECT_ALIGN_POLICY = "dot16_finite_w_align_format_parameterized_rne_integer"
WIDTHSELECT_TF32_POLICY = "fp32_significand_rne_11bit_kept_subnormal_input_overflow_saturated"
WIDTHSELECT_CONTAINER_EXTRA_BITS = 7  # same bound holds for E4M3 and TF32 products
# W_acc custom accumulator (format confirmed by the user, docs/mac_width_selection.md section 4.1)
WIDTHSELECT_ACC_FORMAT_STATUS = "hwacc_v1_confirmed_implemented"
WIDTHSELECT_ACC_POLICY = (
    "two_complement_wacc_plus_1_significand_bias126_reserved255_"
    "rne_common_window_normalize_floor_no_hidden_state_no_zero_bypass"
)

# Retired v1 MX input scheme (independent E4M3 element + E8M0 scale sampling).
# Kept for the historical modules only; the active engine/config/analysis never
# call it.  Its stream version stays 2; new runs record ``input_stream_version``.
MX_STREAM_VERSION = 2

ALIGNMENT_POLICY = "rne_ties_even"

ACCUMULATION_POLICY = "exact_dyadic_partial_single_fp32_rne"

INPUT_FTZ = True

OUTPUT_SUBNORMAL_POLICY = "gradual_underflow"
