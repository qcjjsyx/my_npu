// Fixed numerical and interface definitions from IP_Design.md.
// Tags stay out of the shared metadata structs: TAG_W is a module parameter
// and packed struct members cannot be parameterized.
package mac_types_pkg;
  localparam int DOT_N = 16;
  localparam int OPERAND_W = 26;
  localparam int PSUM_W = 56;
  // Q2.34 operands/state, plus three low working bits for final RNE.
  localparam int ACC_OP_W = 36;
  localparam int ACC_FLOAT_SIG_W = ACC_OP_W;

  localparam int ACC_TAIL_W = 3;
  localparam int ACC_WORK_W = ACC_OP_W + ACC_TAIL_W;

  // Encoding choice for this RTL, not an additional numerical format.
  typedef enum logic [2:0] {
    SRC_FP16 = 3'd0,
    SRC_BF16 = 3'd1,
    SRC_E4M3 = 3'd2,
    SRC_E5M2 = 3'd3,
    SRC_INT16 = 3'd4,
    SRC_INT8 = 3'd5,
    SRC_INT4 = 3'd6
  } src_fmt_t;

  typedef enum logic [2:0] {
    MODE_FP19 = 3'd0,
    MODE_FP9 = 3'd1,
    MODE_INT16 = 3'd2,
    MODE_INT8 = 3'd3,
    MODE_INT4 = 3'd4
  } compute_mode_t;

  typedef enum logic {
    ACC_INT32 = 1'b0,
    ACC_TC_FLOAT = 1'b1
  } acc_kind_t;

  typedef logic [OPERAND_W-1:0] operand_t;
  typedef logic signed [23:0] product12_t;
  typedef logic signed [17:0] product9_t;
  typedef logic signed [31:0] align_lane0_t;
  typedef logic signed [23:0] align_lane1_t;
  typedef logic [PSUM_W-1:0] psum_packed_t;
  typedef logic signed [35:0] contribution0_t;
  typedef logic signed [35:0] contribution1_t;
  typedef logic [7:0] acc_exp_t;
  typedef logic signed [35:0] int_work_t;
  typedef logic signed [ACC_FLOAT_SIG_W-1:0] acc_sig_t;

  // Transaction control and scale metadata accompanying the numerical data
  // through the entry, M, ALIGN and PSUM stage boundaries (IP_Design.md
  // section 4.5). Both scale slots are packed as [group][byte].
  typedef struct packed {
    compute_mode_t mode;
    acc_kind_t     acc_kind;
    logic [7:0]    bias_sum;
    logic [1:0]    group_valid;
    logic [1:0]    acc_first;
    logic          mx_enable;
    logic [1:0][7:0] scale_a;
    logic [1:0][7:0] scale_b;
    logic          int16_phase;
  } txn_meta_t;

  // Per-group alignment results produced by ALIGN and consumed by MX.
  // group_nz records "a nonzero product took part in MAX", which is not the
  // same as a nonzero reduction result. inexact is diagnostic only and is
  // never injected into any numerical path.
  typedef struct packed {
    logic [8:0] emax0;
    logic [5:0] emax1;
    logic [1:0] group_nz;
    logic [1:0] inexact;
  } align_meta_t;

  // Control and diagnostic metadata carried from MX to ACC.
  typedef struct packed {
    compute_mode_t mode;
    acc_kind_t     acc_kind;
    logic [1:0]    group_valid;
    logic [1:0]    acc_first;
    logic          int16_phase;
    logic [1:0]    group_nz;
    logic [1:0]    inexact;
  } mx_meta_t;

  typedef struct packed {
    product12_t product0;
    product9_t product1;
    logic [8:0] exp_sum0;
    logic [5:0] exp_sum1;
    logic [1:0] product_nz;
  } m_lane_t;

  typedef struct packed {
    align_lane0_t lane0;
    align_lane1_t lane1;
  } align_lane_t;

  function automatic logic is_fp8(input src_fmt_t fmt);
    return (fmt == SRC_E4M3) || (fmt == SRC_E5M2);
  endfunction

  // A and B may multiply only as equal formats or as an FP8 pair.
  function automatic logic legal_src_pair(input src_fmt_t fmt_a,
                                           input src_fmt_t fmt_b);
    case (fmt_a)
      SRC_FP16, SRC_BF16, SRC_INT16, SRC_INT8, SRC_INT4:
        return fmt_a == fmt_b;
      SRC_E4M3, SRC_E5M2:
        return is_fp8(fmt_b);
      default: return 1'b0;
    endcase
  endfunction

  function automatic logic legal_mx_enable(input src_fmt_t fmt_a,
                                            input src_fmt_t fmt_b,
                                            input logic mx_enable);
    if (!legal_src_pair(fmt_a, fmt_b)) return 1'b0;
    if (!mx_enable) return 1'b1;
    return is_fp8(fmt_a) || (fmt_a == SRC_INT8);
  endfunction

  function automatic compute_mode_t mode_for(input src_fmt_t fmt_a);
    case (fmt_a)
      SRC_FP16, SRC_BF16: return MODE_FP19;
      SRC_E4M3, SRC_E5M2: return MODE_FP9;
      SRC_INT16: return MODE_INT16;
      SRC_INT8: return MODE_INT8;
      SRC_INT4: return MODE_INT4;
      default: return MODE_FP19; // Invalid request must be rejected separately.
    endcase
  endfunction

  function automatic acc_kind_t acc_kind_for(input compute_mode_t mode,
                                              input logic mx_enable);
    if ((mode == MODE_FP19) || (mode == MODE_FP9) ||
        ((mode == MODE_INT8) && mx_enable)) return ACC_TC_FLOAT;
    return ACC_INT32;
  endfunction

  function automatic logic is_wide_mode(input compute_mode_t mode);
    return (mode == MODE_FP19) || (mode == MODE_INT16);
  endfunction

  // Modes whose M output carries live floating-point exponent sums and
  // product-nz flags. Integer modes isolate that logic (section 2.6).
  function automatic logic is_float_mode(input compute_mode_t mode);
    return (mode == MODE_FP19) || (mode == MODE_FP9);
  endfunction

  // The lane1 datapath (MUL9, tree1 side) carries a live result only for the
  // group0 high limb of INT16 or for group1 of the DUAL modes. FP19 keeps
  // MUL9 operand-isolated (sections 1.2 and 3.5.3).
  function automatic logic lane1_active(input compute_mode_t mode,
                                         input logic [1:0] group_valid);
    case (mode)
      MODE_INT16: return group_valid[0];
      MODE_FP9, MODE_INT8, MODE_INT4: return group_valid[1];
      default: return 1'b0;
    endcase
  endfunction

  function automatic logic [7:0] source_bias(input src_fmt_t fmt);
    case (fmt)
      SRC_FP16, SRC_E5M2: return 8'd15;
      SRC_BF16: return 8'd127;
      SRC_E4M3: return 8'd7;
      default: return 8'd0;
    endcase
  endfunction

  function automatic logic [7:0] bias_sum_for(input src_fmt_t fmt_a,
                                               input src_fmt_t fmt_b);
    logic [8:0] wide_sum;
    wide_sum = {1'b0, source_bias(fmt_a)} + {1'b0, source_bias(fmt_b)};
    return wide_sum[7:0]; // Valid floating combinations have sum <= 254.
  endfunction
endpackage
