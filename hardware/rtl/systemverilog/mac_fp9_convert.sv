// One E4M3FN/E5M2 raw byte to finite FP9 compute fields (effective exponent
// plus two's-complement significand). Combinational, with no
// transaction-valid interpretation.
//
// Same three-step structure as mac_fp19_convert (IP_Design.md 2.1/2.3/2.2.1).
// E4M3FN has no Inf encoding: exp=15 with frac<=6 stays finite normal, only
// exp=15 with frac=7 is NaN.
module mac_fp9_convert (
  input  logic [7:0] i_raw,
  input  mac_types_pkg::src_fmt_t i_src_fmt,
  output logic [4:0] o_exp_eff,
  output logic signed [4:0] o_sig_tc
);
  import mac_types_pkg::*;

  // Step 1: raw field unpack, classification and the per-format maximum
  // finite replacement code.
  logic sign_src;
  logic [4:0] exp_src;
  logic [2:0] frac9_src;
  logic is_zero, is_nan, is_inf, fmt_supported;
  logic [4:0] max_finite_exp;
  logic [2:0] max_finite_frac9;

  // Step 2 results.
  logic flush_zero;
  logic [4:0] finite_exp;
  logic [2:0] finite_frac9;

  // Step 3 temporaries.
  logic [3:0] mag_u4;
  logic signed [4:0] mag_pos_s5;

  always_comb begin
    sign_src = 1'b0;
    exp_src = '0;
    frac9_src = '0;
    is_zero = 1'b0;
    is_nan = 1'b0;
    is_inf = 1'b0;
    fmt_supported = 1'b1;
    max_finite_exp = '0;
    max_finite_frac9 = '0;
    case (i_src_fmt)
      SRC_E4M3: begin
        sign_src = i_raw[7];
        exp_src = {1'b0, i_raw[6:3]};
        frac9_src = i_raw[2:0];
        is_zero = (i_raw[6:3] == 4'd0) && (i_raw[2:0] == 3'd0);
        is_nan  = (i_raw[6:3] == 4'd15) && (i_raw[2:0] == 3'd7);
        // No Inf branch: E4M3FN exp=15/frac<=6 is finite normal.
      end
      SRC_E5M2: begin
        sign_src = i_raw[7];
        exp_src = i_raw[6:2];
        frac9_src = {i_raw[1:0], 1'b0};
        is_zero = (i_raw[6:2] == 5'd0) && (i_raw[1:0] == 2'd0);
        is_nan  = (i_raw[6:2] == 5'd31) && (i_raw[1:0] != 2'd0);
        is_inf  = (i_raw[6:2] == 5'd31) && (i_raw[1:0] == 2'd0);
        max_finite_exp = 5'd30;
        max_finite_frac9 = {2'b11, 1'b0};
      end
      default: fmt_supported = 1'b0;
    endcase

    // NaN and both signed zeros become canonical +0. Inf keeps its sign and
    // takes the source format's maximum finite magnitude (section 2.3).
    flush_zero = !fmt_supported || is_nan || is_zero;
    finite_exp = is_inf ? max_finite_exp : exp_src;
    finite_frac9 = is_inf ? max_finite_frac9 : frac9_src;

    // Subnormal Preserve (hidden=0, Eeff=1) and the exact conditional
    // negate of the nonnegative magnitude (sections 1.3, 2.2.1).
    if (flush_zero) begin
      o_exp_eff = '0;
      o_sig_tc = '0;
      mag_u4 = '0;
      mag_pos_s5 = '0;
    end else begin
      o_exp_eff = (finite_exp == 5'd0) ? 5'd1 : finite_exp;
      mag_u4 = {finite_exp != 5'd0, finite_frac9};
      mag_pos_s5 = $signed({1'b0, mag_u4});
      o_sig_tc = sign_src ? -mag_pos_s5 : mag_pos_s5;
    end
  end
endmodule
