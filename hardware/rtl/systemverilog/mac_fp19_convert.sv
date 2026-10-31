// One FP16/BF16 raw code to finite FP19 compute fields (effective exponent
// plus two's-complement significand). Combinational, with no
// transaction-valid interpretation.
//
// IP_Design.md section 2.1 requires classification on the raw source fields,
// before any FP19 container extension. The conversion is then:
//   1. unpack and classify the raw fields (section 2.1)
//   2. special-value replacement (section 2.3)
//   3. subnormal Preserve, hidden/Eeff and conditional negate (1.3, 2.2.1)
module mac_fp19_convert (
  input  logic [15:0] i_raw,
  input  mac_types_pkg::src_fmt_t i_src_fmt,
  output logic [7:0] o_exp_eff,
  output logic signed [11:0] o_sig_tc
);
  import mac_types_pkg::*;

  // Step 1: raw field unpack, classification and the per-format maximum
  // finite replacement code.
  logic sign_src;
  logic [7:0] exp_src;
  logic [9:0] frac19_src;
  logic is_zero, is_nan, is_inf, fmt_supported;
  logic [7:0] max_finite_exp;
  logic [9:0] max_finite_frac19;

  // Step 2 results.
  logic flush_zero;
  logic [7:0] finite_exp;
  logic [9:0] finite_frac19;

  // Step 3 temporaries.
  logic [10:0] mag_u11;
  logic signed [11:0] mag_pos_s12;

  always_comb begin
    sign_src = 1'b0;
    exp_src = '0;
    frac19_src = '0;
    is_zero = 1'b0;
    is_nan = 1'b0;
    is_inf = 1'b0;
    fmt_supported = 1'b1;
    max_finite_exp = '0;
    max_finite_frac19 = '0;
    case (i_src_fmt)
      SRC_FP16: begin
        sign_src = i_raw[15];
        exp_src = {3'b000, i_raw[14:10]};
        frac19_src = i_raw[9:0];
        is_zero = (i_raw[14:10] == 5'd0) && (i_raw[9:0] == 10'd0);
        is_nan  = (i_raw[14:10] == 5'd31) && (i_raw[9:0] != 10'd0);
        is_inf  = (i_raw[14:10] == 5'd31) && (i_raw[9:0] == 10'd0);
        max_finite_exp = 8'd30;
        max_finite_frac19 = 10'h3ff;
      end
      SRC_BF16: begin
        sign_src = i_raw[15];
        exp_src = i_raw[14:7];
        frac19_src = {i_raw[6:0], 3'b000};
        is_zero = (i_raw[14:7] == 8'd0) && (i_raw[6:0] == 7'd0);
        is_nan  = (i_raw[14:7] == 8'd255) && (i_raw[6:0] != 7'd0);
        is_inf  = (i_raw[14:7] == 8'd255) && (i_raw[6:0] == 7'd0);
        max_finite_exp = 8'd254;
        max_finite_frac19 = {7'h7f, 3'b000};
      end
      default: fmt_supported = 1'b0;
    endcase

    // NaN and both signed zeros become canonical +0. Inf keeps its sign and
    // takes the source format's maximum finite magnitude (section 2.3).
    flush_zero = !fmt_supported || is_nan || is_zero;
    finite_exp = is_inf ? max_finite_exp : exp_src;
    finite_frac19 = is_inf ? max_finite_frac19 : frac19_src;

    // exp=0 with nonzero fraction is a preserved subnormal: hidden=0,
    // Eeff=1. The conditional negate is exact over the legal magnitude.
    if (flush_zero) begin
      o_exp_eff = '0;
      o_sig_tc = '0;
      mag_u11 = '0;
      mag_pos_s12 = '0;
    end else begin
      o_exp_eff = (finite_exp == 8'd0) ? 8'd1 : finite_exp;
      mag_u11 = {finite_exp != 8'd0, finite_frac19};
      mag_pos_s12 = $signed({1'b0, mag_u11});
      o_sig_tc = sign_src ? -mag_pos_s12 : mag_pos_s12;
    end
  end
endmodule
