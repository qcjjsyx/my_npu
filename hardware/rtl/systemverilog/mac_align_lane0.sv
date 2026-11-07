// One physical 32-bit lane0 distance/shifter/RNE path (IP_Design.md
// 6.4-6.7). FP19 uses the full 32-bit quantization window; FP9 sign-extends
// its 24-bit Y into the shared shifter and keeps the 24-bit window. Integer
// modes bypass with a plain sign extension and never touch the shifter.
module mac_align_lane0 (
  input  mac_types_pkg::compute_mode_t i_mode,
  input  logic i_active,
  input  mac_types_pkg::product12_t i_product,
  input  logic [8:0] i_exp_sum,
  input  logic i_product_nz,
  input  logic [8:0] i_emax,
  output mac_types_pkg::align_lane0_t o_aligned,
  output logic o_inexact,
  output logic o_round_overflow
);
  import mac_types_pkg::*;

  logic signed [31:0] y;
  logic [8:0] distance;
  logic signed [31:0] rounded;
  logic round_inexact;
  logic fp9_mode;
  logic [9:0] distance19;
  logic [6:0] distance9;

  assign fp9_mode = (i_mode == MODE_FP9);
  // Both subtractions are explicitly widened so the top carry is preserved;
  // the parent asserts the distance is nonnegative and uses the low bits.
  assign distance19 = {1'b0, i_emax} - {1'b0, i_exp_sum};
  assign distance9 = {1'b0, i_emax[5:0]} - {1'b0, i_exp_sum[5:0]};

  // Payload extraction and fixed fraction boost (K=9 for FP19, K=15 for
  // FP9, section 6.4). The boost left-shifts every redundant sign bit out,
  // so the results are plain concatenations. Inactive and integer lanes
  // drive a zero Y and distance to keep the shifter and RNE logic quiet.
  always_comb begin
    y = '0;
    distance = '0;
    if (i_active) begin
      case (i_mode)
        MODE_FP19: begin
          // signed23 payload: {sign, p[22:0]} <<< 9 keeps no sign copies.
          y = {i_product[22:0], 9'b0};
          distance = distance19[8:0];
        end
        MODE_FP9: begin
          // signed9 payload boosted into signed24, then sign-extended to
          // the 32-bit shifter without the forbidden extra shift (6.5).
          y = {{8{i_product[8]}}, i_product[8:0], 15'b0};
          distance = {3'b000, distance9[5:0]};
        end
        default: begin end
      endcase
    end
  end

  mac_align_rne #(.W(32)) u_rne (
    .i_y(y), .i_distance(distance),
    .i_window24(fp9_mode),
    .o_x(rounded), .o_inexact(round_inexact),
    .o_overflow(o_round_overflow)
  );

  always_comb begin
    o_aligned = '0;
    o_inexact = 1'b0;
    if (i_active) begin
      if (is_float_mode(i_mode)) begin
        // A zero product keeps a zero aligned value (section 2.2.3).
        if (i_product_nz) begin
          o_aligned = rounded;
          o_inexact = round_inexact;
        end
      end else begin
        o_aligned = $signed({{8{i_product[23]}}, i_product});
      end
    end
  end
endmodule
