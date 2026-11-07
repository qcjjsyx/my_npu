// Independent physical 24-bit lane1 distance/shifter/RNE path (IP_Design.md
// 6.5-6.7). Only FP9 uses the shifter; FP19 keeps the lane quiet and the
// integer modes bypass with a plain sign extension. Lane1 never shares the
// lane0 shifter.
module mac_align_lane1 (
  input  mac_types_pkg::compute_mode_t i_mode,
  input  logic i_active,
  input  mac_types_pkg::product9_t i_product,
  input  logic [5:0] i_exp_sum,
  input  logic i_product_nz,
  input  logic [5:0] i_emax,
  output mac_types_pkg::align_lane1_t o_aligned,
  output logic o_inexact,
  output logic o_round_overflow
);
  import mac_types_pkg::*;

  logic signed [23:0] y;
  logic [8:0] distance;
  logic [6:0] distance9;
  logic signed [23:0] rounded;
  logic round_inexact;

  // Widened subtraction keeps the top carry; the parent asserts the result
  // is nonnegative and uses the low 6 bits.
  assign distance9 = {1'b0, i_emax} - {1'b0, i_exp_sum};

  // signed9 payload with the fixed K=15 fraction boost (section 6.4). An
  // inactive or non-FP9 lane drives zero to keep the shifter quiet.
  always_comb begin
    y = '0;
    distance = '0;
    if (i_active && i_mode == MODE_FP9) begin
      y = {i_product[8:0], 15'b0};
      distance = {3'b000, distance9[5:0]};
    end
  end

  mac_align_rne #(.W(24)) u_rne (
    .i_y(y), .i_distance(distance), .i_window24(1'b1),
    .o_x(rounded), .o_inexact(round_inexact),
    .o_overflow(o_round_overflow)
  );

  always_comb begin
    o_aligned = '0;
    o_inexact = 1'b0;
    if (i_active) begin
      if (i_mode == MODE_FP9) begin
        // A zero product keeps a zero aligned value (section 2.2.3).
        if (i_product_nz) begin
          o_aligned = rounded;
          o_inexact = round_inexact;
        end
      end else if (i_mode != MODE_FP19) begin
        o_aligned = $signed({{6{i_product[17]}}, i_product});
      end
    end
  end
endmodule
