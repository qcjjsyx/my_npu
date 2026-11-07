// Per-term alignment quantization: floor arithmetic shift plus nonnegative
// two's-complement remainder, then round-to-nearest-even with ties-to-even
// (IP_Design.md 6.6). The W=33 lane0 instance can select the FP9 24-bit
// quantization window; the W=24 lane1 instance always uses that window.
//
// The complete distance is compared with the window first, so a large
// distance can never wrap a truncated shifter. For 0 < d < window the
// discarded low d bits are collected by one complementary left shift into
// fixed top positions: guard = lost[W-1], round = lost[W-2], sticky = the
// OR below them. This avoids every variable-width part-select.
module mac_align_rne #(
  parameter integer W
) (
  input  logic signed [W-1:0] i_y,
  input  logic [8:0] i_distance,
  input  logic i_window24,
  output logic signed [W-1:0] o_x,
  output logic o_inexact,
  output logic o_overflow
);
  localparam logic [8:0] PHYS_W = W;
  logic [8:0] limit;
  logic [8:0] back_shift;
  logic signed [W-1:0] q;
  logic signed [W-1:0] lost;
  logic signed [W:0] rounded;
  logic guard_bit, round_bit, sticky_bit, increment;

  assign limit = i_window24 ? 9'd24 : PHYS_W;
  assign back_shift = PHYS_W - i_distance;

  // The RNE sum may not fit the signed W-bit container for an extreme
  // positive Y; the parent treats this as a contract assertion, not a
  // saturating numerical feature.
  assign o_overflow = (i_distance != 9'd0) && (i_distance < limit) &&
                      (rounded[W] != rounded[W-1]);

  always_comb begin
    q = '0;
    lost = '0;
    rounded = '0;
    guard_bit = 1'b0;
    round_bit = 1'b0;
    sticky_bit = 1'b0;
    increment = 1'b0;
    o_x = '0;
    o_inexact = 1'b0;
    if (i_distance == 9'd0) begin
      o_x = i_y; // No shift, no quantization.
    end else if (i_distance >= limit) begin
      o_inexact = (i_y != '0); // Legal inputs round to exactly zero.
    end else begin
      // 0 < d < limit <= W, so back_shift stays within 1..W-1.
      q = i_y >>> i_distance;
      lost = i_y <<< back_shift;
      guard_bit = lost[W-1];
      round_bit = lost[W-2];
      sticky_bit = |lost[W-3:0];
      o_inexact = |lost;
      increment = guard_bit & (round_bit | sticky_bit | q[0]);
      rounded = {q[W-1], q} + {{W{1'b0}}, increment};
      o_x = rounded[W-1:0];
    end
  end

endmodule
