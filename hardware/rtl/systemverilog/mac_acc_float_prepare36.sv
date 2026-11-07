// Q2.34 operand preparation. Normalize each finite coefficient EXACTLY in
// ACC (not MX), then shift-right-jam its magnitude into a common exponent
// with three low work bits. There is no per-operand RNE.
//
// Sufficiency: normalized nonzero magnitudes are in [2^34,2^35). Distance
// <=3 loses no bits after appending three zeros, so deep cancellation is
// exact. At distance >=4 the smaller magnitude is <1/8 of the larger's
// minimum magnitude: subtraction can left-normalize by at most one bit.
// Final rounding therefore discards at least two work bits. Jam preserves
// the guard/halfway ordering for addition and subtraction; it cannot be
// interpreted as an independent operand rounding. See verification tests
// against the arbitrary-precision oracle, including close cancellation.
module mac_acc_float_prepare36 (
  input logic i_first,
  input mac_types_pkg::acc_sig_t i_old_sig,
  input mac_types_pkg::acc_exp_t i_old_exp,
  input mac_types_pkg::acc_sig_t i_new_sig,
  input mac_types_pkg::acc_exp_t i_new_exp,
  output logic signed [38:0] o_old_operand,
  output logic signed [38:0] o_new_operand,
  output logic signed [10:0] o_window_exp
);
  logic signed [35:0] old_sig;
  logic [35:0] old_mag, new_mag, old_norm, new_norm;
  logic [38:0] old_work, new_work;
  integer old_lead, new_lead, old_e, new_e, common_e;

  function automatic logic [38:0] shift_right_jam(
    input logic [38:0] value, input integer distance
  );
    logic [38:0] shifted, lost;
    begin
      shifted = '0;
      lost = '0;
      if (distance == 0) shifted = value;
      else if (distance < 39) begin
        shifted = value >> distance;
        lost = value << (39-distance);
        shifted[0] = shifted[0] | (|lost);
      end else shifted[0] = |value;
      return shifted;
    end
  endfunction

  always @* begin
    old_sig = i_first ? 36'sd0 : i_old_sig;
    old_mag = old_sig[35] ? (~$unsigned(old_sig) + 36'd1) : $unsigned(old_sig);
    new_mag = i_new_sig[35] ? (~$unsigned(i_new_sig) + 36'd1) : $unsigned(i_new_sig);
    old_lead = 0;
    new_lead = 0;
    for (integer j=0; j<36; j=j+1) begin
      if (old_mag[j]) old_lead = j;
      if (new_mag[j]) new_lead = j;
    end
    old_e = $signed({1'b0,i_old_exp}) + old_lead - 34;
    new_e = $signed({1'b0,i_new_exp}) + new_lead - 34;
    // lead=35 only occurs for -2^35, whose right shift is exact.
    old_norm = old_lead > 34 ? old_mag >> 1 : old_mag << (34-old_lead);
    new_norm = new_lead > 34 ? new_mag >> 1 : new_mag << (34-new_lead);
    common_e = 0;
    if (old_mag != 0 && new_mag != 0) common_e = old_e > new_e ? old_e : new_e;
    else if (old_mag != 0) common_e = old_e;
    else if (new_mag != 0) common_e = new_e;
    old_work = '0;
    new_work = '0;
    if (old_mag != 0) old_work = shift_right_jam({old_norm,3'b0},common_e-old_e);
    if (new_mag != 0) new_work = shift_right_jam({new_norm,3'b0},common_e-new_e);
    o_old_operand = old_sig[35] ? -$signed(old_work) : $signed(old_work);
    o_new_operand = i_new_sig[35] ? -$signed(new_work) : $signed(new_work);
    o_window_exp = common_e;
  end
endmodule
