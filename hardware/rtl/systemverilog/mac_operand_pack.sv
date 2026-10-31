// One raw A or B operand container to the fixed 26-bit M input payload
// (IP_Design.md 2.5/2.6). Combinational; pair legality, reserved integer
// encodings, metadata and in_fire are checked and registered by the later
// entry/M boundary, not here.
//
// The two FP8 lanes use two parallel converters; integer lanes pack
// directly with lane0 low and lane1 high. An invalid group drives both the
// exponent and data fields of its lanes to zero; every reserved container
// bit is zero.
module mac_operand_pack (
  input  logic [15:0] i_raw,
  input  mac_types_pkg::src_fmt_t i_src_fmt,
  input  logic [1:0] i_group_valid,
  output mac_types_pkg::operand_t o_payload
);
  import mac_types_pkg::*;

  logic fp19_selected, fp9_selected;
  logic [15:0] raw19_iso;
  logic [7:0] raw9_0_iso, raw9_1_iso;
  logic [7:0] exp19;
  logic signed [11:0] sig19;
  logic [4:0] exp9_0, exp9_1;
  logic signed [4:0] sig9_0, sig9_1;

  // Operand isolation: an unselected or invalid lane sees a zero raw code,
  // which converts to canonical +0 and keeps the shared converters quiet.
  assign fp19_selected = (i_src_fmt == SRC_FP16) || (i_src_fmt == SRC_BF16);
  assign fp9_selected = (i_src_fmt == SRC_E4M3) || (i_src_fmt == SRC_E5M2);
  assign raw19_iso = (fp19_selected && i_group_valid[0]) ? i_raw : 16'd0;
  assign raw9_0_iso = (fp9_selected && i_group_valid[0]) ? i_raw[7:0] : 8'd0;
  assign raw9_1_iso = (fp9_selected && i_group_valid[1]) ? i_raw[15:8] : 8'd0;

  mac_fp19_convert u_fp19 (
    .i_raw(raw19_iso), .i_src_fmt(i_src_fmt),
    .o_exp_eff(exp19), .o_sig_tc(sig19)
  );
  mac_fp9_convert u_fp9_lane0 (
    .i_raw(raw9_0_iso), .i_src_fmt(i_src_fmt),
    .o_exp_eff(exp9_0), .o_sig_tc(sig9_0)
  );
  mac_fp9_convert u_fp9_lane1 (
    .i_raw(raw9_1_iso), .i_src_fmt(i_src_fmt),
    .o_exp_eff(exp9_1), .o_sig_tc(sig9_1)
  );

  always_comb begin
    o_payload = '0;
    case (i_src_fmt)
      SRC_FP16, SRC_BF16: begin
        if (i_group_valid[0]) begin
          o_payload[23:16] = exp19;
          o_payload[11:0] = sig19;
        end
      end
      SRC_E4M3, SRC_E5M2: begin
        if (i_group_valid[0]) begin
          o_payload[20:16] = exp9_0;
          o_payload[4:0] = sig9_0;
        end
        if (i_group_valid[1]) begin
          o_payload[25:21] = exp9_1;
          o_payload[9:5] = sig9_1;
        end
      end
      SRC_INT16: begin
        if (i_group_valid[0]) o_payload[15:0] = i_raw;
      end
      SRC_INT8: begin
        if (i_group_valid[0]) o_payload[7:0] = i_raw[7:0];
        if (i_group_valid[1]) o_payload[15:8] = i_raw[15:8];
      end
      SRC_INT4: begin
        if (i_group_valid[0]) o_payload[3:0] = i_raw[3:0];
        if (i_group_valid[1]) o_payload[7:4] = i_raw[7:4];
      end
      default: o_payload = '0; // No valid-transaction semantics for illegal fmt.
    endcase
  end
endmodule
