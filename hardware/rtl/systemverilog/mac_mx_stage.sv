// MX stage (IP_Design.md chapter 8): decode the 56-bit PSUM container per
// mode and produce Q2.34 coefficients and bias127 exponents. This stage performs no tree
// merge, no INT16 phase reweighting, no rounding and no saturation; the
// contribution and the exponent chain use the same elastic output slot, so
// both hold together under backpressure.
module mac_mx_stage #(
  parameter integer TAG_W
) (
  input  logic i_clk,
  input  logic i_rstn,
  input  logic i_psum_valid,
  output logic o_psum_ready,
  input  mac_types_pkg::psum_packed_t i_psum_data,
  input  mac_types_pkg::txn_meta_t i_psum_meta,
  input  mac_types_pkg::align_meta_t i_psum_ameta,
  input  logic [TAG_W-1:0] i_psum_tag [0:1],

  output logic o_mx_valid,
  input  logic i_mx_ready,
  output mac_types_pkg::contribution0_t o_mx_contribution0,
  output mac_types_pkg::contribution1_t o_mx_contribution1,
  output mac_types_pkg::acc_exp_t o_mx_exp0,
  output mac_types_pkg::acc_exp_t o_mx_exp1,
  output mac_types_pkg::mx_meta_t o_mx_meta,
  output logic [TAG_W-1:0] o_mx_tag [0:1]
);
  import mac_types_pkg::*;

  logic mx_space;
  logic psum_fire;
  contribution0_t contribution0_d;
  contribution1_t contribution1_d;
  acc_exp_t exp0_d;
  acc_exp_t exp1_d;
  logic signed [8:0] scale_a_s9 [0:1];
  logic signed [8:0] scale_b_s9 [0:1];
  logic signed [9:0] scale_sum_s10 [0:1];
  logic signed [12:0] exp_s13 [0:1];
  logic [8:0] emax_u9 [0:1];
  mx_meta_t meta_q;
  logic [TAG_W-1:0] tag_q [0:1];

  // This simulator generation cannot index packed struct members with a
  // variable, so the per-group fields used inside loops are mirrored into
  // plain locals.
  logic [1:0] psum_group_valid;
  logic [1:0][7:0] psum_scale_a, psum_scale_b;
  assign psum_group_valid = i_psum_meta.group_valid;
  assign psum_scale_a = i_psum_meta.scale_a;
  assign psum_scale_b = i_psum_meta.scale_b;

  // Per-group E8M0 combined scale exponent (section 8.5). Only
  // scale-enabled FP9 and INT8 transactions consume scale metadata; every
  // other path selects an exact zero adjustment without reading the scale
  // inputs. The 8'hff NaN code is rejected by the input contract, never
  // interpreted as an exponent.
  always_comb begin
    scale_a_s9[0] = '0;
    scale_a_s9[1] = '0;
    scale_b_s9[0] = '0;
    scale_b_s9[1] = '0;
    scale_sum_s10[0] = '0;
    scale_sum_s10[1] = '0;
    for (int g = 0; g < 2; g++) begin
      if (i_psum_meta.mx_enable && psum_group_valid[g] &&
          ((i_psum_meta.mode == MODE_FP9) ||
           (i_psum_meta.mode == MODE_INT8))) begin
        scale_a_s9[g] = $signed({1'b0, psum_scale_a[g]}) - 9'sd127;
        scale_b_s9[g] = $signed({1'b0, psum_scale_b[g]}) - 9'sd127;
        scale_sum_s10[g] =
            $signed({scale_a_s9[g][8], scale_a_s9[g]}) +
            $signed({scale_b_s9[g][8], scale_b_s9[g]});
      end
    end
  end

  // Contribution unpack (section 8.2): narrow integer interfaces are
  // sign-extended only where their redundant high bits have been checked.
  always_comb begin
    contribution0_d = '0;
    contribution1_d = '0;
    if (is_wide_mode(i_psum_meta.mode)) begin
      if (i_psum_meta.group_valid[0])
        contribution0_d = $signed(i_psum_data[35:0]);
    end else begin
      if (i_psum_meta.group_valid[0]) begin
        case (i_psum_meta.mode)
          MODE_FP9: contribution0_d = {i_psum_data[27:0], 8'b0};
          MODE_INT8: contribution0_d = i_psum_meta.mx_enable
            ? $signed({i_psum_data[18:0], 17'b0})
            : $signed({{17{i_psum_data[18]}}, i_psum_data[18:0]});
          MODE_INT4: contribution0_d = {{24{i_psum_data[11]}}, i_psum_data[11:0]};
          default: contribution0_d = '0;
        endcase
      end
      if (i_psum_meta.group_valid[1]) begin
        case (i_psum_meta.mode)
          MODE_FP9: contribution1_d = {i_psum_data[55:28], 8'b0};
          MODE_INT8: contribution1_d = i_psum_meta.mx_enable
            ? $signed({i_psum_data[46:28], 17'b0})
            : $signed({{17{i_psum_data[46]}}, i_psum_data[46:28]});
          MODE_INT4: contribution1_d = {{24{i_psum_data[39]}}, i_psum_data[39:28]};
          default: contribution1_d = '0;
        endcase
      end
    end
  end

  // Full signed intermediate E: a nonzero out-of-domain MX request is
  // rejected by the contract assertion. No wrap/FTZ/Inf policy is assigned.
  always_comb begin
    emax_u9[0] = i_psum_ameta.emax0;
    emax_u9[1] = {3'b000, i_psum_ameta.emax1};
    exp_s13[0] = '0;
    exp_s13[1] = '0;
    for (int g = 0; g < 2; g++) begin
      if (psum_group_valid[g]) begin
        case (i_psum_meta.mode)
          MODE_FP19, MODE_FP9:
            exp_s13[g] = $signed({4'b0, emax_u9[g]}) -
                         $signed({5'b0, i_psum_meta.bias_sum}) + 13'sd132 +
                         $signed({{3{scale_sum_s10[g][9]}}, scale_sum_s10[g]});
          MODE_INT8: if (i_psum_meta.mx_enable)
            exp_s13[g] = $signed({{3{scale_sum_s10[g][9]}}, scale_sum_s10[g]}) + 13'sd132;
          default: begin end
        endcase
      end
    end
    // Zero has no exponent dependence, including all-zero BF16 blocks.
    exp0_d = (contribution0_d == 0) ? 8'd0 : exp_s13[0][7:0];
    exp1_d = (contribution1_d == 0) ? 8'd0 : exp_s13[1][7:0];
  end

  assign mx_space = !o_mx_valid || i_mx_ready;
  assign o_psum_ready = i_rstn && mx_space;
  assign psum_fire = i_psum_valid && o_psum_ready;

  assign o_mx_meta = meta_q;
  assign o_mx_tag = tag_q;

  always_ff @(posedge i_clk) begin
    if (!i_rstn) begin
      o_mx_valid <= 1'b0;
      o_mx_contribution0 <= '0;
      o_mx_contribution1 <= '0;
      o_mx_exp0 <= '0;
      o_mx_exp1 <= '0;
      meta_q <= '0;
      tag_q[0] <= '0;
      tag_q[1] <= '0;
    end else begin
      if (mx_space) o_mx_valid <= i_psum_valid;
      if (psum_fire) begin
        o_mx_contribution0 <= contribution0_d;
        o_mx_contribution1 <= contribution1_d;
        o_mx_exp0 <= exp0_d;
        o_mx_exp1 <= exp1_d;
        meta_q.mode <= i_psum_meta.mode;
        meta_q.acc_kind <= i_psum_meta.acc_kind;
        meta_q.group_valid <= i_psum_meta.group_valid;
        meta_q.acc_first <= i_psum_meta.acc_first;
        meta_q.int16_phase <= i_psum_meta.int16_phase;
        meta_q.group_nz <= i_psum_ameta.group_nz;
        meta_q.inexact <= i_psum_ameta.inexact;
        tag_q[0] <= i_psum_tag[0];
        tag_q[1] <= i_psum_tag[1];
      end
    end
  end

`ifndef SYNTHESIS
  initial if (TAG_W < 1) $fatal(1, "TAG_W must be explicit and positive");

  always @(posedge i_clk) begin
    if (psum_fire) begin
      if (i_psum_meta.group_valid == 2'b00 ||
          !((i_psum_meta.mode == MODE_FP19) || (i_psum_meta.mode == MODE_FP9) ||
            (i_psum_meta.mode == MODE_INT16) || (i_psum_meta.mode == MODE_INT8) ||
            (i_psum_meta.mode == MODE_INT4)))
        $fatal(1, "invalid MX mode or group_valid");

      if (is_wide_mode(i_psum_meta.mode) && i_psum_meta.group_valid != 2'b01)
        $fatal(1, "WIDE MX requires group_valid=01");

      if (i_psum_meta.mx_enable &&
          !((i_psum_meta.mode == MODE_FP9) || (i_psum_meta.mode == MODE_INT8)))
        $fatal(1, "MX scale is only defined for FP9 and INT8");

      if (i_psum_meta.acc_kind !=
          acc_kind_for(i_psum_meta.mode, i_psum_meta.mx_enable))
        $fatal(1, "MX acc_kind does not match mode and scale enable");

      if (i_psum_meta.mode == MODE_FP19 &&
          i_psum_data[55:36] !== {20{i_psum_data[35]}})
        $fatal(1, "FP19 PSUM sign extension");

      if (i_psum_meta.mode == MODE_INT16 &&
          i_psum_data[36] !== i_psum_data[35])
        $fatal(1, "INT16 PSUM signed36 redundancy");

      if (i_psum_meta.mode == MODE_INT8) begin
        if (i_psum_meta.group_valid[0] &&
            i_psum_data[27:19] !== {9{i_psum_data[18]}})
          $fatal(1, "INT8 group0 PSUM signed19 redundancy");
        if (i_psum_meta.group_valid[1] &&
            i_psum_data[55:47] !== {9{i_psum_data[46]}})
          $fatal(1, "INT8 group1 PSUM signed19 redundancy");
      end

      if (i_psum_meta.mode == MODE_INT4) begin
        if (i_psum_meta.group_valid[0] &&
            i_psum_data[27:11] !== {17{i_psum_data[10]}})
          $fatal(1, "INT4 group0 PSUM signed12 redundancy");
        if (i_psum_meta.group_valid[1] &&
            i_psum_data[55:39] !== {17{i_psum_data[38]}})
          $fatal(1, "INT4 group1 PSUM signed12 redundancy");
      end

      if (i_psum_meta.mx_enable) begin
        for (int g = 0; g < 2; g++) begin
          if (psum_group_valid[g] &&
              (psum_scale_a[g] == 8'hff ||
               psum_scale_b[g] == 8'hff))
            $fatal(1, "MX active group requires finite E8M0 scales");
        end
      end

      if (i_psum_meta.acc_kind == ACC_TC_FLOAT) begin
        if (contribution0_d != 0 && (exp_s13[0] < 1 || exp_s13[0] > 254))
          $fatal(1, "MX input exponent boundary deferred group0");
        if (contribution1_d != 0 && (exp_s13[1] < 1 || exp_s13[1] > 254))
          $fatal(1, "MX input exponent boundary deferred group1");
      end
    end
  end
`endif
endmodule
