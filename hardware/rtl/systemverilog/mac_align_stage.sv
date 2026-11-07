// ALIGN stage: two per-group MAX trees, 16 lane0 and 16 lane1 physical
// distance/shifter/RNE paths, and one elastic output slot (IP_Design.md
// chapter 6). Numerical outputs keep the full 32/24-bit register widths;
// the FP19 limb split happens later inside PSUM.
module mac_align_stage #(
  parameter integer TAG_W
) (
  input  logic i_clk,
  input  logic i_rstn,
  input  logic i_m_valid,
  output logic o_m_ready,
  input  mac_types_pkg::product12_t i_m_product0 [0:15],
  input  mac_types_pkg::product9_t i_m_product1 [0:15],
  input  logic [8:0] i_m_exp_sum0 [0:15],
  input  logic [5:0] i_m_exp_sum1 [0:15],
  input  logic [1:0] i_m_product_nz [0:15],
  input  mac_types_pkg::txn_meta_t i_m_meta,
  input  logic [TAG_W-1:0] i_m_tag [0:1],

  output logic o_align_valid,
  input  logic i_align_ready,
  output mac_types_pkg::align_lane0_t o_align_lane0 [0:15],
  output mac_types_pkg::align_lane1_t o_align_lane1 [0:15],
  output mac_types_pkg::txn_meta_t o_align_meta,
  output mac_types_pkg::align_meta_t o_align_ameta,
  output logic [TAG_W-1:0] o_align_tag [0:1]
);
  import mac_types_pkg::*;

  logic [15:0] max_nz0, max_nz1;
  logic [8:0] max_exp0;
  logic [5:0] max_exp1;
  logic has_nz0, has_nz1;
  align_lane0_t aligned0 [0:15], lane0_q [0:15];
  align_lane1_t aligned1 [0:15], lane1_q [0:15];
  logic [15:0] inexact0, inexact1;
  logic [15:0] overflow0, overflow1;
  logic lane0_active, lane1_is_active;
  txn_meta_t meta_q;
  align_meta_t ameta_q;
  logic [TAG_W-1:0] tag_q [0:1];
  logic space, fire;

  assign lane0_active = i_m_meta.group_valid[0];
  assign lane1_is_active = lane1_active(i_m_meta.mode, i_m_meta.group_valid);

  assign space = !o_align_valid || i_align_ready;
  assign o_m_ready = i_rstn && space;
  assign fire = i_m_valid && o_m_ready;

  // Zero products never enter MAX; the two groups use independent trees.
  for (genvar i = 0; i < DOT_N; i++) begin : gen_align
    assign max_nz0[i] = lane0_active && is_float_mode(i_m_meta.mode) &&
                        i_m_product_nz[i][0];
    assign max_nz1[i] = lane1_is_active && (i_m_meta.mode == MODE_FP9) &&
                        i_m_product_nz[i][1];
    mac_align_lane0 u_lane0 (
      .i_mode(i_m_meta.mode), .i_active(lane0_active),
      .i_product(i_m_product0[i]), .i_exp_sum(i_m_exp_sum0[i]),
      .i_product_nz(i_m_product_nz[i][0]), .i_emax(max_exp0),
      .o_aligned(aligned0[i]), .o_inexact(inexact0[i]),
      .o_round_overflow(overflow0[i])
    );
    mac_align_lane1 u_lane1 (
      .i_mode(i_m_meta.mode), .i_active(lane1_is_active),
      .i_product(i_m_product1[i]), .i_exp_sum(i_m_exp_sum1[i]),
      .i_product_nz(i_m_product_nz[i][1]), .i_emax(max_exp1),
      .o_aligned(aligned1[i]), .o_inexact(inexact1[i]),
      .o_round_overflow(overflow1[i])
    );

    // Inactive lane registers retain stale bits; the valid/group/mode masks
    // below keep them invisible (same contract as the M output stage).
    assign o_align_lane0[i] =
      (o_align_valid && o_align_meta.group_valid[0]) ? lane0_q[i] : '0;
    assign o_align_lane1[i] =
      (o_align_valid &&
       lane1_active(o_align_meta.mode, o_align_meta.group_valid))
      ? lane1_q[i] : '0;
  end

  mac_align_max16 #(.EXP_W(9)) u_max0 (
    .i_exp(i_m_exp_sum0), .i_nz(max_nz0),
    .o_max_exp(max_exp0), .o_has_nonzero(has_nz0)
  );
  mac_align_max16 #(.EXP_W(6)) u_max1 (
    .i_exp(i_m_exp_sum1), .i_nz(max_nz1),
    .o_max_exp(max_exp1), .o_has_nonzero(has_nz1)
  );

  assign o_align_meta = meta_q;
  assign o_align_ameta = ameta_q;
  assign o_align_tag = tag_q;

  always_ff @(posedge i_clk) begin
    if (!i_rstn) begin
      o_align_valid <= 1'b0;
      meta_q <= '0;
      ameta_q <= '0;
      tag_q[0] <= '0;
      tag_q[1] <= '0;
    end else begin
      if (space) o_align_valid <= i_m_valid;
      if (fire) begin
        meta_q <= i_m_meta;
        ameta_q.emax0 <= max_exp0;
        ameta_q.emax1 <= max_exp1;
        ameta_q.group_nz <= {has_nz1, has_nz0};
        ameta_q.inexact <= {(|inexact1), (|inexact0)};
        tag_q[0] <= i_m_tag[0];
        tag_q[1] <= i_m_tag[1];
        for (int i = 0; i < DOT_N; i++) begin
          if (lane0_active) lane0_q[i] <= aligned0[i];
          if (lane1_is_active) lane1_q[i] <= aligned1[i];
        end
      end
    end
  end

`ifndef SYNTHESIS
  initial if (TAG_W < 1) $fatal(1, "TAG_W must be explicit and positive");
  always @(posedge i_clk) begin
    if (fire) begin
      if (i_m_meta.group_valid == 2'b00 ||
          (is_wide_mode(i_m_meta.mode) && i_m_meta.group_valid != 2'b01) ||
          !((i_m_meta.mode == MODE_FP19) || (i_m_meta.mode == MODE_FP9) ||
            (i_m_meta.mode == MODE_INT16) || (i_m_meta.mode == MODE_INT8) ||
            (i_m_meta.mode == MODE_INT4)))
        $fatal(1, "invalid ALIGN mode or group_valid");
      for (int i = 0; i < DOT_N; i++) begin
        if (i_m_meta.mode == MODE_FP19 &&
            (i_m_product0[i][23] !== i_m_product0[i][22] ||
             i_m_product_nz[i][0] !== (i_m_product0[i] != 0) ||
             (i_m_product_nz[i][0] && overflow0[i]) ||
             (i_m_product_nz[i][0] && max_exp0 < i_m_exp_sum0[i])))
          $fatal(1, "FP19 ALIGN input redundancy/nonzero/distance");
        if (i_m_meta.mode == MODE_FP9) begin
          if (i_m_meta.group_valid[0] &&
              (i_m_product0[i][23:9] !== {15{i_m_product0[i][8]}} ||
               i_m_exp_sum0[i][8:6] != 0 ||
               i_m_product_nz[i][0] !== (i_m_product0[i] != 0) ||
               (i_m_product_nz[i][0] && overflow0[i]) ||
               (i_m_product_nz[i][0] && max_exp0 < i_m_exp_sum0[i])))
            $fatal(1, "FP9 lane0 ALIGN input redundancy/nonzero/distance");
          if (i_m_meta.group_valid[1] &&
              (i_m_product1[i][17:9] !== {9{i_m_product1[i][8]}} ||
               i_m_product_nz[i][1] !== (i_m_product1[i] != 0) ||
               (i_m_product_nz[i][1] && overflow1[i]) ||
               (i_m_product_nz[i][1] && max_exp1 < i_m_exp_sum1[i])))
            $fatal(1, "FP9 lane1 ALIGN input redundancy/nonzero/distance");
          if (i_m_meta.group_valid[0] &&
              aligned0[i][31:24] !== {8{aligned0[i][23]}})
            $fatal(1, "FP9 lane0 ALIGN result sign redundancy");
        end
      end
    end
  end
`endif
endmodule
