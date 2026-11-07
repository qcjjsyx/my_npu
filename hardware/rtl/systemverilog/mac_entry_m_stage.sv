// Converted-operand holding slot followed by one elastic M output slot
// (IP_Design.md 4.5.1, 5.1.3, 5.5). A full INT16 transaction occupies two
// successful M issues, phase0 then phase1, out of the same holding slot.
//
// Possible later layout exploration: the 2xDOT_N mac_operand_pack instances
// could move to their own pipeline boundary once a real broadcast/reuse
// topology exists; that changes the stage contract and is not done here.
module mac_entry_m_stage #(
  parameter integer TAG_W
) (
  input  logic i_clk,
  input  logic i_rstn,
  input  logic i_in_valid,
  output logic o_in_ready,
  input  logic [15:0] i_raw_a [0:mac_types_pkg::DOT_N-1],
  input  logic [15:0] i_raw_b [0:mac_types_pkg::DOT_N-1],
  input  mac_types_pkg::src_fmt_t i_src_fmt_a,
  input  mac_types_pkg::src_fmt_t i_src_fmt_b,
  input  logic [1:0] i_group_valid,
  input  logic [1:0] i_acc_first,
  input  logic [TAG_W-1:0] i_tag [0:1],
  input  logic i_mx_enable,
  input  logic [7:0] i_scale_a [0:1],
  input  logic [7:0] i_scale_b [0:1],

  output logic o_m_valid,
  input  logic i_m_ready,
  output mac_types_pkg::product12_t o_m_product0 [0:mac_types_pkg::DOT_N-1],
  output mac_types_pkg::product9_t o_m_product1 [0:mac_types_pkg::DOT_N-1],
  output logic [8:0] o_m_exp_sum0 [0:mac_types_pkg::DOT_N-1],
  output logic [5:0] o_m_exp_sum1 [0:mac_types_pkg::DOT_N-1],
  output logic [1:0] o_m_product_nz [0:mac_types_pkg::DOT_N-1],
  output mac_types_pkg::txn_meta_t o_m_meta,
  output logic [TAG_W-1:0] o_m_tag [0:1]
);
  import mac_types_pkg::*;

  operand_t converted_a [0:DOT_N-1], converted_b [0:DOT_N-1];
  operand_t slot_a_q [0:DOT_N-1], slot_b_q [0:DOT_N-1];
  product12_t front_product0 [0:DOT_N-1];
  product9_t front_product1 [0:DOT_N-1];
  logic [8:0] front_exp_sum0 [0:DOT_N-1];
  logic [5:0] front_exp_sum1 [0:DOT_N-1];
  logic [1:0] front_product_nz [0:DOT_N-1];
  product12_t product0_q [0:DOT_N-1];
  product9_t product1_q [0:DOT_N-1];
  logic [8:0] exp_sum0_q [0:DOT_N-1];
  logic [5:0] exp_sum1_q [0:DOT_N-1];
  logic [1:0] product_nz_q [0:DOT_N-1];
  logic active0 [0:DOT_N-1], active1 [0:DOT_N-1];

  logic slot_valid_q, phase_q;
  txn_meta_t in_meta;
  txn_meta_t slot_meta_q;
  txn_meta_t m_meta_q;
  logic [TAG_W-1:0] slot_tag_q [0:1];
  logic [TAG_W-1:0] m_tag_q [0:1];
  logic m_space, issue, release_slot, in_fire;

  // Assemble the input transaction metadata once; the holding slot and the
  // M register copy it as one unit.
  always_comb begin
    in_meta.mode = mode_for(i_src_fmt_a);
    in_meta.acc_kind = acc_kind_for(mode_for(i_src_fmt_a), i_mx_enable);
    in_meta.bias_sum = bias_sum_for(i_src_fmt_a, i_src_fmt_b);
    in_meta.group_valid = i_group_valid;
    in_meta.acc_first = i_acc_first;
    in_meta.mx_enable = i_mx_enable;
    in_meta.scale_a[0] = i_scale_a[0];
    in_meta.scale_a[1] = i_scale_a[1];
    in_meta.scale_b[0] = i_scale_b[0];
    in_meta.scale_b[1] = i_scale_b[1];
    in_meta.int16_phase = 1'b0;
  end

  assign m_space = !o_m_valid || i_m_ready;
  assign issue = i_rstn && slot_valid_q && m_space;
  // INT16 issues twice from the same slot; everything else releases at once.
  assign release_slot = issue &&
                        ((slot_meta_q.mode != MODE_INT16) || phase_q);
  assign o_in_ready = i_rstn && (!slot_valid_q || release_slot);
  assign in_fire = i_in_valid && o_in_ready;

  for (genvar g = 0; g < DOT_N; g++) begin : gen_frontends
    mac_operand_pack u_pack_a (
      .i_raw(i_raw_a[g]), .i_src_fmt(i_src_fmt_a),
      .i_group_valid(i_group_valid), .o_payload(converted_a[g])
    );
    mac_operand_pack u_pack_b (
      .i_raw(i_raw_b[g]), .i_src_fmt(i_src_fmt_b),
      .i_group_valid(i_group_valid), .o_payload(converted_b[g])
    );
    mac_mul_frontend u_mul (
      .i_operand_a(slot_a_q[g]), .i_operand_b(slot_b_q[g]),
      .i_mode(slot_meta_q.mode), .i_group_valid(slot_meta_q.group_valid),
      .i_int16_phase(phase_q),
      .o_product0(front_product0[g]), .o_product1(front_product1[g]),
      .o_exp_sum0(front_exp_sum0[g]), .o_exp_sum1(front_exp_sum1[g]),
      .o_product_nz(front_product_nz[g]),
      .o_product0_active(active0[g]),
      .o_product1_active(active1[g])
    );

    // Invalid physical result registers retain their previous bits. The
    // valid/group/mode output masks below guarantee no stale product,
    // exponent or flag becomes visible as a numerical operand (section 2.6).
    assign o_m_product0[g] = (o_m_valid && o_m_meta.group_valid[0])
                              ? product0_q[g] : '0;
    assign o_m_product1[g] =
      (o_m_valid && lane1_active(o_m_meta.mode, o_m_meta.group_valid))
      ? product1_q[g] : '0;
    assign o_m_exp_sum0[g] =
      (o_m_valid && is_float_mode(o_m_meta.mode)) ? exp_sum0_q[g] : '0;
    assign o_m_exp_sum1[g] =
      (o_m_valid && is_float_mode(o_m_meta.mode)) ? exp_sum1_q[g] : '0;
    assign o_m_product_nz[g] =
      (o_m_valid && is_float_mode(o_m_meta.mode)) ? product_nz_q[g] : '0;
  end

  assign o_m_meta = m_meta_q;
  assign o_m_tag = m_tag_q;

  always_ff @(posedge i_clk) begin
    if (!i_rstn) begin
      slot_valid_q <= 1'b0;
      phase_q <= 1'b0;
      o_m_valid <= 1'b0;
      m_meta_q <= '0;
      m_tag_q[0] <= '0;
      m_tag_q[1] <= '0;
    end else begin
      if (m_space) o_m_valid <= issue;

      if (issue) begin
        // phase1 never reapplies acc_first (section 5.5).
        m_meta_q <= slot_meta_q;
        m_meta_q.acc_first <=
          (slot_meta_q.mode == MODE_INT16 && phase_q)
            ? 2'b00 : slot_meta_q.acc_first;
        m_meta_q.int16_phase <= (slot_meta_q.mode == MODE_INT16) && phase_q;
        m_tag_q[0] <= slot_tag_q[0];
        m_tag_q[1] <= slot_tag_q[1];
        for (int i = 0; i < DOT_N; i++) begin
          if (active0[i]) product0_q[i] <= front_product0[i];
          if (active1[i]) product1_q[i] <= front_product1[i];
          // Exponent/nz registers stop updating in integer modes; the
          // mode-based output masks above isolate their stale contents.
          if (is_float_mode(slot_meta_q.mode)) begin
            exp_sum0_q[i] <= front_exp_sum0[i];
            exp_sum1_q[i] <= front_exp_sum1[i];
            product_nz_q[i] <= front_product_nz[i];
          end
        end
      end

      if (in_fire) begin
        slot_valid_q <= 1'b1;
        phase_q <= 1'b0;
        slot_meta_q <= in_meta;
        slot_tag_q[0] <= i_tag[0];
        slot_tag_q[1] <= i_tag[1];
        for (int i = 0; i < DOT_N; i++) begin
          slot_a_q[i] <= converted_a[i];
          slot_b_q[i] <= converted_b[i];
        end
      end else if (release_slot) begin
        slot_valid_q <= 1'b0;
      end else if (issue && slot_meta_q.mode == MODE_INT16 && !phase_q) begin
        phase_q <= 1'b1;
      end
    end
  end

`ifndef SYNTHESIS
  // These conditions have no specified valid-transaction numerical fallback.
  // Check them only at the external input handshake, not on idle bus values.
  initial if (TAG_W < 1) $fatal(1, "TAG_W must be explicit and positive");
  always @(posedge i_clk) begin
    if (i_rstn && in_fire) begin
      if (!legal_src_pair(i_src_fmt_a, i_src_fmt_b) ||
          !legal_mx_enable(i_src_fmt_a, i_src_fmt_b, i_mx_enable))
        $fatal(1, "illegal source-format combination or MX enable");
      if (i_group_valid == 2'b00 ||
          (is_wide_mode(mode_for(i_src_fmt_a)) && i_group_valid != 2'b01))
        $fatal(1, "invalid group_valid");
      for (int group = 0; group < 2; group++) begin
        if (i_mx_enable && i_group_valid[group] &&
            (i_scale_a[group] == 8'hff || i_scale_b[group] == 8'hff))
          $fatal(1, "E8M0 255 has no defined valid-transaction policy");
      end
      for (int i = 0; i < DOT_N; i++) begin
        case (i_src_fmt_a)
          SRC_INT16:
            if (i_raw_a[i] == 16'h8000 || i_raw_b[i] == 16'h8000)
              $fatal(1, "reserved INT16 input");
          SRC_INT8: begin
            if ((i_group_valid[0] &&
                 (i_raw_a[i][7:0] == 8'h80 || i_raw_b[i][7:0] == 8'h80)) ||
                (i_group_valid[1] &&
                 (i_raw_a[i][15:8] == 8'h80 || i_raw_b[i][15:8] == 8'h80)))
              $fatal(1, "reserved INT8 input");
          end
          SRC_INT4: begin
            if (i_raw_a[i][15:8] != 8'd0 || i_raw_b[i][15:8] != 8'd0)
              $fatal(1, "INT4 raw upper byte must be zero");
            if ((i_group_valid[0] &&
                 (i_raw_a[i][3:0] == 4'h8 || i_raw_b[i][3:0] == 4'h8)) ||
                (i_group_valid[1] &&
                 (i_raw_a[i][7:4] == 4'h8 || i_raw_b[i][7:4] == 4'h8)))
              $fatal(1, "reserved INT4 input");
          end
          default: begin end
        endcase
      end
    end
  end
`endif
endmodule
