// PSUM stage (IP_Design.md chapter 7): mode-adapt the 16 lane0/lane1 terms
// into the two physical 16-to-2 DesignWare reduction trees, finish each
// tree with one 28-bit carry-propagate adder, merge the wide-mode limbs in
// the explicit merge CPA, and pack everything into the single elastic
// 56-bit output slot.
//
// The two trees are pure carry-save networks with no cross-tree carry; the
// only carry propagation is each tree's final DW01_add plus the wide merge.
module mac_psum_stage #(
  parameter integer TAG_W
) (
  input  logic i_clk,
  input  logic i_rstn,
  input  logic i_align_valid,
  output logic o_align_ready,
  input  mac_types_pkg::align_lane0_t i_align_lane0 [0:15],
  input  mac_types_pkg::align_lane1_t i_align_lane1 [0:15],
  input  mac_types_pkg::txn_meta_t i_align_meta,
  input  mac_types_pkg::align_meta_t i_align_ameta,
  input  logic [TAG_W-1:0] i_align_tag [0:1],

  output logic o_psum_valid,
  input  logic i_psum_ready,
  output mac_types_pkg::psum_packed_t o_psum_data,
  output mac_types_pkg::txn_meta_t o_psum_meta,
  output mac_types_pkg::align_meta_t o_psum_ameta,
  output logic [TAG_W-1:0] o_psum_tag [0:1]
);
  import mac_types_pkg::*;

  localparam integer TREE_W = 28;
  localparam integer TREE_INPUTS = DOT_N;
  localparam integer TREE_BUS_W = TREE_W * TREE_INPUTS;

  logic [TREE_BUS_W-1:0] tree0_input;
  logic [TREE_BUS_W-1:0] tree1_input;
  logic signed [23:0] tree0_term_s24 [0:DOT_N-1];
  logic signed [23:0] tree1_term_s24 [0:DOT_N-1];
  logic signed [27:0] tree0_term_s28 [0:DOT_N-1];
  logic signed [27:0] tree1_term_s28 [0:DOT_N-1];
  logic [27:0] tree0_csa_sum;
  logic [27:0] tree0_csa_carry;
  logic [27:0] tree1_csa_sum;
  logic [27:0] tree1_csa_carry;
  logic [27:0] tree0_sum_bits;
  logic [27:0] tree1_sum_bits;
  logic tree0_carry_out;
  logic tree1_carry_out;
  logic signed [27:0] tree0_sum_s28;
  logic signed [27:0] tree1_sum_s28;

  logic signed [36:0] fp19_low_s37;
  logic signed [36:0] fp19_high_s37;
  logic signed [36:0] fp19_sum_s37;
  logic signed [35:0] fp19_result_s36;
  logic signed [36:0] int16_low_s37;
  logic signed [36:0] int16_high_s37;
  logic signed [36:0] int16_phase_sum_s37;
  logic signed [36:0] int16_weighted_s37;
  psum_packed_t psum_data_d;
  txn_meta_t meta_q;
  align_meta_t ameta_q;
  logic [TAG_W-1:0] tag_q [0:1];
  logic psum_space;
  logic psum_fire;

  assign tree0_sum_s28 = $signed(tree0_sum_bits);
  assign tree1_sum_s28 = $signed(tree1_sum_bits);

  // Mode adaptation (sections 7.1-7.4). The semantic tree terms are
  // signed24; each is sign-extended to the signed28 physical tree input so
  // every 16-term sum fits without modular overflow. FP19 splits each
  // signed32 aligned value into an unsigned low limb and a signed high
  // limb; INT16 routes the two partial-product limbs; the DUAL modes pass
  // each group's terms to its own tree.
  always_comb begin
    tree0_input = '0;
    tree1_input = '0;
    for (int i = 0; i < DOT_N; i++) begin
      tree0_term_s24[i] = '0;
      tree1_term_s24[i] = '0;

      case (i_align_meta.mode)
        MODE_FP19: begin
          if (i_align_meta.group_valid[0]) begin
            // X = (signed high limb << 23) + unsigned low limb.
            tree0_term_s24[i] = $signed({1'b0, i_align_lane0[i][22:0]});
            tree1_term_s24[i] = $signed({{15{i_align_lane0[i][31]}},
                                         i_align_lane0[i][31:23]});
          end
        end
        MODE_INT16: begin
          if (i_align_meta.group_valid[0]) begin
            tree0_term_s24[i] = $signed(i_align_lane0[i][23:0]);
            tree1_term_s24[i] = $signed(i_align_lane1[i]);
          end
        end
        MODE_FP9, MODE_INT8, MODE_INT4: begin
          if (i_align_meta.group_valid[0])
            tree0_term_s24[i] = $signed(i_align_lane0[i][23:0]);
          if (i_align_meta.group_valid[1])
            tree1_term_s24[i] = $signed(i_align_lane1[i]);
        end
        default: begin end
      endcase

      tree0_term_s28[i] = {{4{tree0_term_s24[i][23]}}, tree0_term_s24[i]};
      tree1_term_s28[i] = {{4{tree1_term_s24[i][23]}}, tree1_term_s24[i]};
      tree0_input[i* TREE_W +: TREE_W] = tree0_term_s28[i];
      tree1_input[i* TREE_W +: TREE_W] = tree1_term_s28[i];
    end
  end

  // DW02_tree is a 3:2 compressor. DW01_add performs the only carry
  // propagation in each independent 16-input reduction tree.
  DW02_tree #(TREE_INPUTS, TREE_W, 1) u_tree0 (
    .INPUT(tree0_input),
    .OUT0(tree0_csa_sum),
    .OUT1(tree0_csa_carry)
  );

  DW01_add #(TREE_W) u_tree0_final_add (
    .A(tree0_csa_sum),
    .B(tree0_csa_carry),
    .CI(1'b0),
    .SUM(tree0_sum_bits),
    .CO(tree0_carry_out)
  );

  DW02_tree #(TREE_INPUTS, TREE_W, 1) u_tree1 (
    .INPUT(tree1_input),
    .OUT0(tree1_csa_sum),
    .OUT1(tree1_csa_carry)
  );

  DW01_add #(TREE_W) u_tree1_final_add (
    .A(tree1_csa_sum),
    .B(tree1_csa_carry),
    .CI(1'b0),
    .SUM(tree1_sum_bits),
    .CO(tree1_carry_out)
  );

  // Wide-mode merges stay in PSUM (sections 7.4, 7.7). Each merge is
  // computed only in its own mode so the wide adders stay quiet otherwise.
  // Temporary widths are explicit so limb shifts cannot discard significant
  // bits before the addition.
  always_comb begin
    fp19_low_s37 = '0;
    fp19_high_s37 = '0;
    fp19_sum_s37 = '0;
    fp19_result_s36 = '0;
    int16_low_s37 = '0;
    int16_high_s37 = '0;
    int16_phase_sum_s37 = '0;
    int16_weighted_s37 = '0;
    psum_data_d = '0;

    case (i_align_meta.mode)
      MODE_FP19: begin
        // S36 = (H << 23) + L with the full unsigned27 low-tree result;
        // the low-tree growth bits [26:23] are never dropped (7.4.1/7.4.3).
        fp19_low_s37 = $signed({10'b0, tree0_sum_bits[26:0]});
        fp19_high_s37 = $signed({{24{tree1_sum_bits[12]}},
                                 tree1_sum_bits[12:0]});
        fp19_sum_s37 = (fp19_high_s37 <<< 23) + fp19_low_s37;
        fp19_result_s36 = $signed(fp19_sum_s37[35:0]);
        psum_data_d = {{20{fp19_result_s36[35]}}, fp19_result_s36};
      end
      MODE_INT16: begin
        // T = L + (H << 8); the phase1 overall 2^8 weight is applied here
        // exactly once, never again downstream (7.4.2/7.4.4).
        int16_low_s37 = {{9{tree0_sum_s28[27]}}, tree0_sum_s28};
        int16_high_s37 = {{9{tree1_sum_s28[27]}}, tree1_sum_s28};
        int16_phase_sum_s37 = int16_low_s37 + (int16_high_s37 <<< 8);
        if (i_align_meta.int16_phase)
          int16_weighted_s37 = int16_phase_sum_s37 <<< 8;
        else
          int16_weighted_s37 = int16_phase_sum_s37;
        psum_data_d = {{19{int16_weighted_s37[36]}}, int16_weighted_s37};
      end
      MODE_FP9, MODE_INT8, MODE_INT4: begin
        // DUAL packing: [27:0] group0, [55:28] group1 (7.7).
        psum_data_d = {tree1_sum_bits, tree0_sum_bits};
      end
      default: begin end
    endcase
  end

  assign psum_space = !o_psum_valid || i_psum_ready;
  assign o_align_ready = i_rstn && psum_space;
  assign psum_fire = i_align_valid && o_align_ready;

  assign o_psum_meta = meta_q;
  assign o_psum_ameta = ameta_q;
  assign o_psum_tag = tag_q;

  always_ff @(posedge i_clk) begin
    if (!i_rstn) begin
      o_psum_valid <= 1'b0;
      o_psum_data <= '0;
      meta_q <= '0;
      ameta_q <= '0;
      tag_q[0] <= '0;
      tag_q[1] <= '0;
    end else begin
      if (psum_space) o_psum_valid <= i_align_valid;
      if (psum_fire) begin
        o_psum_data <= psum_data_d;
        meta_q <= i_align_meta;
        ameta_q <= i_align_ameta;
        tag_q[0] <= i_align_tag[0];
        tag_q[1] <= i_align_tag[1];
      end
    end
  end

`ifndef SYNTHESIS
  initial if (TAG_W < 1) $fatal(1, "TAG_W must be explicit and positive");

  always @(posedge i_clk) begin
    if (psum_fire) begin
      if (i_align_meta.group_valid == 2'b00 ||
          !((i_align_meta.mode == MODE_FP19) || (i_align_meta.mode == MODE_FP9) ||
            (i_align_meta.mode == MODE_INT16) || (i_align_meta.mode == MODE_INT8) ||
            (i_align_meta.mode == MODE_INT4)))
        $fatal(1, "invalid PSUM mode or group_valid");

      if (is_wide_mode(i_align_meta.mode) && i_align_meta.group_valid != 2'b01)
        $fatal(1, "WIDE PSUM requires group_valid=01");

      for (int i = 0; i < DOT_N; i++) begin
        if (i_align_meta.mode != MODE_FP19 &&
            i_align_meta.group_valid[0] &&
            i_align_lane0[i][31:24] !== {8{i_align_lane0[i][23]}})
          $fatal(1, "PSUM lane0 signed24 input redundancy");

        if (i_align_meta.mode == MODE_INT8 && i_align_meta.group_valid[0] &&
            i_align_lane0[i][23:15] !== {9{i_align_lane0[i][14]}})
          $fatal(1, "INT8 PSUM lane0 signed15 input redundancy");
        if (i_align_meta.mode == MODE_INT8 && i_align_meta.group_valid[1] &&
            i_align_lane1[i][23:15] !== {9{i_align_lane1[i][14]}})
          $fatal(1, "INT8 PSUM lane1 signed15 input redundancy");

        if (i_align_meta.mode == MODE_INT4 && i_align_meta.group_valid[0] &&
            i_align_lane0[i][23:7] !== {17{i_align_lane0[i][6]}})
          $fatal(1, "INT4 PSUM lane0 signed7 input redundancy");
        if (i_align_meta.mode == MODE_INT4 && i_align_meta.group_valid[1] &&
            i_align_lane1[i][23:7] !== {17{i_align_lane1[i][6]}})
          $fatal(1, "INT4 PSUM lane1 signed7 input redundancy");

        if (i_align_meta.mode == MODE_INT16 && i_align_meta.group_valid[0]) begin
          if (!i_align_meta.int16_phase &&
              i_align_lane0[i][23:16] !== 8'b0)
            $fatal(1, "INT16 phase0 low-limb input must be unsigned16");
          if (i_align_lane1[i][23:16] !== {8{i_align_lane1[i][15]}})
            $fatal(1, "INT16 high-limb input signed16 redundancy");
          if (i_align_meta.int16_phase &&
              i_align_lane0[i][23:16] !== {8{i_align_lane0[i][15]}})
            $fatal(1, "INT16 phase1 low-limb input signed16 redundancy");
        end
      end

      case (i_align_meta.mode)
        MODE_FP19: begin
          if (tree0_sum_bits[27] !== 1'b0)
            $fatal(1, "FP19 low tree is not unsigned27");
          if (tree1_sum_bits[27:13] !== {15{tree1_sum_bits[12]}})
            $fatal(1, "FP19 high tree signed13 redundancy");
          if (fp19_sum_s37[36] !== fp19_sum_s37[35])
            $fatal(1, "FP19 reconstructed signed36 overflow");
        end
        MODE_INT16: begin
          if (!i_align_meta.int16_phase) begin
            if (tree0_sum_bits[27:20] !== 8'b0)
              $fatal(1, "INT16 phase0 low tree unsigned20 overflow");
          end else if (tree0_sum_bits[27:20] !== {8{tree0_sum_bits[19]}}) begin
            $fatal(1, "INT16 phase1 low tree signed20 redundancy");
          end
          if (tree1_sum_bits[27:20] !== {8{tree1_sum_bits[19]}})
            $fatal(1, "INT16 high tree signed20 redundancy");
          if (!i_align_meta.int16_phase &&
              int16_phase_sum_s37[36:28] !== {9{int16_phase_sum_s37[27]}})
            $fatal(1, "INT16 phase0 result signed28 redundancy");
          if (i_align_meta.int16_phase &&
              int16_phase_sum_s37[36:27] !== {10{int16_phase_sum_s37[26]}})
            $fatal(1, "INT16 phase1 result signed27 redundancy");
        end
        MODE_INT8: begin
          if (tree0_sum_bits[27:19] !== {9{tree0_sum_bits[18]}} ||
              tree1_sum_bits[27:19] !== {9{tree1_sum_bits[18]}})
            $fatal(1, "INT8 PSUM signed19 tree overflow");
        end
        MODE_INT4: begin
          if (tree0_sum_bits[27:11] !== {17{tree0_sum_bits[10]}} ||
              tree1_sum_bits[27:11] !== {17{tree1_sum_bits[10]}})
            $fatal(1, "INT4 PSUM signed11 tree overflow");
        end
        default: begin end
      endcase
    end
  end
`endif
endmodule
