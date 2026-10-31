// Q2.34 ACC: exact input normalization, three tail work bits, one final RNE.
// State: signed36 M, unsigned8 bias127 E. Zero M=0/E=0; infinity M=0/E=255
// with explicit signed2 infinity_sign (+1/-1). No NaN input is defined.
// The update sink keeps the existing one-accept/clock, no-result-ready API.
module mac_acc_stage #(parameter integer TAG_W) (
  input logic i_clk, i_rstn, i_mx_valid,
  output logic o_mx_ready,
  input mac_types_pkg::contribution0_t i_mx_contribution0,
  input mac_types_pkg::contribution1_t i_mx_contribution1,
  input mac_types_pkg::acc_exp_t i_mx_exp0, i_mx_exp1,
  input mac_types_pkg::mx_meta_t i_mx_meta,
  input logic [TAG_W-1:0] i_mx_tag [0:1],
  output logic o_acc_update_valid,
  output logic [1:0] o_acc_update_group_valid,
  output mac_types_pkg::acc_kind_t o_acc_update_kind,
  output logic [TAG_W-1:0] o_acc_update_tag [0:1],
  output logic [1:0] o_acc_state_valid,
  output mac_types_pkg::acc_kind_t o_acc_state_kind [0:1],
  output logic signed [31:0] o_acc_int32 [0:1],
  output mac_types_pkg::acc_sig_t o_acc_float_sig [0:1],
  output mac_types_pkg::acc_exp_t o_acc_float_exp [0:1],
  output logic signed [1:0] o_acc_float_inf_sign [0:1]
);
  import mac_types_pkg::*;
  logic mx_fire, int16_pending_q;
  int_work_t int16_mid_q;
  logic [1:0] mx_group_valid, mx_acc_first;
  logic signed [35:0] float_input [0:1];
  acc_exp_t input_exp [0:1];
  logic signed [38:0] float_old_op [0:1], float_new_op [0:1];
  logic signed [10:0] float_window_exp [0:1];
  logic signed [38:0] adder_a [0:1], adder_b [0:1];
  logic signed [39:0] adder_sum [0:1];
  logic signed [36:0] adder_sum_s37 [0:1];
  acc_sig_t float_norm_sig_d [0:1];
  acc_exp_t float_norm_exp_d [0:1];
  logic signed [1:0] float_inf_d [0:1];
  logic signed [31:0] int_sat_d [0:1];
  int_work_t int_old_s36 [0:1], int_input_s36 [0:1];
`ifndef SYNTHESIS
  compute_mode_t acc_mode_q [0:1];
`endif
  assign o_mx_ready = i_rstn;
  assign mx_fire = i_mx_valid && o_mx_ready;
  assign mx_group_valid = i_mx_meta.group_valid;
  assign mx_acc_first = i_mx_meta.acc_first;
  assign float_input[0] = i_mx_meta.acc_kind == ACC_TC_FLOAT ? i_mx_contribution0 : '0;
  assign float_input[1] = i_mx_meta.acc_kind == ACC_TC_FLOAT ? i_mx_contribution1 : '0;
  assign input_exp[0] = i_mx_exp0;
  assign input_exp[1] = i_mx_exp1;

  for (genvar g=0; g<2; g=g+1) begin : gen_float
    mac_acc_float_prepare36 u_prepare (
      .i_first(mx_acc_first[g]), .i_old_sig(o_acc_float_sig[g]),
      .i_old_exp(o_acc_float_exp[g]), .i_new_sig(float_input[g]),
      .i_new_exp(input_exp[g]), .o_old_operand(float_old_op[g]),
      .o_new_operand(float_new_op[g]), .o_window_exp(float_window_exp[g])
    );
    // 36+36 ->37 core bits, with three appended low work bits (39 ->40).
    // Integer operands have three zero low bits; the same adder is shared.
    assign adder_sum[g] = $signed({adder_a[g][38],adder_a[g]}) +
                          $signed({adder_b[g][38],adder_b[g]});
    assign adder_sum_s37[g] = adder_sum[g][39:3];
  end

  always @* begin
    for (integer g=0; g<2; g=g+1) begin
      int_input_s36[g] = g == 0 ? i_mx_contribution0 : i_mx_contribution1;
      int_old_s36[g] = '0;
      if (i_mx_meta.mode == MODE_INT16 && i_mx_meta.int16_phase)
        int_old_s36[g] = int16_mid_q;
      else if (!mx_acc_first[g]) int_old_s36[g] = $signed(o_acc_int32[g]);
      adder_a[g] = i_mx_meta.acc_kind == ACC_TC_FLOAT ? float_old_op[g] : {int_old_s36[g],3'b0};
      adder_b[g] = i_mx_meta.acc_kind == ACC_TC_FLOAT ? float_new_op[g] : {int_input_s36[g],3'b0};
      if (adder_sum_s37[g] > 37'sd2147483647) int_sat_d[g] = 32'sh7fffffff;
      else if (adder_sum_s37[g] < -37'sd2147483648) int_sat_d[g] = 32'sh80000000;
      else int_sat_d[g] = adder_sum_s37[g][31:0];
    end
  end

  // Normalize and round the signed sum once. The signed2 Inf field is
  // separate so neither sign is confused with the canonical zero payload.
  for (genvar g=0; g<2; g=g+1) begin : gen_round
    logic [39:0] magnitude, rounded, lost;
    integer lead, shift, exponent;
    always @* begin
      magnitude = adder_sum[g][39] ? (~$unsigned(adder_sum[g]) + 40'd1) : $unsigned(adder_sum[g]);
      rounded = '0;
      lost = '0;
      lead = 0;
      shift = 0;
      exponent = 0;
      float_norm_sig_d[g] = '0;
      float_norm_exp_d[g] = '0;
      float_inf_d[g] = '0;
      if (!mx_acc_first[g] && o_acc_float_inf_sign[g] != 0) begin
        float_norm_exp_d[g] = 8'd255;
        float_inf_d[g] = o_acc_float_inf_sign[g];
      end else if (magnitude != 0) begin
        for (integer j=0; j<40; j=j+1) if (magnitude[j]) lead = j;
        shift = lead - 34;
        exponent = $signed(float_window_exp[g]) + shift - 3;
        if (shift > 0) begin
          rounded = magnitude >> shift;
          lost = magnitude << (40-shift);
          if (lost[39] && ((|lost[38:0]) || rounded[0])) rounded = rounded + 40'd1;
        end else rounded = magnitude << (-shift);
        if (rounded[35]) begin
          rounded = rounded >> 1;
          exponent = exponent + 1;
        end
        if (exponent > 254) begin
          float_norm_exp_d[g] = 8'd255;
          float_inf_d[g] = adder_sum[g][39] ? -2'sd1 : 2'sd1;
        end else if (exponent >= 1) begin
          float_norm_sig_d[g] = adder_sum[g][39] ? -$signed(rounded[35:0]) : $signed(rounded[35:0]);
          float_norm_exp_d[g] = exponent;
        end // exponent <1: FTZ, canonical M=0/E=0.
      end
    end
  end

  // Phase 6: commit one accepted update and its metadata at the clock edge.
  always_ff @(posedge i_clk) begin
    if (!i_rstn) begin
      o_acc_update_valid <= 1'b0;
      o_acc_update_group_valid <= '0;
      o_acc_update_kind <= ACC_INT32;
      o_acc_state_valid <= '0;
      int16_pending_q <= 1'b0;
      int16_mid_q <= '0;
      for (int k = 0; k < 2; k = k + 1) begin
        o_acc_state_kind[k] <= ACC_INT32;
        o_acc_int32[k] <= '0;
        o_acc_float_sig[k] <= '0;
        o_acc_float_exp[k] <= '0;
        o_acc_float_inf_sign[k] <= '0;
        o_acc_update_tag[k] <= '0;
`ifndef SYNTHESIS
        acc_mode_q[k] <= MODE_FP19;
`endif
      end
    end else begin
      o_acc_update_valid <= 1'b0;
      o_acc_update_group_valid <= '0;
      if (mx_fire) begin
        if (!(i_mx_meta.mode == MODE_INT16 && !i_mx_meta.int16_phase)) begin
          o_acc_update_valid <= 1'b1;
          o_acc_update_group_valid <= i_mx_meta.group_valid;
          o_acc_update_kind <= i_mx_meta.acc_kind;
          for (int k = 0; k < 2; k = k + 1)
            o_acc_update_tag[k] <= i_mx_tag[k];
        end

        if (i_mx_meta.mode == MODE_INT16) begin
          if (!i_mx_meta.int16_phase) begin
            int16_mid_q <= adder_sum_s37[0][35:0];
            int16_pending_q <= 1'b1;
`ifndef SYNTHESIS
            acc_mode_q[0] <= MODE_INT16;
`endif
          end else begin
            o_acc_int32[0] <= int_sat_d[0];
            o_acc_state_kind[0] <= ACC_INT32;
            o_acc_state_valid[0] <= 1'b1;
            int16_mid_q <= {{4{int_sat_d[0][31]}}, int_sat_d[0]};
            int16_pending_q <= 1'b0;
`ifndef SYNTHESIS
            acc_mode_q[0] <= MODE_INT16;
`endif
          end
        end else begin
          for (int k = 0; k < 2; k = k + 1) begin
            if (mx_group_valid[k]) begin
              o_acc_state_valid[k] <= 1'b1;
              o_acc_state_kind[k] <= i_mx_meta.acc_kind;
              if (i_mx_meta.acc_kind == ACC_TC_FLOAT) begin
                o_acc_float_sig[k] <= float_norm_sig_d[k];
                o_acc_float_exp[k] <= float_norm_exp_d[k];
                o_acc_float_inf_sign[k] <= float_inf_d[k];
              end else begin
                o_acc_int32[k] <= int_sat_d[k];
              end
`ifndef SYNTHESIS
              acc_mode_q[k] <= i_mx_meta.mode;
`endif
            end
          end
        end
      end
    end
  end

`ifndef SYNTHESIS
  initial if (TAG_W < 1) $fatal(1, "TAG_W must be explicit and positive");

  always @(posedge i_clk) begin
    if (mx_fire) begin
      if (i_mx_meta.group_valid == 2'b00)
        $fatal(1, "ACC requires at least one valid group");
      if (!((i_mx_meta.mode == MODE_FP19) || (i_mx_meta.mode == MODE_FP9) ||
            (i_mx_meta.mode == MODE_INT16) || (i_mx_meta.mode == MODE_INT8) ||
            (i_mx_meta.mode == MODE_INT4)))
        $fatal(1, "ACC received an illegal mode");
      if ((i_mx_meta.mode == MODE_FP19 || i_mx_meta.mode == MODE_FP9) &&
          i_mx_meta.acc_kind != ACC_TC_FLOAT)
        $fatal(1, "floating mode requires TC_FLOAT");
      if ((i_mx_meta.mode == MODE_INT16 || i_mx_meta.mode == MODE_INT4) &&
          i_mx_meta.acc_kind != ACC_INT32)
        $fatal(1, "integer mode requires INT32");
      if (i_mx_meta.acc_kind == ACC_TC_FLOAT &&
          !((i_mx_meta.mode == MODE_FP19) || (i_mx_meta.mode == MODE_FP9) ||
            (i_mx_meta.mode == MODE_INT8)))
        $fatal(1, "TC_FLOAT is not defined for this mode");
      if ((i_mx_meta.mode == MODE_FP19 || i_mx_meta.mode == MODE_INT16) &&
          i_mx_meta.group_valid != 2'b01)
        $fatal(1, "wide ACC mode requires group_valid=01");

      if (i_mx_meta.acc_kind == ACC_INT32 && i_mx_meta.mode == MODE_INT8) begin
        if (mx_group_valid[0] && i_mx_contribution0[35:19] !== {17{i_mx_contribution0[18]}})
          $fatal(1, "INT8 group0 contribution must be signed19");
        if (mx_group_valid[1] && i_mx_contribution1[35:19] !== {17{i_mx_contribution1[18]}})
          $fatal(1, "INT8 group1 contribution must be signed19");
      end
      if (i_mx_meta.mode == MODE_INT4) begin
        if (mx_group_valid[0] && i_mx_contribution0[35:11] !== {25{i_mx_contribution0[10]}})
          $fatal(1, "INT4 group0 contribution must be signed12");
        if (mx_group_valid[1] && i_mx_contribution1[35:11] !== {25{i_mx_contribution1[10]}})
          $fatal(1, "INT4 group1 contribution must be signed12");
      end

      if (i_mx_meta.mode == MODE_INT16) begin
        if (!i_mx_meta.int16_phase && int16_pending_q)
          $fatal(1, "INT16 phase0 arrived before previous phase1");
        if (i_mx_meta.int16_phase && !int16_pending_q)
          $fatal(1, "INT16 phase1 arrived without phase0");
        if (i_mx_meta.int16_phase && i_mx_meta.acc_first != 2'b00)
          $fatal(1, "INT16 phase1 must not reapply acc_first");
        if (!i_mx_meta.int16_phase && !i_mx_meta.acc_first[0] &&
            (!o_acc_state_valid[0] || o_acc_state_kind[0] != ACC_INT32 ||
             acc_mode_q[0] != MODE_INT16))
          $fatal(1, "INT16 phase0 requires acc_first for an invalid/noninteger state");
      end else begin
        if (int16_pending_q)
          $fatal(1, "non-INT16 update interleaved between INT16 phases");
        for (int k = 0; k < 2; k = k + 1) begin
          if (mx_group_valid[k] && !mx_acc_first[k] &&
              (!o_acc_state_valid[k] || o_acc_state_kind[k] != i_mx_meta.acc_kind ||
               acc_mode_q[k] != i_mx_meta.mode))
            $fatal(1, "ACC state requires acc_first before mode/type change or initial use");
        end
      end

      if (i_mx_meta.acc_kind == ACC_INT32) begin
        if (i_mx_meta.group_valid[0] &&
            adder_sum_s37[0][36] !== adder_sum_s37[0][35])
          $fatal(1, "integer group0 sum exceeds signed36 work range");
        if (i_mx_meta.group_valid[1] &&
            adder_sum_s37[1][36] !== adder_sum_s37[1][35])
          $fatal(1, "integer group1 sum exceeds signed36 work range");
      end

      if (i_mx_meta.acc_kind == ACC_TC_FLOAT) begin
        for (int k=0; k<2; k=k+1)
          if (mx_group_valid[k] && float_input[k] != 0 &&
              (input_exp[k] == 0 || input_exp[k] == 255))
            $fatal(1, "ACC nonzero input exponent boundary deferred");
      end
    end
  end
`endif
endmodule
