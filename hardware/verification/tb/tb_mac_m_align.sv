module tb_mac_m_align;
  import mac_types_pkg::*;
  logic i_clk = 1'b0;
  always #5 i_clk = ~i_clk;
  logic i_rstn, i_in_valid, o_in_ready, o_m_valid, o_m_ready;
  logic o_align_valid, i_align_ready;
  logic [15:0] i_raw_a [0:15], i_raw_b [0:15];
  src_fmt_t i_src_fmt_a, i_src_fmt_b;
  logic [1:0] i_group_valid, i_acc_first;
  logic [3:0] i_tag [0:1];
  logic i_mx_enable;
  logic [7:0] i_scale_a [0:1], i_scale_b [0:1];
  product12_t o_m_product0 [0:15];
  product9_t o_m_product1 [0:15];
  logic [8:0] o_m_exp_sum0 [0:15];
  logic [5:0] o_m_exp_sum1 [0:15];
  logic [1:0] o_m_product_nz [0:15];
  txn_meta_t o_m_meta, o_align_meta;
  align_meta_t o_align_ameta;
  logic [3:0] o_m_tag [0:1], o_align_tag [0:1];
  align_lane0_t o_align_lane0 [0:15];
  align_lane1_t o_align_lane1 [0:15];

  mac_entry_m_stage #(.TAG_W(4)) u_m (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_in_valid(i_in_valid),
    .o_in_ready(o_in_ready), .i_raw_a(i_raw_a), .i_raw_b(i_raw_b),
    .i_src_fmt_a(i_src_fmt_a), .i_src_fmt_b(i_src_fmt_b),
    .i_group_valid(i_group_valid), .i_acc_first(i_acc_first),
    .i_tag(i_tag), .i_mx_enable(i_mx_enable),
    .i_scale_a(i_scale_a), .i_scale_b(i_scale_b),
    .o_m_valid(o_m_valid), .i_m_ready(o_m_ready),
    .o_m_product0(o_m_product0), .o_m_product1(o_m_product1),
    .o_m_exp_sum0(o_m_exp_sum0), .o_m_exp_sum1(o_m_exp_sum1), .o_m_product_nz(o_m_product_nz),
    .o_m_meta(o_m_meta), .o_m_tag(o_m_tag)
  );

  mac_align_stage #(.TAG_W(4)) u_align (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_m_valid(o_m_valid),
    .o_m_ready(o_m_ready), .i_m_product0(o_m_product0), .i_m_product1(o_m_product1),
    .i_m_exp_sum0(o_m_exp_sum0), .i_m_exp_sum1(o_m_exp_sum1), .i_m_product_nz(o_m_product_nz),
    .i_m_meta(o_m_meta), .i_m_tag(o_m_tag),
    .o_align_valid(o_align_valid), .i_align_ready(i_align_ready),
    .o_align_lane0(o_align_lane0), .o_align_lane1(o_align_lane1),
    .o_align_meta(o_align_meta), .o_align_ameta(o_align_ameta),
    .o_align_tag(o_align_tag)
  );

  task automatic tick;
    @(posedge i_clk);
    #1;
  endtask

  initial begin
    i_rstn = 1'b0;
    i_in_valid = 1'b0;
    i_align_ready = 1'b0;
    i_src_fmt_a = SRC_INT16;
    i_src_fmt_b = SRC_INT16;
    i_group_valid = 2'b01;
    i_acc_first = 2'b01;
    i_mx_enable = 1'b0;
    i_tag[0] = 4'd3; i_tag[1] = '0;
    for (int g = 0; g < 2; g++) begin
      i_scale_a[g] = '0; i_scale_b[g] = '0;
    end
    for (int i = 0; i < DOT_N; i++) begin
      i_raw_a[i] = '0; i_raw_b[i] = '0;
    end
    tick();
    i_rstn = 1'b1;
    i_raw_a[0] = 16'h7fff;
    i_raw_b[0] = 16'hffff;
    i_in_valid = 1'b1;
    tick(); // entry slot
    i_in_valid = 1'b0;
    tick(); // M phase0
    if (!o_m_valid || o_m_meta.int16_phase || o_align_valid)
      $fatal(1, "M/ALIGN phase0 pipeline timing");
    tick(); // ALIGN phase0 and M phase1
    if (!o_align_valid || o_align_meta.mode != MODE_INT16 ||
        o_align_meta.int16_phase ||
        o_align_lane0[0] !== 32'sd65025 || o_align_lane1[0] !== 24'sd32385 ||
        o_align_ameta.emax0 != 0 || o_align_ameta.emax1 != 0 ||
        o_align_ameta.group_nz != 0 || o_align_ameta.inexact != 0 ||
        o_align_meta.group_valid != 2'b01 || o_align_meta.acc_first != 2'b01 ||
        o_align_tag[0] != 4'd3 ||
        !o_m_valid || !o_m_meta.int16_phase)
      $fatal(1, "INT16 phase0 ALIGN or phase1 M mismatch");
    repeat (2) begin
      tick();
      if (!o_align_valid || o_align_meta.int16_phase ||
          o_align_lane0[0] !== 32'sd65025 ||
          o_align_lane1[0] !== 24'sd32385 || !o_m_valid ||
          !o_m_meta.int16_phase || o_m_ready)
        $fatal(1, "M/ALIGN stalled phase mismatch");
    end
    i_align_ready = 1'b1;
    tick(); // phase0 consumed; phase1 replaces it
    if (!o_align_valid || !o_align_meta.int16_phase ||
        o_align_lane0[0] !== -32'sd255 ||
        o_align_lane1[0] !== -24'sd127 || o_align_meta.acc_first != 0 ||
        o_align_tag[0] != 4'd3)
      $fatal(1, "INT16 phase1 ALIGN mismatch");
    tick(); // drain
    if (o_align_valid || o_m_valid) $fatal(1, "INT16 pipeline did not drain");

    i_src_fmt_a = SRC_E4M3;
    i_src_fmt_b = SRC_E5M2;
    i_group_valid = 2'b11;
    i_acc_first = 2'b11;
    i_tag[0] = 4'd4; i_tag[1] = 4'd5;
    i_mx_enable = 1'b1;
    i_scale_a[0] = 8'd0; i_scale_a[1] = 8'd254;
    i_scale_b[0] = 8'd127; i_scale_b[1] = 8'd127;
    i_raw_a[0] = 16'h3c38; i_raw_b[0] = 16'h403c;
    i_raw_a[15] = 16'h7e7e; i_raw_b[15] = 16'h3c3c;
    i_in_valid = 1'b1;
    tick();
    i_in_valid = 1'b0;
    tick();
    tick();
    if (!o_align_valid || o_align_meta.mode != MODE_FP9 ||
        o_align_meta.acc_kind != ACC_TC_FLOAT ||
        o_align_meta.group_valid != 2'b11 || o_align_meta.acc_first != 2'b11 ||
        !o_align_meta.mx_enable || o_align_meta.bias_sum != 8'd22 ||
        o_align_tag[0] != 4'd4 || o_align_tag[1] != 4'd5 ||
        o_align_meta.scale_a[0] != 8'd0 || o_align_meta.scale_a[1] != 8'd254 ||
        o_align_ameta.emax0 != 9'd30 || o_align_ameta.emax1 != 6'd30 ||
        o_align_ameta.group_nz != 2'b11 ||
        o_align_ameta.inexact != 0 || o_align_lane0[0] !== 32'sd8192 ||
        o_align_lane1[0] !== 24'sd24576 ||
        o_align_lane0[15] !== 32'sd3670016 || o_align_lane1[15] !== 24'sd3670016)
      $fatal(1, "FP9 M/ALIGN integration mismatch");
    tick();
    if (o_align_valid) $fatal(1, "FP9 pipeline did not drain");
    $display("mac_m_align PASS");
    $finish;
  end
endmodule
