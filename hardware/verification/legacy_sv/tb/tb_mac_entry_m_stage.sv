module tb_mac_entry_m_stage;
  import mac_types_pkg::*;

  logic i_clk = 1'b0;
  always #5 i_clk = ~i_clk;
  logic i_rstn, i_in_valid, o_in_ready, o_m_valid, i_m_ready;
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
  txn_meta_t o_m_meta;
  logic [3:0] o_m_tag [0:1];

  mac_entry_m_stage #(.TAG_W(4)) dut (
    .i_clk(i_clk), .i_rstn(i_rstn),
    .i_in_valid(i_in_valid), .o_in_ready(o_in_ready),
    .i_raw_a(i_raw_a), .i_raw_b(i_raw_b),
    .i_src_fmt_a(i_src_fmt_a), .i_src_fmt_b(i_src_fmt_b),
    .i_group_valid(i_group_valid), .i_acc_first(i_acc_first),
    .i_tag(i_tag), .i_mx_enable(i_mx_enable),
    .i_scale_a(i_scale_a), .i_scale_b(i_scale_b),
    .o_m_valid(o_m_valid), .i_m_ready(i_m_ready),
    .o_m_product0(o_m_product0), .o_m_product1(o_m_product1),
    .o_m_exp_sum0(o_m_exp_sum0), .o_m_exp_sum1(o_m_exp_sum1),
    .o_m_product_nz(o_m_product_nz),
    .o_m_meta(o_m_meta), .o_m_tag(o_m_tag)
  );

  task automatic tick;
    @(posedge i_clk);
    #1;
  endtask

  task automatic expect_int16_phase0;
    if (!o_m_valid || o_m_meta.mode != MODE_INT16 ||
        o_m_meta.int16_phase != 1'b0 ||
        o_m_product0[0] !== 24'sd65025 || o_m_product1[0] !== 18'sd32385 ||
        o_m_exp_sum0[0] !== 9'd0 || o_m_exp_sum1[0] !== 6'd0 ||
        o_m_product_nz[0] !== 2'b00 || o_m_meta.group_valid != 2'b01 ||
        o_m_meta.acc_first != 2'b01 || o_m_tag[0] !== 4'd3 ||
        o_m_meta.acc_kind != ACC_INT32 || o_m_meta.bias_sum != 8'd0)
      $fatal(1, "INT16 phase0 or metadata mismatch");
  endtask

  task automatic expect_int16_phase1;
    if (!o_m_valid || o_m_meta.mode != MODE_INT16 ||
        o_m_meta.int16_phase != 1'b1 ||
        o_m_product0[0] !== -24'sd255 || o_m_product1[0] !== -18'sd127 ||
        o_m_meta.acc_first != 2'b00 || o_m_tag[0] !== 4'd3)
      $fatal(1, "INT16 phase1 or first mismatch");
  endtask

  task automatic expect_fp9;
    if (!o_m_valid || o_m_meta.mode != MODE_FP9 ||
        o_m_meta.int16_phase != 1'b0 ||
        o_m_product0[0] !== 24'sd64 || o_m_product1[0] !== 18'sd96 ||
        o_m_exp_sum0[0] !== 9'd22 || o_m_exp_sum1[0] !== 6'd23 ||
        o_m_product_nz[0] !== 2'b11 || o_m_meta.group_valid != 2'b11 ||
        o_m_meta.acc_first != 2'b11 || o_m_tag[0] !== 4'd4 ||
        o_m_tag[1] !== 4'd5 ||
        o_m_meta.bias_sum != 8'd22 || o_m_meta.acc_kind != ACC_TC_FLOAT ||
        !o_m_meta.mx_enable || o_m_meta.scale_a[0] != 8'd0 ||
        o_m_meta.scale_a[1] != 8'd254 ||
        o_m_product0[15] !== 24'sd112 || o_m_product1[15] !== 18'sd112 ||
        o_m_exp_sum0[15] !== 9'd30 || o_m_exp_sum1[15] !== 6'd30)
      $fatal(1, "FP9 mixed-source output or metadata mismatch");
  endtask

  initial begin
    i_rstn = 1'b0;
    i_in_valid = 1'b0;
    i_m_ready = 1'b0;
    i_src_fmt_a = SRC_INT16;
    i_src_fmt_b = SRC_INT16;
    i_group_valid = 2'b01;
    i_acc_first = 2'b01;
    i_mx_enable = 1'b0;
    i_tag[0] = 4'd3;
    i_tag[1] = 4'd0;
    i_scale_a[0] = 8'd0;
    i_scale_a[1] = 8'd0;
    i_scale_b[0] = 8'd0;
    i_scale_b[1] = 8'd0;
    for (int i = 0; i < DOT_N; i++) begin
      i_raw_a[i] = '0;
      i_raw_b[i] = '0;
    end
    tick();
    if (o_in_ready || o_m_valid) $fatal(1, "reset exposed an input/output handshake");
    tick();
    i_rstn = 1'b1;
    i_raw_a[0] = 16'h7fff;
    i_raw_b[0] = 16'hffff;
    i_in_valid = 1'b1;
    #1;
    if (!o_in_ready) $fatal(1, "empty entry slot did not accept input");
    tick(); // one full INT16 input is captured
    if (o_m_valid || o_in_ready) $fatal(1, "INT16 slot/phase0 control mismatch");

    i_in_valid = 1'b0;
    i_raw_a[0] = 16'h0001; // External input may change after acceptance.
    i_raw_b[0] = 16'h0001;
    i_tag[0] = 4'd9;
    tick(); // phase0 issues into M; output is blocked
    expect_int16_phase0();
    if (o_in_ready) $fatal(1, "phase1 not protected under backpressure");
    repeat (3) begin
      tick();
      expect_int16_phase0();
      if (o_in_ready) $fatal(1, "slot released during M stall");
    end

    // Refill with the next dot16 of the same INT16 long dot product.
    i_m_ready = 1'b1;
    i_in_valid = 1'b1;
    i_src_fmt_a = SRC_INT16;
    i_src_fmt_b = SRC_INT16;
    i_group_valid = 2'b01;
    i_acc_first = 2'b00;
    i_raw_a[0] = 16'h0002;
    i_raw_b[0] = 16'h0003;
    i_tag[0] = 4'd3;
    i_mx_enable = 1'b0;
    #1;
    if (!o_in_ready) $fatal(1, "INT16 phase1 did not permit refill");
    tick();
    expect_int16_phase1();
    i_in_valid = 1'b0;
    tick();
    if (!o_m_valid || o_m_meta.mode != MODE_INT16 ||
        o_m_meta.int16_phase != 1'b0 ||
        o_m_product0[0] !== 24'sd6 || o_m_product1[0] !== 18'sd0 ||
        o_m_meta.acc_first != 2'b00 || o_m_tag[0] !== 4'd3)
      $fatal(1, "second INT16 phase0 mismatch");
    tick();
    if (!o_m_valid || o_m_meta.mode != MODE_INT16 ||
        o_m_meta.int16_phase != 1'b1 ||
        o_m_product0[0] !== 24'sd0 || o_m_product1[0] !== 18'sd0 ||
        o_m_meta.acc_first != 2'b00)
      $fatal(1, "second INT16 phase1 mismatch");
    tick();
    if (o_m_valid) $fatal(1, "second INT16 transaction did not drain");

    // New long dot product: mixed FP8 uses independent A/B source decoders.
    i_src_fmt_a = SRC_E4M3;
    i_src_fmt_b = SRC_E5M2;
    i_group_valid = 2'b11;
    i_acc_first = 2'b11;
    i_raw_a[0] = 16'h3c38;
    i_raw_b[0] = 16'h403c;
    i_raw_a[15] = 16'h7e7e;
    i_raw_b[15] = 16'h3c3c;
    i_tag[0] = 4'd4;
    i_tag[1] = 4'd5;
    i_mx_enable = 1'b1;
    i_scale_a[0] = 8'd0;
    i_scale_a[1] = 8'd254;
    i_scale_b[0] = 8'd127;
    i_scale_b[1] = 8'd127;
    i_in_valid = 1'b1;
    tick();
    i_in_valid = 1'b0;
    tick();
    expect_fp9();

    // The same FP9 long dot product can continue while its prior M phase stalls.
    i_m_ready = 1'b0;
    i_in_valid = 1'b1;
    i_acc_first = 2'b00;
    #1;
    if (!o_in_ready) $fatal(1, "empty slot failed under M backpressure");
    tick();
    expect_fp9();
    i_in_valid = 1'b0;
    repeat (2) begin
      tick();
      expect_fp9();
      if (o_in_ready) $fatal(1, "full slot accepted while M blocked");
    end

    // Release the stalled FP9 phase, then drain before changing modes.
    i_m_ready = 1'b1;
    tick();
    if (!o_m_valid || o_m_meta.mode != MODE_FP9 || o_m_meta.acc_first != 2'b00 ||
        o_m_product0[0] !== 24'sd64 || o_m_product1[0] !== 18'sd96)
      $fatal(1, "stalled FP9 continuation mismatch");
    tick();
    if (o_m_valid) $fatal(1, "FP9 did not drain before mode change");

    // DUAL single-valid group masks the stale group0 physical product.
    i_in_valid = 1'b1;
    i_src_fmt_a = SRC_INT8;
    i_src_fmt_b = SRC_INT8;
    i_group_valid = 2'b10;
    i_acc_first = 2'b10;
    i_raw_a[0] = 16'h8100;
    i_raw_b[0] = 16'h7f00;
    i_raw_a[15] = 16'h0000;
    i_raw_b[15] = 16'h0000;
    i_tag[0] = 4'd0;
    i_tag[1] = 4'd6;
    i_mx_enable = 1'b0;
    tick();
    i_in_valid = 1'b0;
    tick();
    if (!o_m_valid || o_m_meta.mode != MODE_INT8 ||
        o_m_meta.acc_kind != ACC_INT32 ||
        o_m_meta.group_valid != 2'b10 || o_m_product0[0] !== 24'sd0 ||
        o_m_product1[0] !== -18'sd16129 || o_m_tag[1] !== 4'd6)
      $fatal(1, "INT8 group1 result or isolation mismatch");
    tick();
    if (o_m_valid) $fatal(1, "INT8 did not drain before mode change");

    // FP19 uses MUL12 and masks the previous MUL9 result without updating it.
    i_in_valid = 1'b1;
    i_src_fmt_a = SRC_FP16;
    i_src_fmt_b = SRC_FP16;
    i_group_valid = 2'b01;
    i_acc_first = 2'b01;
    i_raw_a[0] = 16'h3c00;
    i_raw_b[0] = 16'h3c00;
    i_tag[0] = 4'd7;
    i_tag[1] = 4'd0;
    tick();
    i_in_valid = 1'b0;
    tick();
    if (!o_m_valid || o_m_meta.mode != MODE_FP19 ||
        o_m_meta.acc_kind != ACC_TC_FLOAT ||
        o_m_product0[0] !== 24'sd1048576 || o_m_product1[0] !== 18'sd0 ||
        o_m_exp_sum0[0] !== 9'd30 || o_m_product_nz[0] !== 2'b01 ||
        o_m_tag[0] !== 4'd7 || o_m_meta.group_valid != 2'b01)
      $fatal(1, "FP19 result or MUL9 isolation mismatch");
    if (dut.product1_q[0] !== -18'sd16129)
      $fatal(1, "inactive MUL9 result register unexpectedly updated");
    tick(); // drain
    if (o_m_valid) $fatal(1, "unexpected duplicate M issue");

    // Synchronous reset cancels an accepted INT16 transaction before phase1.
    i_src_fmt_a = SRC_INT16;
    i_src_fmt_b = SRC_INT16;
    i_raw_a[0] = 16'h0001;
    i_raw_b[0] = 16'h0001;
    i_in_valid = 1'b1;
    i_m_ready = 1'b0;
    tick();
    i_in_valid = 1'b0;
    tick();
    if (!o_m_valid || o_m_meta.int16_phase != 1'b0) $fatal(1, "reset setup failed");
    i_rstn = 1'b0;
    tick();
    if (o_m_valid || o_in_ready) $fatal(1, "reset left a pending handshake");
    i_rstn = 1'b1;
    i_m_ready = 1'b1;
    #1;
    if (!o_in_ready) $fatal(1, "empty slot did not reopen after reset");
    tick();
    if (o_m_valid) $fatal(1, "cancelled INT16 phase1 reappeared");
    $display("mac_entry_m_stage PASS");
    $finish;
  end
endmodule
