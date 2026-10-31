module tb_mac_align_stage;
  import mac_types_pkg::*;
  logic i_clk = 1'b0;
  always #5 i_clk = ~i_clk;
  logic i_rstn, i_m_valid, o_m_ready, o_align_valid, i_align_ready;
  product12_t i_m_product0 [0:15];
  product9_t i_m_product1 [0:15];
  logic [8:0] i_m_exp_sum0 [0:15];
  logic [5:0] i_m_exp_sum1 [0:15];
  logic [1:0] i_m_product_nz [0:15];
  txn_meta_t i_m_meta;
  logic [3:0] i_m_tag [0:1];

  align_lane0_t o_align_lane0 [0:15], expect_x0 [0:15];
  align_lane1_t o_align_lane1 [0:15], expect_x1 [0:15];
  txn_meta_t o_align_meta;
  align_meta_t o_align_ameta;
  logic [3:0] o_align_tag [0:1];
  logic [8:0] expect_emax0;
  logic [5:0] expect_emax1;
  logic [1:0] expect_group_nz, expect_inexact;
  integer fd, fields, count, bad_case;
  integer scan_mode, scan_kind, scan_gv, scan_first, scan_bias, scan_mx, scan_phase;
  logic [7:0] scan_scale_a0, scan_scale_a1, scan_scale_b0, scan_scale_b1;
  string vector_path;

  mac_align_stage #(.TAG_W(4)) dut (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_m_valid(i_m_valid), .o_m_ready(o_m_ready),
    .i_m_product0(i_m_product0), .i_m_product1(i_m_product1),
    .i_m_exp_sum0(i_m_exp_sum0), .i_m_exp_sum1(i_m_exp_sum1), .i_m_product_nz(i_m_product_nz),
    .i_m_meta(i_m_meta), .i_m_tag(i_m_tag),
    .o_align_valid(o_align_valid), .i_align_ready(i_align_ready),
    .o_align_lane0(o_align_lane0), .o_align_lane1(o_align_lane1),
    .o_align_meta(o_align_meta), .o_align_ameta(o_align_ameta),
    .o_align_tag(o_align_tag)
  );

  task automatic tick;
    @(posedge i_clk);
    #1;
  endtask

  task automatic check_output;
    if (!o_align_valid || o_align_ameta.emax0 !== expect_emax0 ||
        o_align_ameta.emax1 !== expect_emax1 ||
        o_align_ameta.group_nz !== expect_group_nz ||
        o_align_ameta.inexact !== expect_inexact ||
        o_align_meta.mode !== i_m_meta.mode ||
        o_align_meta.acc_kind !== i_m_meta.acc_kind ||
        o_align_meta.bias_sum !== i_m_meta.bias_sum ||
        o_align_meta.group_valid !== i_m_meta.group_valid ||
        o_align_meta.acc_first !== i_m_meta.acc_first ||
        o_align_meta.mx_enable !== i_m_meta.mx_enable ||
        o_align_meta.int16_phase !== i_m_meta.int16_phase ||
        o_align_tag[0] !== i_m_tag[0] || o_align_tag[1] !== i_m_tag[1] ||
        o_align_meta.scale_a[0] !== i_m_meta.scale_a[0] ||
        o_align_meta.scale_a[1] !== i_m_meta.scale_a[1] ||
        o_align_meta.scale_b[0] !== i_m_meta.scale_b[0] ||
        o_align_meta.scale_b[1] !== i_m_meta.scale_b[1])
      $fatal(1, "ALIGN metadata/max vector %0d", count);
    for (int i = 0; i < 16; i++) begin
      if (o_align_lane0[i] !== expect_x0[i] || o_align_lane1[i] !== expect_x1[i])
        $fatal(1, "ALIGN vector %0d lane %0d got %h/%h expected %h/%h",
               count, i, o_align_lane0[i], o_align_lane1[i], expect_x0[i], expect_x1[i]);
    end
  endtask

  initial begin
    if (!$value$plusargs("VEC=%s", vector_path)) $fatal(1, "missing VEC");
    bad_case = 0;
    fields = $value$plusargs("BAD=%d", bad_case);
    fd = $fopen(vector_path, "r");
    if (fd == 0) $fatal(1, "cannot open VEC");
    i_rstn = 1'b0;
    i_m_valid = 1'b0;
    i_align_ready = 1'b0;
    i_m_meta = '0;
    i_m_meta.mode = MODE_FP19;
    i_m_meta.group_valid = 2'b01;
    for (int g = 0; g < 2; g++) i_m_tag[g] = '0;
    for (int i = 0; i < 16; i++) begin
      i_m_product0[i] = '0; i_m_product1[i] = '0;
      i_m_exp_sum0[i] = '0; i_m_exp_sum1[i] = '0; i_m_product_nz[i] = '0;
    end
    tick();
    if (o_m_ready || o_align_valid) $fatal(1, "ALIGN reset handshake");
    i_rstn = 1'b1;
    count = 0;
    while (!$feof(fd)) begin
      fields = $fscanf(fd,
        "%h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h\n",
        scan_mode, scan_kind, scan_gv, scan_first, scan_bias, scan_mx, scan_phase,
        i_m_tag[0], i_m_tag[1],
        scan_scale_a0, scan_scale_a1, scan_scale_b0, scan_scale_b1,
        expect_emax0, expect_emax1, expect_group_nz, expect_inexact);
      if (fields == 17) begin
        i_m_meta.mode = compute_mode_t'(scan_mode[2:0]);
        i_m_meta.acc_kind = acc_kind_t'(scan_kind[0]);
        i_m_meta.group_valid = scan_gv[1:0];
        i_m_meta.acc_first = scan_first[1:0];
        i_m_meta.bias_sum = scan_bias[7:0];
        i_m_meta.mx_enable = scan_mx[0];
        i_m_meta.int16_phase = scan_phase[0];
        i_m_meta.scale_a[0] = scan_scale_a0;
        i_m_meta.scale_a[1] = scan_scale_a1;
        i_m_meta.scale_b[0] = scan_scale_b0;
        i_m_meta.scale_b[1] = scan_scale_b1;
        for (int i = 0; i < 16; i++) begin
          fields = $fscanf(fd, "%h %h %h %h %h %h %h\n",
                           i_m_product0[i], i_m_product1[i], i_m_exp_sum0[i], i_m_exp_sum1[i], i_m_product_nz[i],
                           expect_x0[i], expect_x1[i]);
          if (fields != 7) $fatal(1, "bad lane vector %0d/%0d", count, i);
        end
        if (count == 0) begin
          if (bad_case == 1) i_m_product0[0] = 24'h800000;
          if (bad_case == 2) i_m_product_nz[0][0] = 1'b1;
          if (bad_case == 3) i_m_meta.group_valid = 2'b00;
        end
        i_m_valid = 1'b1;
        i_align_ready = 1'b1; // Includes full-slot replacement after a stall.
        #1;
        if (!o_m_ready) $fatal(1, "ALIGN did not accept/refill input");
        tick();
        check_output();
        i_m_valid = 1'b0;
        i_align_ready = 1'b0;
        for (int i = 0; i < 16; i++) begin
          i_m_product0[i] = '0; i_m_product1[i] = '0;
          i_m_exp_sum0[i] = '0; i_m_exp_sum1[i] = '0; i_m_product_nz[i] = '0;
        end
        repeat (2) begin
          #1;
          if (o_m_ready) $fatal(1, "ALIGN accepted while blocked");
          tick();
          check_output();
        end
        if ((count & 1) == 0) begin
          i_align_ready = 1'b1;
          tick();
          if (o_align_valid) $fatal(1, "ALIGN failed to drain");
        end
        count = count + 1;
      end else if (fields != -1) begin
        $fatal(1, "bad header after %0d", count);
      end
    end
    $fclose(fd);
    // Synchronous reset cancels a result held under backpressure.
    i_m_valid = 1'b1;
    i_align_ready = 1'b1;
    tick();
    if (!o_align_valid) $fatal(1, "reset setup failed");
    i_m_valid = 1'b0;
    i_align_ready = 1'b0;
    i_rstn = 1'b0;
    tick();
    if (o_align_valid || o_m_ready) $fatal(1, "reset left pending ALIGN result");
    $display("mac_align_stage PASS: %0d vectors", count);
    $finish;
  end
endmodule
