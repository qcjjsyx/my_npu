module tb_mac_align_psum;
  import mac_types_pkg::*;

  logic i_clk = 1'b0;
  always #5 i_clk = ~i_clk;

  logic i_rstn;
  logic i_m_valid;
  logic o_m_ready;
  product12_t i_m_product0 [0:15];
  product9_t i_m_product1 [0:15];
  logic [8:0] i_m_exp_sum0 [0:15];
  logic [5:0] i_m_exp_sum1 [0:15];
  logic [1:0] i_m_product_nz [0:15];
  txn_meta_t i_m_meta;
  logic [3:0] i_m_tag [0:1];

  logic o_align_valid;
  logic i_align_ready;
  align_lane0_t o_align_lane0 [0:15];
  align_lane1_t o_align_lane1 [0:15];
  txn_meta_t o_align_meta;
  align_meta_t o_align_ameta;
  logic [3:0] o_align_tag [0:1];

  logic o_psum_valid;
  logic i_psum_ready;
  psum_packed_t o_psum_data;
  txn_meta_t o_psum_meta;
  align_meta_t o_psum_ameta;
  logic [3:0] o_psum_tag [0:1];

  integer fd, fields, count, wait_cycles, accepted;
  integer scan_mode, scan_kind, scan_group_valid, scan_first, scan_bias;
  integer scan_mx, scan_phase, scan_tag0, scan_tag1;
  integer scan_scale_a0, scan_scale_a1, scan_scale_b0, scan_scale_b1;
  integer scan_emax0, scan_emax1, scan_group_nz, scan_inexact;
  logic [55:0] scan_expected;
  logic [23:0] scan_product0;
  logic [17:0] scan_product1;
  logic [8:0] scan_exp0;
  logic [5:0] scan_exp1;
  logic [1:0] scan_nz;
  string vector_path;

  mac_align_stage #(.TAG_W(4)) u_align (
    .i_clk(i_clk), .i_rstn(i_rstn),
    .i_m_valid(i_m_valid), .o_m_ready(o_m_ready),
    .i_m_product0(i_m_product0), .i_m_product1(i_m_product1),
    .i_m_exp_sum0(i_m_exp_sum0), .i_m_exp_sum1(i_m_exp_sum1),
    .i_m_product_nz(i_m_product_nz),
    .i_m_meta(i_m_meta), .i_m_tag(i_m_tag),
    .o_align_valid(o_align_valid), .i_align_ready(i_align_ready),
    .o_align_lane0(o_align_lane0), .o_align_lane1(o_align_lane1),
    .o_align_meta(o_align_meta), .o_align_ameta(o_align_ameta),
    .o_align_tag(o_align_tag)
  );

  mac_psum_stage #(.TAG_W(4)) u_psum (
    .i_clk(i_clk), .i_rstn(i_rstn),
    .i_align_valid(o_align_valid), .o_align_ready(i_align_ready),
    .i_align_lane0(o_align_lane0), .i_align_lane1(o_align_lane1),
    .i_align_meta(o_align_meta), .i_align_ameta(o_align_ameta),
    .i_align_tag(o_align_tag),
    .o_psum_valid(o_psum_valid), .i_psum_ready(i_psum_ready),
    .o_psum_data(o_psum_data),
    .o_psum_meta(o_psum_meta), .o_psum_ameta(o_psum_ameta),
    .o_psum_tag(o_psum_tag)
  );

  initial begin
    if (!$value$plusargs("VEC=%s", vector_path)) $fatal(1, "missing VEC");
    fd = $fopen(vector_path, "r");
    if (fd == 0) $fatal(1, "cannot open VEC");
    i_rstn = 1'b0;
    i_m_valid = 1'b0;
    i_psum_ready = 1'b0;
    i_m_meta = '0;
    i_m_meta.mode = MODE_INT8;
    i_m_meta.group_valid = 2'b11;
    for (int g = 0; g < 2; g++) i_m_tag[g] = '0;
    for (int lane = 0; lane < 16; lane++) begin
      i_m_product0[lane] = '0;
      i_m_product1[lane] = '0;
      i_m_exp_sum0[lane] = '0;
      i_m_exp_sum1[lane] = '0;
      i_m_product_nz[lane] = '0;
    end
    repeat (2) @(posedge i_clk);
    #1;
    if (o_m_ready || o_align_valid || o_psum_valid)
      $fatal(1, "ALIGN/PSUM reset handshake");
    i_rstn = 1'b1;
    count = 0;

    while (!$feof(fd)) begin
      fields = $fscanf(fd,
        "%h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h\n",
        scan_mode, scan_kind, scan_group_valid, scan_first, scan_bias,
        scan_mx, scan_phase, scan_tag0, scan_tag1,
        scan_scale_a0, scan_scale_a1, scan_scale_b0, scan_scale_b1,
        scan_emax0, scan_emax1, scan_group_nz, scan_inexact, scan_expected);
      if (fields == 18) begin
        i_m_meta.mode = compute_mode_t'(scan_mode[2:0]);
        i_m_meta.acc_kind = acc_kind_t'(scan_kind[0]);
        i_m_meta.group_valid = scan_group_valid[1:0];
        i_m_meta.acc_first = scan_first[1:0];
        i_m_meta.bias_sum = scan_bias[7:0];
        i_m_meta.mx_enable = scan_mx[0];
        i_m_meta.int16_phase = scan_phase[0];
        i_m_tag[0] = scan_tag0[3:0];
        i_m_tag[1] = scan_tag1[3:0];
        i_m_meta.scale_a[0] = scan_scale_a0[7:0];
        i_m_meta.scale_a[1] = scan_scale_a1[7:0];
        i_m_meta.scale_b[0] = scan_scale_b0[7:0];
        i_m_meta.scale_b[1] = scan_scale_b1[7:0];
        for (int lane = 0; lane < 16; lane++) begin
          fields = $fscanf(fd, "%h %h %h %h %h\n",
                           scan_product0, scan_product1,
                           scan_exp0, scan_exp1, scan_nz);
          if (fields != 5) $fatal(1, "bad M vector %0d/%0d", count, lane);
          i_m_product0[lane] = $signed(scan_product0);
          i_m_product1[lane] = $signed(scan_product1);
          i_m_exp_sum0[lane] = scan_exp0;
          i_m_exp_sum1[lane] = scan_exp1;
          i_m_product_nz[lane] = scan_nz;
        end

        i_psum_ready = 1'b0;
        @(negedge i_clk);
        i_m_valid = 1'b1;
        accepted = 0;
        wait_cycles = 0;
        while (!accepted && wait_cycles < 100) begin
          @(posedge i_clk);
          if (o_m_ready) accepted = 1;
          wait_cycles = wait_cycles + 1;
        end
        if (!accepted) $fatal(1, "ALIGN failed to accept vector %0d", count);
        #1;
        i_m_valid = 1'b0;

        wait_cycles = 0;
        while (!o_psum_valid && wait_cycles < 100) begin
          @(negedge i_clk);
          wait_cycles = wait_cycles + 1;
        end
        if (!o_psum_valid) $fatal(1, "PSUM failed to produce vector %0d", count);
        if (o_psum_data !== scan_expected ||
            o_psum_meta.mode !== scan_mode[2:0] ||
            o_psum_meta.acc_kind !== scan_kind[0] ||
            o_psum_ameta.emax0 !== scan_emax0[8:0] ||
            o_psum_ameta.emax1 !== scan_emax1[5:0] ||
            o_psum_ameta.group_nz !== scan_group_nz[1:0] ||
            o_psum_ameta.inexact !== scan_inexact[1:0] ||
            o_psum_meta.bias_sum !== scan_bias[7:0] ||
            o_psum_meta.group_valid !== scan_group_valid[1:0] ||
            o_psum_meta.acc_first !== scan_first[1:0] ||
            o_psum_tag[0] !== scan_tag0[3:0] ||
            o_psum_tag[1] !== scan_tag1[3:0] ||
            o_psum_meta.mx_enable !== scan_mx[0] ||
            o_psum_meta.scale_a[0] !== scan_scale_a0[7:0] ||
            o_psum_meta.scale_a[1] !== scan_scale_a1[7:0] ||
            o_psum_meta.scale_b[0] !== scan_scale_b0[7:0] ||
            o_psum_meta.scale_b[1] !== scan_scale_b1[7:0] ||
            o_psum_meta.int16_phase !== scan_phase[0])
          $fatal(1, "ALIGN->PSUM vector %0d mismatch data=%h expected=%h",
                 count, o_psum_data, scan_expected);

        repeat (2) begin
          @(posedge i_clk);
          #1;
          if (!o_psum_valid || o_psum_data !== scan_expected ||
              o_psum_tag[0] !== scan_tag0[3:0])
            $fatal(1, "ALIGN->PSUM stalled output changed, vector %0d", count);
        end
        @(negedge i_clk);
        i_psum_ready = 1'b1;
        @(posedge i_clk);
        #1;
        if (o_psum_valid) $fatal(1, "PSUM did not drain vector %0d", count);
        i_psum_ready = 1'b0;
        count = count + 1;
      end else if (fields != -1) begin
        $fatal(1, "bad vector header after %0d", count);
      end
    end
    $fclose(fd);
    if (count == 0) $fatal(1, "empty integration vector file");

    // Drive three copies of the final legal vector through the real ALIGN ->
    // PSUM connection. The third input must wait while both result slots are
    // occupied, then be accepted as the first two transactions advance.
    @(negedge i_clk);
    i_psum_ready = 1'b0;
    i_m_tag[0] = 4'h4;
    i_m_tag[1] = 4'h4;
    i_m_valid = 1'b1;
    accepted = 0;
    wait_cycles = 0;
    while (!accepted && wait_cycles < 100) begin
      @(posedge i_clk);
      if (o_m_ready) accepted = 1;
      wait_cycles = wait_cycles + 1;
    end
    if (!accepted) $fatal(1, "ALIGN failed to accept chain item A");
    #1;
    i_m_valid = 1'b0;

    @(negedge i_clk);
    i_m_tag[0] = 4'h5;
    i_m_tag[1] = 4'h5;
    i_m_valid = 1'b1;
    accepted = 0;
    wait_cycles = 0;
    while (!accepted && wait_cycles < 100) begin
      @(posedge i_clk);
      if (o_m_ready) accepted = 1;
      wait_cycles = wait_cycles + 1;
    end
    if (!accepted) $fatal(1, "ALIGN failed to accept chain item B");
    #1;
    i_m_valid = 1'b0;

    @(negedge i_clk);
    i_m_tag[0] = 4'h6;
    i_m_tag[1] = 4'h6;
    i_m_valid = 1'b1;
    repeat (2) begin
      @(posedge i_clk);
      if (o_m_ready) $fatal(1, "ALIGN ignored PSUM backpressure on item C");
      if (!o_psum_valid || o_psum_data !== scan_expected ||
          o_psum_tag[0] !== 4'h4 || o_psum_tag[1] !== 4'h4)
        $fatal(1, "chain item A changed while both stages were occupied");
    end

    @(negedge i_clk);
    i_psum_ready = 1'b1;
    @(posedge i_clk);
    if (!o_psum_valid || o_psum_data !== scan_expected ||
        o_psum_tag[0] !== 4'h4 || o_psum_tag[1] !== 4'h4 || !o_m_ready)
      $fatal(1, "chain failed simultaneous consume/refill on item A");
    #1;
    i_m_valid = 1'b0;
    i_psum_ready = 1'b0;

    @(posedge i_clk);
    if (!o_psum_valid || o_psum_data !== scan_expected ||
        o_psum_tag[0] !== 4'h5 || o_psum_tag[1] !== 4'h5)
      $fatal(1, "chain item B missing or metadata mismatched");
    @(negedge i_clk);
    i_psum_ready = 1'b1;
    @(posedge i_clk);
    if (!o_psum_valid || o_psum_data !== scan_expected ||
        o_psum_tag[0] !== 4'h5 || o_psum_tag[1] !== 4'h5)
      $fatal(1, "chain failed simultaneous consume/refill on item B");
    #1;
    i_psum_ready = 1'b0;

    @(posedge i_clk);
    if (!o_psum_valid || o_psum_data !== scan_expected ||
        o_psum_tag[0] !== 4'h6 || o_psum_tag[1] !== 4'h6)
      $fatal(1, "chain item C missing or metadata mismatched");
    @(negedge i_clk);
    i_psum_ready = 1'b1;
    @(posedge i_clk);
    #1;
    if (o_psum_valid) $fatal(1, "chain failed to drain final item C");

    $display("mac_align_psum PASS: %0d vectors", count);
    $finish;
  end
endmodule
