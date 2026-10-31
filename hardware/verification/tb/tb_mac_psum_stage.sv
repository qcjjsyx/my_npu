module tb_mac_psum_stage;
  import mac_types_pkg::*;

  localparam integer MAX_VECTORS = 512;

  logic i_clk = 1'b0;
  always #5 i_clk = ~i_clk;

  logic i_rstn;
  logic i_align_valid;
  logic o_align_ready;
  align_lane0_t i_align_lane0 [0:15];
  align_lane1_t i_align_lane1 [0:15];
  txn_meta_t i_align_meta;
  align_meta_t i_align_ameta;
  logic [3:0] i_align_tag [0:1];

  logic o_psum_valid;
  logic i_psum_ready;
  psum_packed_t o_psum_data;
  txn_meta_t o_psum_meta;
  align_meta_t o_psum_ameta;
  logic [3:0] o_psum_tag [0:1];

  logic [2:0] mode_mem [0:MAX_VECTORS-1];
  logic kind_mem [0:MAX_VECTORS-1];
  logic [8:0] emax0_mem [0:MAX_VECTORS-1];
  logic [5:0] emax1_mem [0:MAX_VECTORS-1];
  logic [1:0] group_nz_mem [0:MAX_VECTORS-1];
  logic [1:0] inexact_mem [0:MAX_VECTORS-1];
  logic [7:0] bias_mem [0:MAX_VECTORS-1];
  logic [1:0] group_valid_mem [0:MAX_VECTORS-1];
  logic [1:0] acc_first_mem [0:MAX_VECTORS-1];
  logic [3:0] tag0_mem [0:MAX_VECTORS-1];
  logic [3:0] tag1_mem [0:MAX_VECTORS-1];
  logic mx_mem [0:MAX_VECTORS-1];
  logic [7:0] scale_a0_mem [0:MAX_VECTORS-1];
  logic [7:0] scale_a1_mem [0:MAX_VECTORS-1];
  logic [7:0] scale_b0_mem [0:MAX_VECTORS-1];
  logic [7:0] scale_b1_mem [0:MAX_VECTORS-1];
  logic phase_mem [0:MAX_VECTORS-1];
  logic [55:0] expect_data_mem [0:MAX_VECTORS-1];
  logic [31:0] lane0_mem [0:MAX_VECTORS-1][0:15];
  logic [23:0] lane1_mem [0:MAX_VECTORS-1][0:15];

  integer fd;
  integer fields;
  integer vector_count;
  integer source_index;
  integer output_index;
  integer cycle_count;
  integer simultaneous_count;
  integer stalled_count;
  integer scan_mode, scan_kind, scan_group_valid, scan_first, scan_bias;
  integer scan_mx, scan_phase, scan_tag0, scan_tag1;
  integer scan_scale_a0, scan_scale_a1, scan_scale_b0, scan_scale_b1;
  integer scan_emax0, scan_emax1, scan_group_nz, scan_inexact;
  logic [55:0] scan_expected;
  logic [31:0] scan_lane0;
  logic [23:0] scan_lane1;
  integer i;
  string vector_path;
  logic input_fired_prev;
  logic output_stalled_prev;
  psum_packed_t held_data;
  txn_meta_t held_meta;
  align_meta_t held_ameta;
  logic [3:0] held_tag0, held_tag1;

  mac_psum_stage #(.TAG_W(4)) dut (
    .i_clk(i_clk), .i_rstn(i_rstn),
    .i_align_valid(i_align_valid), .o_align_ready(o_align_ready),
    .i_align_lane0(i_align_lane0), .i_align_lane1(i_align_lane1),
    .i_align_meta(i_align_meta), .i_align_ameta(i_align_ameta),
    .i_align_tag(i_align_tag),
    .o_psum_valid(o_psum_valid), .i_psum_ready(i_psum_ready),
    .o_psum_data(o_psum_data),
    .o_psum_meta(o_psum_meta), .o_psum_ameta(o_psum_ameta),
    .o_psum_tag(o_psum_tag)
  );

  task automatic load_input(input integer index);
    begin
      i_align_meta.mode = compute_mode_t'(mode_mem[index]);
      i_align_meta.acc_kind = acc_kind_t'(kind_mem[index]);
      i_align_ameta.emax0 = emax0_mem[index];
      i_align_ameta.emax1 = emax1_mem[index];
      i_align_ameta.group_nz = group_nz_mem[index];
      i_align_ameta.inexact = inexact_mem[index];
      i_align_meta.bias_sum = bias_mem[index];
      i_align_meta.group_valid = group_valid_mem[index];
      i_align_meta.acc_first = acc_first_mem[index];
      i_align_meta.mx_enable = mx_mem[index];
      i_align_meta.int16_phase = phase_mem[index];
      i_align_tag[0] = tag0_mem[index];
      i_align_tag[1] = tag1_mem[index];
      i_align_meta.scale_a[0] = scale_a0_mem[index];
      i_align_meta.scale_a[1] = scale_a1_mem[index];
      i_align_meta.scale_b[0] = scale_b0_mem[index];
      i_align_meta.scale_b[1] = scale_b1_mem[index];
      for (integer lane = 0; lane < 16; lane++) begin
        i_align_lane0[lane] = $signed(lane0_mem[index][lane]);
        i_align_lane1[lane] = $signed(lane1_mem[index][lane]);
      end
    end
  endtask

  function automatic logic meta_matches(input integer index);
    return o_psum_meta.mode === mode_mem[index] &&
           o_psum_meta.acc_kind === kind_mem[index] &&
           o_psum_ameta.emax0 === emax0_mem[index] &&
           o_psum_ameta.emax1 === emax1_mem[index] &&
           o_psum_ameta.group_nz === group_nz_mem[index] &&
           o_psum_ameta.inexact === inexact_mem[index] &&
           o_psum_meta.bias_sum === bias_mem[index] &&
           o_psum_meta.group_valid === group_valid_mem[index] &&
           o_psum_meta.acc_first === acc_first_mem[index] &&
           o_psum_tag[0] === tag0_mem[index] &&
           o_psum_tag[1] === tag1_mem[index] &&
           o_psum_meta.mx_enable === mx_mem[index] &&
           o_psum_meta.scale_a[0] === scale_a0_mem[index] &&
           o_psum_meta.scale_a[1] === scale_a1_mem[index] &&
           o_psum_meta.scale_b[0] === scale_b0_mem[index] &&
           o_psum_meta.scale_b[1] === scale_b1_mem[index] &&
           o_psum_meta.int16_phase === phase_mem[index];
  endfunction

  task automatic check_output(input integer index);
    begin
      if (!o_psum_valid || o_psum_data !== expect_data_mem[index] ||
          !meta_matches(index))
        $fatal(1, "PSUM vector %0d data/metadata mismatch got=%h expected=%h",
               index, o_psum_data, expect_data_mem[index]);
    end
  endtask

  initial begin
    if (!$value$plusargs("VEC=%s", vector_path)) $fatal(1, "missing VEC");
    fd = $fopen(vector_path, "r");
    if (fd == 0) $fatal(1, "cannot open VEC");
    vector_count = 0;
    while (!$feof(fd)) begin
      fields = $fscanf(fd,
        "%h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h\n",
        scan_mode, scan_kind, scan_group_valid, scan_first, scan_bias,
        scan_mx, scan_phase, scan_tag0, scan_tag1,
        scan_scale_a0, scan_scale_a1, scan_scale_b0, scan_scale_b1,
        scan_emax0, scan_emax1, scan_group_nz, scan_inexact, scan_expected);
      if (fields == 18) begin
        if (vector_count >= MAX_VECTORS) $fatal(1, "too many vectors");
        mode_mem[vector_count] = scan_mode[2:0];
        kind_mem[vector_count] = scan_kind[0];
        group_valid_mem[vector_count] = scan_group_valid[1:0];
        acc_first_mem[vector_count] = scan_first[1:0];
        bias_mem[vector_count] = scan_bias[7:0];
        mx_mem[vector_count] = scan_mx[0];
        phase_mem[vector_count] = scan_phase[0];
        tag0_mem[vector_count] = scan_tag0[3:0];
        tag1_mem[vector_count] = scan_tag1[3:0];
        scale_a0_mem[vector_count] = scan_scale_a0[7:0];
        scale_a1_mem[vector_count] = scan_scale_a1[7:0];
        scale_b0_mem[vector_count] = scan_scale_b0[7:0];
        scale_b1_mem[vector_count] = scan_scale_b1[7:0];
        emax0_mem[vector_count] = scan_emax0[8:0];
        emax1_mem[vector_count] = scan_emax1[5:0];
        group_nz_mem[vector_count] = scan_group_nz[1:0];
        inexact_mem[vector_count] = scan_inexact[1:0];
        expect_data_mem[vector_count] = scan_expected;
        for (i = 0; i < 16; i = i + 1) begin
          fields = $fscanf(fd, "%h %h\n", scan_lane0, scan_lane1);
          if (fields != 2) $fatal(1, "bad lane vector %0d/%0d", vector_count, i);
          lane0_mem[vector_count][i] = scan_lane0;
          lane1_mem[vector_count][i] = scan_lane1;
        end
        vector_count = vector_count + 1;
      end else if (fields != -1) begin
        $fatal(1, "bad vector header at index %0d", vector_count);
      end
    end
    $fclose(fd);
    if (vector_count == 0) $fatal(1, "empty vector file");

    i_rstn = 1'b0;
    i_align_valid = 1'b0;
    i_psum_ready = 1'b0;
    i_align_meta = '0;
    i_align_ameta = '0;
    for (i = 0; i < 16; i = i + 1) begin
      i_align_lane0[i] = '0;
      i_align_lane1[i] = '0;
    end
    i_align_tag[0] = '0;
    i_align_tag[1] = '0;
    repeat (2) @(posedge i_clk);
    #1;
    if (o_psum_valid || o_align_ready || o_psum_data !== '0)
      $fatal(1, "PSUM reset state");
    i_rstn = 1'b1;

    source_index = 0;
    output_index = 0;
    cycle_count = 0;
    simultaneous_count = 0;
    stalled_count = 0;
    input_fired_prev = 1'b0;
    output_stalled_prev = 1'b0;

    while (output_index < vector_count && cycle_count < 10000) begin
      @(negedge i_clk);
      // Long enough stalls exercise output stability and shared backpressure.
      i_psum_ready = ((cycle_count % 11) != 4) && ((cycle_count % 11) != 5);
      if (!i_align_valid || input_fired_prev) begin
        if (source_index < vector_count) begin
          load_input(source_index);
          i_align_valid = 1'b1;
        end else begin
          i_align_valid = 1'b0;
        end
      end

      @(posedge i_clk);
      if (output_stalled_prev &&
          (o_psum_data !== held_data || o_psum_meta !== held_meta ||
           o_psum_ameta !== held_ameta ||
           o_psum_tag[0] !== held_tag0 || o_psum_tag[1] !== held_tag1))
        $fatal(1, "PSUM output changed while stalled");
      output_stalled_prev = o_psum_valid && !i_psum_ready;
      if (output_stalled_prev) begin
        held_data = o_psum_data;
        held_meta = o_psum_meta;
        held_ameta = o_psum_ameta;
        held_tag0 = o_psum_tag[0];
        held_tag1 = o_psum_tag[1];
        stalled_count = stalled_count + 1;
      end

      if (o_psum_valid && i_psum_ready) begin
        check_output(output_index);
        output_index = output_index + 1;
      end
      input_fired_prev = i_align_valid && o_align_ready;
      if (input_fired_prev) source_index = source_index + 1;
      if (o_psum_valid && i_psum_ready && i_align_valid && o_align_ready)
        simultaneous_count = simultaneous_count + 1;
      cycle_count = cycle_count + 1;
    end

    if (output_index != vector_count || source_index != vector_count)
      $fatal(1, "PSUM lost transaction: accepted=%0d emitted=%0d expected=%0d",
             source_index, output_index, vector_count);
    if (simultaneous_count == 0 || stalled_count == 0)
      $fatal(1, "PSUM handshake coverage missing simultaneous=%0d stalled=%0d",
             simultaneous_count, stalled_count);

    // Load one final valid result, hold it, then verify synchronous reset
    // cancels it and clears the output slot.
    @(negedge i_clk);
    i_psum_ready = 1'b1;
    load_input(0);
    i_align_valid = 1'b1;
    @(posedge i_clk);
    #1;
    if (!o_psum_valid) $fatal(1, "PSUM reset setup failed");
    @(negedge i_clk);
    i_align_valid = 1'b0;
    i_psum_ready = 1'b0;
    @(posedge i_clk);
    #1;
    if (!o_psum_valid) $fatal(1, "PSUM did not hold stalled result");
    @(negedge i_clk);
    i_rstn = 1'b0;
    @(posedge i_clk);
    #1;
    if (o_psum_valid || o_align_ready || o_psum_data !== '0)
      $fatal(1, "PSUM reset did not cancel held result");

    $display("mac_psum_stage PASS: %0d vectors, %0d cycles, %0d simultaneous, %0d stalled",
             vector_count, cycle_count, simultaneous_count, stalled_count);
    $finish;
  end
endmodule
