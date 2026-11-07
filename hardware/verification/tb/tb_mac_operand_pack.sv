module tb_mac_operand_pack;
  logic [15:0] i_raw_a, i_raw_b;
  logic [2:0] i_src_fmt_a, i_src_fmt_b;
  logic [1:0] i_group_valid;
  mac_types_pkg::operand_t o_payload_a, o_payload_b;
  logic [25:0] expected_a, expected_b;
  integer fd, read_count, vector_count;
  string vector_path;

  mac_operand_pack dut_a (
    .i_raw(i_raw_a), .i_src_fmt(i_src_fmt_a),
    .i_group_valid(i_group_valid), .o_payload(o_payload_a)
  );
  mac_operand_pack dut_b (
    .i_raw(i_raw_b), .i_src_fmt(i_src_fmt_b),
    .i_group_valid(i_group_valid), .o_payload(o_payload_b)
  );

  initial begin
    if (!$value$plusargs("VEC=%s", vector_path)) $fatal(1, "missing +VEC path");
    fd = $fopen(vector_path, "r");
    if (fd == 0) $fatal(1, "cannot open vector file");
    vector_count = 0;
    while (!$feof(fd)) begin
      read_count = $fscanf(fd, "%h %h %h %h %h %h %h\n",
                           i_src_fmt_a, i_src_fmt_b, i_group_valid,
                           i_raw_a, i_raw_b, expected_a, expected_b);
      if (read_count == 7) begin
        #1;
        if (o_payload_a !== expected_a || o_payload_b !== expected_b)
          $fatal(1, "vector %0d fmt=%h/%h valid=%h raw=%h/%h got=%h/%h expected=%h/%h",
                 vector_count, i_src_fmt_a, i_src_fmt_b, i_group_valid, i_raw_a, i_raw_b,
                 o_payload_a, o_payload_b, expected_a, expected_b);
        vector_count = vector_count + 1;
      end else if (read_count != -1) begin
        $fatal(1, "malformed vector after %0d", vector_count);
      end
    end
    $fclose(fd);
    $display("mac_operand_pack PASS: %0d vectors", vector_count);
  end
endmodule
