module tb_mac_mul_frontend;
  mac_types_pkg::operand_t i_operand_a, i_operand_b;
  logic [2:0] i_mode;
  logic [1:0] i_group_valid;
  logic i_int16_phase;
  mac_types_pkg::product12_t o_product0;
  mac_types_pkg::product9_t o_product1;
  logic [8:0] o_exp_sum0;
  logic [5:0] o_exp_sum1;
  logic [1:0] o_product_nz;
  logic o_product0_active, o_product1_active;
  logic [23:0] expected_product0;
  logic [17:0] expected_product1;
  logic [8:0] expected_exp0;
  logic [5:0] expected_exp1;
  logic [1:0] expected_nz, expected_active;
  integer fd, read_count, vector_count;
  string vector_path;

  mac_mul_frontend dut (
    .i_operand_a(i_operand_a), .i_operand_b(i_operand_b), .i_mode(i_mode),
    .i_group_valid(i_group_valid), .i_int16_phase(i_int16_phase),
    .o_product0(o_product0), .o_product1(o_product1),
    .o_exp_sum0(o_exp_sum0), .o_exp_sum1(o_exp_sum1),
    .o_product_nz(o_product_nz),
    .o_product0_active(o_product0_active), .o_product1_active(o_product1_active)
  );

  initial begin
    if (!$value$plusargs("VEC=%s", vector_path)) $fatal(1, "missing +VEC path");
    fd = $fopen(vector_path, "r");
    if (fd == 0) $fatal(1, "cannot open vector file");
    vector_count = 0;
    while (!$feof(fd)) begin
      read_count = $fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h\n",
                           i_mode, i_group_valid, i_int16_phase, i_operand_a, i_operand_b,
                           expected_product0, expected_product1,
                           expected_exp0, expected_exp1, expected_nz,
                           expected_active);
      if (read_count == 11) begin
        #1;
        if (o_product0 !== expected_product0 ||
            o_product1 !== expected_product1 ||
            o_exp_sum0 !== expected_exp0 || o_exp_sum1 !== expected_exp1 ||
            o_product_nz !== expected_nz ||
            {o_product1_active, o_product0_active} !== expected_active)
          $fatal(1, "vector %0d mode=%h gv=%h phase=%h a=%h b=%h got p=%h/%h e=%h/%h nz=%h act=%b expected p=%h/%h e=%h/%h nz=%h act=%b",
                 vector_count, i_mode, i_group_valid, i_int16_phase, i_operand_a, i_operand_b,
                 o_product0, o_product1, o_exp_sum0, o_exp_sum1, o_product_nz,
                 {o_product1_active, o_product0_active}, expected_product0, expected_product1,
                 expected_exp0, expected_exp1, expected_nz, expected_active);
        vector_count = vector_count + 1;
      end else if (read_count != -1) begin
        $fatal(1, "malformed vector after %0d", vector_count);
      end
    end
    $fclose(fd);
    $display("mac_mul_frontend PASS: %0d vectors", vector_count);
  end
endmodule
