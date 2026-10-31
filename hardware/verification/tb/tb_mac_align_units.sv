module tb_mac_align_units;
  logic signed [31:0] i_y32;
  logic signed [23:0] i_y24;
  logic [8:0] i_distance;
  logic i_window24;
  logic signed [31:0] o_x32;
  logic signed [23:0] o_x24;
  logic o_lost32, o_lost24;
  logic [8:0] i_exps9 [0:15];
  logic [15:0] i_nz;
  logic [8:0] o_max_exp;
  logic o_has_nonzero;
  integer fd, count, fields, kind, expect_lost;
  logic [31:0] input_y, expect_x;
  logic [8:0] expect_max;
  logic expect_nz;
  string vector_path;

  mac_align_rne #(.W(32)) u_32 (
    .i_y(i_y32), .i_distance(i_distance), .i_window24(i_window24),
    .o_x(o_x32), .o_inexact(o_lost32)
  );
  mac_align_rne #(.W(24)) u_24 (
    .i_y(i_y24), .i_distance(i_distance), .i_window24(1'b1),
    .o_x(o_x24), .o_inexact(o_lost24)
  );
  mac_align_max16 #(.EXP_W(9)) u_max (
    .i_exp(i_exps9), .i_nz(i_nz), .o_max_exp(o_max_exp),
    .o_has_nonzero(o_has_nonzero)
  );

  initial begin
    if (!$value$plusargs("VEC=%s", vector_path)) $fatal(1, "missing VEC");
    fd = $fopen(vector_path, "r");
    if (fd == 0) $fatal(1, "cannot open VEC");
    count = 0;
    while (!$feof(fd)) begin
      fields = $fscanf(fd, "%h %h %h %h %h %h %h\n",
                       kind, input_y, i_distance, i_window24,
                       expect_x, expect_lost, i_nz);
      if (fields == 7) begin
        for (int i = 0; i < 16; i++) begin
          fields = $fscanf(fd, "%h\n", i_exps9[i]);
          if (fields != 1) $fatal(1, "bad exponent vector");
        end
        fields = $fscanf(fd, "%h %h\n", expect_max, expect_nz);
        if (fields != 2) $fatal(1, "bad expected max");
        i_y32 = input_y;
        i_y24 = input_y[23:0];
        #1;
        if (o_max_exp !== expect_max || o_has_nonzero !== expect_nz)
          $fatal(1, "max vector %0d got %h/%b expected %h/%b",
                 count, o_max_exp, o_has_nonzero, expect_max, expect_nz);
        if (kind == 2) begin
          if (o_x24 !== expect_x[23:0] || o_lost24 !== expect_lost[0])
            $fatal(1, "RNE24 vector %0d y=%h d=%0d got %h/%b expected %h/%b",
                   count, input_y, i_distance, o_x24, o_lost24,
                   expect_x[23:0], expect_lost[0]);
        end else if (o_x32 !== expect_x || o_lost32 !== expect_lost[0]) begin
          $fatal(1, "RNE32 vector %0d kind=%0d y=%h d=%0d got %h/%b expected %h/%b",
                 count, kind, input_y, i_distance, o_x32, o_lost32,
                 expect_x, expect_lost[0]);
        end
        count = count + 1;
      end else if (fields != -1) begin
        $fatal(1, "bad vector after %0d", count);
      end
    end
    $fclose(fd);
    $display("mac_align_units PASS: %0d vectors", count);
  end
endmodule
