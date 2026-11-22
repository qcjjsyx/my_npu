`timescale 1ns/1ps

module tb_new_arch_mac_input_convert;
    reg clk;
    reg rst_n;
    reg [15:0] raw;
    reg [2:0] fmt;
    reg valid;
    wire [21:0] fields;
    wire out_valid;

    mac_input_convert dut (
        .clk(clk), .rst_n(rst_n), .i_raw(raw), .i_fmt(fmt), .i_valid(valid),
        .o_fields(fields), .o_valid(out_valid)
    );

    reg [8191:0] vector_path;
    integer vector_file;
    integer columns;
    integer cycle;
    reg [21:0] expected_fields;
    reg expected_valid;
    reg [21:0] previous_fields;
    reg previous_valid;

    initial begin
        clk = 1'b0;
        cycle = 0;
        if (!$value$plusargs("VEC=%s", vector_path)) begin
            $display("Missing +VEC=<path>");
            $finish;
        end
        vector_file = $fopen(vector_path, "r");
        if (vector_file == 0) begin
            $display("Cannot open vector file %0s", vector_path);
            $finish;
        end

        columns = $fscanf(vector_file, "%h %h %h %h %h %h",
                          rst_n, valid, fmt, raw, expected_valid, expected_fields);
        while (columns == 6) begin
            #4;
            if (!rst_n)
                {previous_valid, previous_fields} = 23'd0;
            if (cycle != 0 && {out_valid, fields} !== {previous_valid, previous_fields}) begin
                $display("cycle %0d before edge: got %b/%06h, expected %b/%06h (rst_n=%b)",
                         cycle, out_valid, fields, previous_valid, previous_fields, rst_n);
                $finish;
            end

            clk = 1'b1;
            #1;
            if ({out_valid, fields} !== {expected_valid, expected_fields}) begin
                $display("cycle %0d after edge: fmt=%0d raw=%04h rst_n=%b valid=%b got %b/%06h, expected %b/%06h",
                         cycle, fmt, raw, rst_n, valid, out_valid, fields,
                         expected_valid, expected_fields);
                $finish;
            end
            previous_fields = expected_fields;
            previous_valid = expected_valid;
            cycle = cycle + 1;

            #5;
            clk = 1'b0;
            columns = $fscanf(vector_file, "%h %h %h %h %h %h",
                              rst_n, valid, fmt, raw, expected_valid, expected_fields);
        end
        if (columns > 0 || !$feof(vector_file)) begin
            $display("Malformed input at cycle %0d (%0d columns)", cycle, columns);
            $finish;
        end
        $fclose(vector_file);
        $display("PASS: %0d cycles", cycle);
        $finish;
    end
endmodule
