`timescale 1ns/1ps

module tb_new_arch_mac_forward;
    reg clk, rst_n, valid;
    reg [7:0] meta;
    reg [351:0] a, b;
    reg [15:0] scale_a, scale_b;
    wire [2:0] mode = meta[7:5];
    wire mx = meta[4];
    wire out_valid, empty;
    wire [7:0] out_meta;
    wire signed [40:0] m0, m1;
    wire signed [9:0] h0, h1;
    wire signed [44:0] int0, int1;
    mac_forward dut (
        .clk(clk), .rst_n(rst_n), .i_a_fields(a), .i_b_fields(b),
        .i_mode(mode), .i_mx(mx), .i_scale_a(scale_a), .i_scale_b(scale_b),
        .i_ctx(meta[3:2]), .i_first(meta[1]), .i_last(meta[0]), .i_valid(valid),
        .o_meta(out_meta), .o_valid(out_valid), .o_pipe_empty(empty),
        .o_m_v0(m0), .o_m_v1(m1), .o_h_v0(h0), .o_h_v1(h1),
        .o_i_v0(int0), .o_i_v1(int1)
    );

    wire [287:0] w0, w1;
    wire [159:0] s0, s1;
    wire [143:0] t;
    genvar j;
    generate for (j = 0; j < 16; j = j+1) begin : observe_mul
        localparam CLUSTER = j / 2;
        localparam LANE = j % 2;
        assign w0[j*18 +: 18] = dut.mul.cluster[CLUSTER].u_cluster.r_w0[LANE];
        assign w1[j*18 +: 18] = dut.mul.cluster[CLUSTER].u_cluster.r_w1[LANE];
        assign s0[j*10 +: 10] = dut.mul.cluster[CLUSTER].u_cluster.r_s0[LANE];
        assign s1[j*10 +: 10] = dut.mul.cluster[CLUSTER].u_cluster.r_s1[LANE];
        assign t[j*9 +: 9] = dut.mul.cluster[CLUSTER].u_cluster.r_t[LANE];
    end endgenerate

    new_arch_mac_forward_input_check input_check (
        .mode(mode), .mx(mx), .a(a), .b(b), .scale_a(scale_a), .scale_b(scale_b),
        .h_wide_a(dut.h_wide_a), .h_wide_b(dut.h_wide_b)
    );

    task check_mx_range;
        input signed [40:0] n;
        input signed [10:0] e;
        input signed [10:0] h;
        begin
            if (n == 41'sh10000000000 || e < -11'sd512 || e > 11'sd511 ||
                h < -11'sd512 || h > 11'sd511) begin
                $display("FAIL MX range: N=%0d e=%0d H=%0d", n, e, h);
                $finish;
            end
        end
    endtask

    task check_async_reset;
        integer stage;
        begin
            if ((|{dut.r_valid, dut.r_a, dut.r_b, w0, w1, s0, s1, t,
                   dut.p0, dut.p1, dut.p2, dut.p3, dut.q0, dut.q1,
                   dut.align.r_p0, dut.align.r_p1, dut.align.r_p2, dut.align.r_p3,
                   dut.align.r_q0, dut.align.r_q1,
                   dut.align.r_max_a, dut.align.r_max_b, dut.align.r_any_a, dut.align.r_any_b,
                   dut.tree_a, dut.tree_b, dut.align_e_a, dut.align_e_b,
                   dut.reduce.tree_a.r_lo, dut.reduce.tree_b.r_lo,
                   dut.reduce.sum_a, dut.reduce.sum_b, dut.reduce.r_e_lo_a, dut.reduce.r_e_lo_b,
                   dut.reduce.r_e_hi_a, dut.reduce.r_e_hi_b,
                   dut.n_a, dut.n_b, dut.sum_e_a, dut.sum_e_b, dut.sum_h_a, dut.sum_h_b,
                   m0, m1, h0, h1, int0, int1}) !== 1'b0 || empty !== 1'b1) begin
                $display("FAIL forward registers not cleared before clock edge");
                $finish;
            end
            for (stage = 0; stage < 9; stage = stage+1) begin
                if (dut.r_meta[stage] !== 8'd0) begin
                    $display("FAIL metadata not cleared at stage %0d", stage);
                    $finish;
                end
            end
            for (stage = 0; stage < 7; stage = stage+1) begin
                if ({dut.r_h_a[stage], dut.r_h_b[stage]} !== 18'd0) begin
                    $display("FAIL scale not cleared at stage %0d", stage);
                    $finish;
                end
            end
        end
    endtask

    reg [8191:0] vec_path;
    integer fd, columns, cycle;
    reg [8:0] previous_valid;
    initial begin
        clk = 1'b0;
        cycle = 0;
        if (!$value$plusargs("VEC=%s", vec_path)) begin
            $display("Missing +VEC=<path>");
            $finish;
        end
        fd = $fopen(vec_path, "r");
        if (fd == 0) begin
            $display("Cannot open vector file");
            $finish;
        end
        columns = $fscanf(fd, "%h %h %h %h %h %b %b", rst_n, valid, meta, scale_a, scale_b, a, b);
        while (columns == 7) begin
            #4;
            if (!rst_n)
                check_async_reset;
            else if (cycle != 0 && previous_valid !== dut.r_valid) begin
                $display("FAIL valid changed before edge at cycle %0d", cycle);
                $finish;
            end
            if (rst_n && valid) input_check.check;
            if (rst_n && dut.r_valid[7] && (!dut.r_meta[7][7] || dut.r_meta[7][4])) begin
                check_mx_range(dut.n_a, dut.mx.lane_a.e_wide, dut.mx.lane_a.h_wide);
                check_mx_range(dut.n_b, dut.mx.lane_b.e_wide, dut.mx.lane_b.h_wide);
            end
            clk = 1'b1;
            #1;
            previous_valid = dut.r_valid;
            $display("V %0d %h %h %h", cycle, dut.r_valid, empty, out_valid);
            if (dut.r_valid[0])
                $display("E0 %0d %h %h %h", cycle, dut.r_meta[0], dut.r_h_a[0], dut.r_h_b[0]);
            if (dut.r_valid[1])
                $display("E1 %0d %h %h %h %h %h %h %h %h", cycle, dut.r_meta[1],
                         dut.r_h_a[1], dut.r_h_b[1], w0, w1, s0, s1, t);
            if (dut.r_valid[2])
                $display("E2 %0d %h %h %h %h %h %h %h %h %h", cycle, dut.r_meta[2],
                         dut.r_h_a[2], dut.r_h_b[2], dut.p0, dut.p1, dut.p2, dut.p3, dut.q0, dut.q1);
            if (dut.r_valid[3])
                $display("E3 %0d %h %h %h %h %h %h %h", cycle, dut.r_meta[3],
                         dut.r_h_a[3], dut.r_h_b[3], dut.align.r_max_a, dut.align.r_max_b,
                         dut.align.r_any_a, dut.align.r_any_b);
            if (dut.r_valid[4])
                $display("E4 %0d %h %h %h %h %h %h %h", cycle, dut.r_meta[4],
                         dut.r_h_a[4], dut.r_h_b[4], dut.tree_a, dut.tree_b,
                         dut.align_e_a, dut.align_e_b);
            if (dut.r_valid[5])
                $display("E5 %0d %h %h %h %h %h %h %h", cycle, dut.r_meta[5],
                         dut.r_h_a[5], dut.r_h_b[5], dut.reduce.tree_a.r_lo, dut.reduce.tree_b.r_lo,
                         dut.reduce.r_e_lo_a, dut.reduce.r_e_lo_b);
            if (dut.r_valid[6])
                $display("E6 %0d %h %h %h %h %h %h %h", cycle, dut.r_meta[6],
                         dut.r_h_a[6], dut.r_h_b[6], dut.reduce.sum_a, dut.reduce.sum_b,
                         dut.reduce.r_e_hi_a, dut.reduce.r_e_hi_b);
            if (dut.r_valid[7])
                $display("E7 %0d %h %h %h %h %h %h %h", cycle, dut.r_meta[7],
                         dut.n_a, dut.sum_e_a, dut.sum_h_a, dut.n_b, dut.sum_e_b, dut.sum_h_b);
            if (out_valid)
                $display("E8 %0d %h %h %h %h %h %h %h", cycle, out_meta, m0, h0, int0, m1, h1, int1);
            cycle = cycle + 1;
            #5;
            clk = 1'b0;
            columns = $fscanf(fd, "%h %h %h %h %h %b %b", rst_n, valid, meta, scale_a, scale_b, a, b);
        end
        if (columns > 0 || !$feof(fd)) begin
            $display("FAIL malformed stimulus at cycle %0d (%0d columns)", cycle, columns);
            $finish;
        end
        $fclose(fd);
        $display("PASS: %0d cycles", cycle);
        $finish;
    end
endmodule

// Exercise the MX arithmetic directly, including signed41 coefficients
// outside the tighter bound enforced by the legal forward input modes.
module tb_new_arch_mac_mx_lane;
    reg use_float;
    reg signed [40:0] n, expected_m;
    reg signed [9:0] e, expected_h;
    reg signed [8:0] h;
    reg signed [44:0] expected_i;
    wire signed [40:0] got_m;
    wire signed [9:0] got_h;
    wire signed [44:0] got_i;
    mac_mx_lane dut (
        .i_float(use_float), .i_n(n), .i_e(e), .i_h(h),
        .o_m(got_m), .o_h(got_h), .o_i(got_i)
    );
    reg [8191:0] vec_path;
    integer fd, columns, cycle;
    initial begin
        cycle = 0;
        if (!$value$plusargs("VEC=%s", vec_path)) $finish;
        fd = $fopen(vec_path, "r");
        if (fd == 0) $finish;
        columns = $fscanf(fd, "%h %h %h %h %h %h %h", use_float, n, e, h,
                          expected_m, expected_h, expected_i);
        while (columns == 7) begin
            #1;
            if ({got_m, got_h, got_i} !== {expected_m, expected_h, expected_i}) begin
                $display("FAIL MX row %0d N=%0d e=%0d h=%0d got %h/%h/%h expected %h/%h/%h",
                         cycle, n, e, h, got_m, got_h, got_i, expected_m, expected_h, expected_i);
                $finish;
            end
            cycle = cycle + 1;
            columns = $fscanf(fd, "%h %h %h %h %h %h %h", use_float, n, e, h,
                              expected_m, expected_h, expected_i);
        end
        if (columns > 0 || !$feof(fd)) $finish;
        $fclose(fd);
        $display("PASS: %0d cycles", cycle);
        $finish;
    end
endmodule

// Verification-only input contract shared with the full-core testbench.
module new_arch_mac_forward_input_check (
    input wire [2:0] mode,
    input wire mx,
    input wire [351:0] a, b,
    input wire [15:0] scale_a, scale_b,
    input wire signed [9:0] h_wide_a, h_wide_b
);
    task check;
        integer lane, q;
        begin
            if (mode == 3'd7) begin
                $display("REJECT mode");
                $finish;
            end
            if (mx && mode != 3'd0 && mode != 3'd1 && mode != 3'd5) begin
                $display("REJECT mx_mode");
                $finish;
            end
            if (mx) begin
                if (scale_a[7:0] == 8'hff || scale_b[7:0] == 8'hff ||
                    scale_a[15:8] == 8'hff || scale_b[15:8] == 8'hff) begin
                    $display("REJECT scale");
                    $finish;
                end
                if (h_wide_a < -10'sd254 || h_wide_a > 10'sd254 ||
                    h_wide_b < -10'sd254 || h_wide_b > 10'sd254) begin
                    $display("REJECT scale_sum");
                    $finish;
                end
            end
            for (lane = 0; lane < 16; lane = lane+1) begin
                if (mode == 3'd4) begin
                    for (q = 0; q < 4; q = q+1) begin
                        if (a[lane*22+q*5 +: 5] == 5'b11000 ||
                            b[lane*22+q*5 +: 5] == 5'b11000) begin
                            $display("REJECT int4");
                            $finish;
                        end
                    end
                end
                if (mode == 3'd5) begin
                    for (q = 0; q < 2; q = q+1) begin
                        if (a[lane*22+q*8 +: 8] == 8'h80 ||
                            b[lane*22+q*8 +: 8] == 8'h80) begin
                            $display("REJECT int8");
                            $finish;
                        end
                    end
                end
                if (mode == 3'd6 && lane % 2 == 0 &&
                    (a[lane*22 +: 16] == 16'h8000 || b[lane*22 +: 16] == 16'h8000)) begin
                    $display("REJECT int16");
                    $finish;
                end
            end
        end
    endtask
endmodule
