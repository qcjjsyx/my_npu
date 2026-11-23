`timescale 1ns/1ps

module tb_new_arch_mac_core;
    reg clk, rst_n, valid, ready;
    reg [7:0] meta;
    reg [1:0] result_ctx;
    reg [351:0] a, b;
    reg [15:0] scale_a, scale_b;
    wire [2:0] mode = meta[7:5];
    wire mx = meta[4];
    wire [1:0] ctx = meta[3:2];
    wire first = meta[1], last = meta[0];
    wire [2:0] free, done, result_mode;
    wire empty, result_valid, result_mx;
    wire [50:0] state0, state1;
    mac_core dut (
        .clk(clk), .rst_n(rst_n), .i_a_fields(a), .i_b_fields(b),
        .i_mode(mode), .i_mx(mx), .i_scale_a(scale_a), .i_scale_b(scale_b),
        .i_ctx(ctx), .i_first(first), .i_last(last), .i_valid(valid),
        .o_ctx_free(free), .o_ctx_done(done), .i_result_ctx(result_ctx),
        .i_result_ready(ready), .o_result_valid(result_valid),
        .o_state0(state0), .o_state1(state1), .o_result_mode(result_mode),
        .o_result_mx(result_mx), .o_pipe_empty(empty)
    );
    new_arch_mac_forward_input_check input_check (
        .mode(mode), .mx(mx), .a(a), .b(b), .scale_a(scale_a), .scale_b(scale_b),
        .h_wide_a(dut.forward.h_wide_a), .h_wide_b(dut.forward.h_wide_b)
    );
    new_arch_mac_acc_float_check acc0_check (
        .active(dut.en_s3 && dut.float_s3), .bypass(dut.acc0.r_bypass_s2),
        .tail(dut.acc0.r_tail_s2), .t(dut.acc0.r_t),
        .sigma(dut.acc0.normalize.sigma), .rounded(dut.acc0.normalize.m_rounded),
        .state(dut.acc0.next_state)
    );
    new_arch_mac_acc_float_check acc1_check (
        .active(dut.en1_s3 && dut.float_s3), .bypass(dut.acc1.r_bypass_s2),
        .tail(dut.acc1.r_tail_s2), .t(dut.acc1.r_t),
        .sigma(dut.acc1.normalize.sigma), .rounded(dut.acc1.normalize.m_rounded),
        .state(dut.acc1.next_state)
    );

    reg [8191:0] vec_path;
    integer fd, columns, cycle, j;
    integer last_input [0:2];
    reg [2:0] closed;
    reg [3:0] batch_before;
    reg [50:0] expected_slot0 [0:2];
    reg [50:0] expected_slot1 [0:2];
    reg [107:0] prev_s2_0, prev_s2_1;
    reg hold_s2_0, hold_s2_1;
    wire [107:0] acc0_s2_data = {dut.acc0.r_t, dut.acc0.r_h_s2, dut.acc0.r_tail_s2,
                                dut.acc0.r_bypass_s2, dut.acc0.r_bypass_state_s2};
    wire [107:0] acc1_s2_data = {dut.acc1.r_t, dut.acc1.r_h_s2, dut.acc1.r_tail_s2,
                                dut.acc1.r_bypass_s2, dut.acc1.r_bypass_state_s2};
    wire [3:0] batch = {dut.contexts.batch_mode, dut.contexts.batch_mx};
    wire integer_s2 = dut.acc_meta_s1[7] && !dut.acc_meta_s1[4];
    wire signed [45:0] add_wide0 = $signed({dut.acc0.r_a[44], dut.acc0.r_a})
        + $signed({dut.acc0.r_b[44], dut.acc0.r_b});
    wire signed [45:0] add_wide1 = $signed({dut.acc1.r_a[44], dut.acc1.r_a})
        + $signed({dut.acc1.r_b[44], dut.acc1.r_b});

    task check_schedule;
        begin
            if (ctx == 2'd3) begin
                $display("REJECT ctx");
                $finish;
            end
            if (first && !free[ctx]) begin
                if (done[ctx] && ready && result_ctx == ctx)
                    $display("REJECT reuse_on_release");
                else if (closed[ctx])
                    $display("REJECT occupied");
                else
                    $display("REJECT repeated_first");
                $finish;
            end
            if (!first && free[ctx]) begin
                $display("REJECT missing_first");
                $finish;
            end
            if (!first && closed[ctx]) begin
                $display("REJECT appended_after_last");
                $finish;
            end
            if (!first && cycle - last_input[ctx] < 3) begin
                $display("REJECT spacing");
                $finish;
            end
            if ((!(&free) || !empty) && {mode, mx} != batch) begin
                $display("REJECT batch_mode");
                $finish;
            end
        end
    endtask

    task check_done;
        input [1:0] id;
        begin
            if (done[id] !== 1'b0) begin
                $display("FAIL DONE has in-flight ctx=%0d at cycle=%0d", id, cycle);
                $finish;
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        cycle = 0;
        if (!$value$plusargs("VEC=%s", vec_path)) $finish;
        fd = $fopen(vec_path, "r");
        if (fd == 0) $finish;
        columns = $fscanf(fd, "%h %h %h %h %h %h %h %b %b",
                          rst_n, valid, meta, result_ctx, ready, scale_a, scale_b, a, b);
        while (columns == 9) begin
            #4;
            if (rst_n && valid) begin
                input_check.check;
                // Observe the batch register across this otherwise illegal edge.
                if ((&free) && empty && !first && ctx != 2'd3) begin
                    batch_before = batch;
                    clk = 1'b1;
                    #1;
                    if (batch !== batch_before) $display("FAIL invalid input changed batch");
                    else $display("REJECT missing_first_batch_held");
                    $finish;
                end
                check_schedule;
            end
            if (cycle != 0)
                $display("B %0d %h %h %h %h %h %h %h %h", cycle, free, done, empty,
                         result_valid, state0, state1, result_mode, result_mx);
            acc0_check.check;
            acc1_check.check;
            if (integer_s2 && ((dut.en_s2 && add_wide0[45] != add_wide0[44]) ||
                               (dut.en1_s2 && add_wide1[45] != add_wide1[44]))) begin
                $display("FAIL signed45 overflow on integer path");
                $finish;
            end
            if (!dut.en1_s1 &&
                {dut.acc1.align.i_old, dut.acc1.align.i_m,
                 dut.acc1.align.i_h, dut.acc1.align.i_int} !== 147'd0) begin
                $display("FAIL unused ACC1 align inputs");
                $finish;
            end
            if (!rst_n) begin
                if ((|{batch, dut.acc_valid, dut.acc_meta_s1, dut.acc_meta_s2,
                       dut.acc0.r_a, dut.acc0.r_b, dut.acc0.r_t,
                       dut.acc0.r_h_s1, dut.acc0.r_h_s2, dut.acc0.r_tail_s1, dut.acc0.r_tail_s2,
                       dut.acc0.r_bypass_s1, dut.acc0.r_bypass_s2,
                       dut.acc0.r_bypass_state_s1, dut.acc0.r_bypass_state_s2,
                       dut.acc1.r_a, dut.acc1.r_b, dut.acc1.r_t,
                       dut.acc1.r_h_s1, dut.acc1.r_h_s2, dut.acc1.r_tail_s1, dut.acc1.r_tail_s2,
                       dut.acc1.r_bypass_s1, dut.acc1.r_bypass_s2,
                       dut.acc1.r_bypass_state_s1, dut.acc1.r_bypass_state_s2}) !== 1'b0 ||
                    free !== 3'b111 || done !== 3'd0 || empty !== 1'b1) begin
                    $display("FAIL core registers not cleared before clock edge");
                    $finish;
                end
                closed = 3'b000;
                for (j = 0; j < 3; j = j+1) begin
                    last_input[j] = -1;
                    if ({dut.acc0.r_state[j], dut.acc1.r_state[j]} !== 102'd0) begin
                        $display("FAIL ACC state not cleared before clock edge, ctx=%0d", j);
                        $finish;
                    end
                end
            end else begin
                if (result_valid && ready) closed[result_ctx] = 1'b0;
                if (valid) begin
                    last_input[ctx] = cycle;
                    closed[ctx] = last;
                end
            end
            for (j = 0; j < 3; j = j+1) begin
                expected_slot0[j] = dut.acc0.r_state[j];
                expected_slot1[j] = dut.acc1.r_state[j];
            end
            if (dut.en_s3) begin
                // The trace represents writes. ACC1 has no candidate to report when disabled.
                $display("W %0d %h %h %h %h", cycle, dut.acc_meta_s2, dut.en1_s3,
                         dut.acc0.next_state, dut.en1_s3 ? dut.acc1.next_state : 51'd0);
                expected_slot0[dut.acc_meta_s2[3:2]] = dut.acc0.next_state;
                if (dut.en1_s3) expected_slot1[dut.acc_meta_s2[3:2]] = dut.acc1.next_state;
            end
            // Use the enables sampled by the registers at this edge.
            hold_s2_0 = rst_n && !dut.en_s2;
            hold_s2_1 = rst_n && !dut.en1_s2;
            prev_s2_0 = acc0_s2_data;
            prev_s2_1 = acc1_s2_data;
            clk = 1'b1;
            #1;
            if ((hold_s2_0 && acc0_s2_data !== prev_s2_0) ||
                (hold_s2_1 && acc1_s2_data !== prev_s2_1)) begin
                $display("FAIL disabled ACC S2 register changed cycle=%0d", cycle);
                $finish;
            end
            for (j = 0; j < 3; j = j+1) begin
                if (dut.acc0.r_state[j] !== expected_slot0[j] ||
                    dut.acc1.r_state[j] !== expected_slot1[j]) begin
                    $display("FAIL state register write cycle=%0d ctx=%0d", cycle, j);
                    $finish;
                end
            end
            for (j = 0; j < 9; j = j+1) begin
                if (dut.forward.r_valid[j]) check_done(dut.forward.r_meta[j][3:2]);
            end
            if (dut.acc_valid[0]) check_done(dut.acc_meta_s1[3:2]);
            if (dut.acc_valid[1]) check_done(dut.acc_meta_s2[3:2]);

            $display("P %0d %h %h %h %h %h", cycle, {dut.acc_valid, dut.forward.r_valid},
                     dut.mx_valid ? dut.mx_meta : 8'd0,
                     dut.acc_valid[0] ? dut.acc_meta_s1 : 8'd0,
                     dut.acc_valid[1] ? dut.acc_meta_s2 : 8'd0, batch);
            $display("A %0d %h %h %h %h %h %h %h %h", cycle, free, done, empty,
                     result_valid, state0, state1, result_mode, result_mx);
            cycle = cycle + 1;
            #5;
            clk = 1'b0;
            columns = $fscanf(fd, "%h %h %h %h %h %h %h %b %b",
                              rst_n, valid, meta, result_ctx, ready, scale_a, scale_b, a, b);
        end
        if (columns > 0 || !$feof(fd)) $finish;
        $fclose(fd);
        $display("PASS: %0d cycles", cycle);
        $finish;
    end
endmodule

module new_arch_mac_acc_float_check #(
    parameter REACHABLE_CORE_RANGE = 1
) (
    input wire active, bypass, tail,
    input wire signed [44:0] t,
    input wire signed [6:0] sigma,
    input wire signed [41:0] rounded,
    input wire [50:0] state
);
    localparam signed [44:0] T_LIMIT = (45'sd1 << 43) - 45'sd8;
    localparam signed [41:0] M_LIMIT = 42'sd1 << 40;
    localparam signed [40:0] MIN_NORMAL = 41'sd1 << 39;
    wire signed [40:0] m = state[40:0];
    wire signed [9:0] h = state[50:41];
    task check;
        begin
            if (active) begin
                if (!bypass) begin
                    if (t == -45'sd1) begin
                        $display("REJECT float_t_minus_one");
                        $finish;
                    end
                    if (t > T_LIMIT || t < -T_LIMIT || (t == 45'sd0 && tail) ||
                        (t != 45'sd0 && (sigma < -7'sd1 || sigma > 7'sd41 ||
                                         rounded < -M_LIMIT || rounded > M_LIMIT))) begin
                        $display("FAIL ACC float work range T=%0d sigma=%0d rounded=%0d",
                                 t, sigma, rounded);
                        $finish;
                    end
                end
                if ((m == 0 && h != 0) || (m != 0 && m > -MIN_NORMAL && m < MIN_NORMAL) ||
                    m == {1'b1, 40'd0} ||
                    (REACHABLE_CORE_RANGE && (h < -10'sd346 || h > 10'sd306))) begin
                    $display("FAIL ACC state format M=%0d H=%0d", m, h);
                    $finish;
                end
            end
        end
    endtask
endmodule

module tb_new_arch_mac_acc_arithmetic;
    reg use_float, inject_minus_one;
    reg [50:0] old_state;
    reg signed [40:0] m;
    reg signed [9:0] h;
    reg signed [44:0] iv;
    wire signed [44:0] a, b;
    wire signed [9:0] h_a;
    wire tail, bypass;
    wire [50:0] bypass_state, state;
    wire signed [44:0] t = inject_minus_one ? -45'sd1 : a + b;
    mac_acc_align align (
        .i_float(use_float), .i_old(old_state), .i_m(m), .i_h(h), .i_int(iv),
        .o_a(a), .o_b(b), .o_h(h_a), .o_tail(tail),
        .o_bypass(bypass), .o_bypass_state(bypass_state)
    );
    mac_acc_round normalize (
        .i_float(use_float), .i_t(t), .i_h(h_a), .i_tail(tail),
        .i_bypass(bypass), .i_bypass_state(bypass_state), .o_state(state)
    );
    new_arch_mac_acc_float_check #(.REACHABLE_CORE_RANGE(0)) floating (
        .active(use_float), .bypass(bypass), .tail(tail), .t(t),
        .sigma(normalize.sigma), .rounded(normalize.m_rounded), .state(state)
    );
    reg [8191:0] vec_path;
    integer fd, columns, row;
    initial begin
        row = 0;
        if (!$value$plusargs("VEC=%s", vec_path)) $finish;
        fd = $fopen(vec_path, "r");
        if (fd == 0) $finish;
        columns = $fscanf(fd, "%h %h %h %h %h %h", use_float, inject_minus_one, old_state, m, h, iv);
        while (columns == 6) begin
            #1;
            floating.check;
            $display("U %0d %h %h %h %h %h %h %h %h", row,
                     a, b, h_a, tail, bypass, bypass_state, t, state);
            row = row+1;
            columns = $fscanf(fd, "%h %h %h %h %h %h", use_float, inject_minus_one, old_state, m, h, iv);
        end
        if (columns > 0 || !$feof(fd)) $finish;
        $fclose(fd);
        $display("PASS: %0d cycles", row);
        $finish;
    end
endmodule
