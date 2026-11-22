// One feedback lane. Shared transaction metadata and valid are held by the core.
module mac_acc_lane (
    input  wire clk,
    input  wire rst_n,
    input  wire i_en_s1,
    input  wire i_en_s2,
    input  wire i_en_s3,
    input  wire [1:0] i_ctx_s1,
    input  wire i_first_s1,
    input  wire i_float_s1,
    input  wire signed [40:0] i_m,
    input  wire signed [9:0] i_h,
    input  wire signed [44:0] i_int,
    input  wire [1:0] i_ctx_s3,
    input  wire i_float_s3,
    input  wire [1:0] i_result_ctx,
    output wire [50:0] o_result_state
);
    reg [50:0] r_state [0:2];
    function [50:0] select_state;
        input [1:0] ctx;
        input [50:0] state0, state1, state2;
        begin
            case (ctx)
                2'd0: select_state = state0;
                2'd1: select_state = state1;
                2'd2: select_state = state2;
                default: select_state = 51'd0;
            endcase
        end
    endfunction
    wire [50:0] old_state = i_en_s1 && !i_first_s1
        ? select_state(i_ctx_s1, r_state[0], r_state[1], r_state[2]) : 51'd0;

    wire signed [44:0] align_a, align_b;
    wire signed [9:0] align_h;
    wire align_tail, align_bypass;
    wire [50:0] align_state;
    mac_acc_align align (
        .i_float(i_en_s1 && i_float_s1), .i_old(old_state),
        .i_m(i_en_s1 ? i_m : 41'sd0),
        .i_h(i_en_s1 ? i_h : 10'sd0),
        .i_int(i_en_s1 ? i_int : 45'sd0),
        .o_a(align_a), .o_b(align_b), .o_h(align_h),
        .o_tail(align_tail), .o_bypass(align_bypass), .o_bypass_state(align_state)
    );

    reg signed [44:0] r_a, r_b, r_t;
    reg signed [9:0] r_h_s1, r_h_s2;
    reg r_tail_s1, r_tail_s2;
    reg r_bypass_s1, r_bypass_s2;
    reg [50:0] r_bypass_state_s1, r_bypass_state_s2;
    wire signed [44:0] add_a = i_en_s2 ? r_a : 45'sd0;
    wire signed [44:0] add_b = i_en_s2 ? r_b : 45'sd0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_a <= 45'd0;
            r_b <= 45'd0;
            r_h_s1 <= 10'd0;
            r_tail_s1 <= 1'b0;
            r_bypass_s1 <= 1'b0;
            r_bypass_state_s1 <= 51'd0;
        end else if (i_en_s1) begin
            r_a <= align_a;
            r_b <= align_b;
            r_h_s1 <= align_h;
            r_tail_s1 <= align_tail;
            r_bypass_s1 <= align_bypass;
            r_bypass_state_s1 <= align_state;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_t <= 45'd0;
            r_h_s2 <= 10'd0;
            r_tail_s2 <= 1'b0;
            r_bypass_s2 <= 1'b0;
            r_bypass_state_s2 <= 51'd0;
        end else if (i_en_s2) begin
            r_t <= add_a + add_b;
            r_h_s2 <= r_h_s1;
            r_tail_s2 <= r_tail_s1;
            r_bypass_s2 <= r_bypass_s1;
            r_bypass_state_s2 <= r_bypass_state_s1;
        end
    end

    wire [50:0] next_state;
    mac_acc_round normalize (
        .i_float(i_en_s3 && i_float_s3),
        .i_t(i_en_s3 ? r_t : 45'sd0), .i_h(i_en_s3 ? r_h_s2 : 10'sd0),
        .i_tail(i_en_s3 && r_tail_s2), .i_bypass(i_en_s3 && r_bypass_s2),
        .i_bypass_state(i_en_s3 ? r_bypass_state_s2 : 51'd0), .o_state(next_state)
    );

    // S3's combinational result is captured directly in the selected context.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_state[0] <= 51'd0;
            r_state[1] <= 51'd0;
            r_state[2] <= 51'd0;
        end else if (i_en_s3) r_state[i_ctx_s3] <= next_state;
    end

    assign o_result_state = select_state(i_result_ctx, r_state[0], r_state[1], r_state[2]);
endmodule
