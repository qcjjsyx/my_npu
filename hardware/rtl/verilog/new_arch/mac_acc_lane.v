// One feedback lane. Shared transaction metadata and valid are held by the core.
module new_arch_mac_acc_lane (
    input  wire clk,
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
    new_arch_mac_acc_align align (
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
    always @(posedge clk) begin
        if (i_en_s1) begin
            r_a <= align_a;
            r_b <= align_b;
            r_h_s1 <= align_h;
            r_tail_s1 <= align_tail;
            r_bypass_s1 <= align_bypass;
            r_bypass_state_s1 <= align_state;
        end
        if (i_en_s2) begin
            r_t <= add_a + add_b;
            r_h_s2 <= r_h_s1;
            r_tail_s2 <= r_tail_s1;
            r_bypass_s2 <= r_bypass_s1;
            r_bypass_state_s2 <= r_bypass_state_s1;
        end
    end

    wire [50:0] next_state;
    new_arch_mac_acc_round normalize (
        .i_float(i_en_s3 && i_float_s3),
        .i_t(i_en_s3 ? r_t : 45'sd0), .i_h(i_en_s3 ? r_h_s2 : 10'sd0),
        .i_tail(i_en_s3 && r_tail_s2), .i_bypass(i_en_s3 && r_bypass_s2),
        .i_bypass_state(i_en_s3 ? r_bypass_state_s2 : 51'd0), .o_state(next_state)
    );

    // S3's combinational result is captured directly in the selected context.
    always @(posedge clk) begin
        if (i_en_s3) r_state[i_ctx_s3] <= next_state;
    end

    assign o_result_state = select_state(i_result_ctx, r_state[0], r_state[1], r_state[2]);
endmodule

module new_arch_mac_acc_align (
    input  wire i_float,
    input  wire [50:0] i_old,
    input  wire signed [40:0] i_m,
    input  wire signed [9:0] i_h,
    input  wire signed [44:0] i_int,
    output reg signed [44:0] o_a,
    output reg signed [44:0] o_b,
    output reg signed [9:0] o_h,
    output reg o_tail,
    output reg o_bypass,
    output reg [50:0] o_bypass_state
);
    wire signed [40:0] old_m = i_old[40:0];
    wire signed [9:0] old_h = i_old[50:41];
    wire use_align = i_float && old_m != 41'sd0 && i_m != 41'sd0;
    wire old_is_high = old_h >= i_h;
    wire signed [9:0] high_h = use_align ? (old_is_high ? old_h : i_h) : 10'sd0;
    wire signed [9:0] low_h = use_align ? (old_is_high ? i_h : old_h) : 10'sd0;
    wire signed [10:0] d = $signed({high_h[9], high_h}) - $signed({low_h[9], low_h});
    wire [40:0] high_m = use_align ? (old_is_high ? old_m : i_m) : 41'd0;
    wire [40:0] low_m = use_align ? (old_is_high ? i_m : old_m) : 41'd0;

    wire signed [42:0] shifted [0:6];
    wire [6:0] sticky;
    assign shifted[0] = {low_m, 2'b00};
    assign sticky[0] = 1'b0;
    genvar j;
    generate for (j = 0; j < 6; j = j+1) begin : shift
        localparam DISTANCE = 1 << j;
        assign shifted[j+1] = d[j] ? ($signed(shifted[j]) >>> DISTANCE) : shifted[j];
        assign sticky[j+1] = sticky[j] || (d[j] && (|shifted[j][DISTANCE-1:0]));
    end endgenerate
    wire far = d >= 11'sd43;
    wire signed [42:0] low_aligned = far ? {43{low_m[40]}} : shifted[6];

    always @* begin
        o_a = 45'sd0;
        o_b = 45'sd0;
        o_h = 10'sd0;
        o_tail = 1'b0;
        o_bypass = 1'b0;
        o_bypass_state = 51'd0;
        if (!i_float) begin
            o_a = i_old[44:0];
            o_b = i_int;
        end else if (!use_align) begin
            o_bypass = 1'b1;
            if (old_m != 41'sd0)
                o_bypass_state = i_old;
            else if (i_m != 41'sd0)
                o_bypass_state = {i_h, i_m};
        end else begin
            o_a = $signed({high_m, 2'b00});
            o_b = low_aligned;
            o_h = high_h;
            o_tail = far ? (|low_m) : sticky[6];
        end
    end
endmodule

module new_arch_mac_acc_round (
    input  wire i_float,
    input  wire signed [44:0] i_t,
    input  wire signed [9:0] i_h,
    input  wire i_tail,
    input  wire i_bypass,
    input  wire [50:0] i_bypass_state,
    output reg [50:0] o_state
);
    localparam signed [41:0] MAX_MAG = 42'sd1099511627776; // 2**40
    wire [43:0] x = i_t[43:0] ^ {44{i_t[44]}};
    wire [47:0] padded = {4'd0, x};
    reg [2:0] byte_index, bit_index;
    reg [7:0] lead_byte;
    integer j;
    always @* begin
        byte_index = 3'd0;
        for (j = 0; j < 6; j = j+1)
            if (|padded[j*8 +: 8]) byte_index = j[2:0];
        lead_byte = padded[{byte_index, 3'b000} +: 8];
        bit_index = 3'd0;
        for (j = 0; j < 8; j = j+1)
            if (lead_byte[j]) bit_index = j[2:0];
    end
    wire signed [6:0] sigma = 7'sd41 - $signed({1'b0, byte_index, bit_index});
    wire signed [44:0] shifted = sigma[6] ? (i_t >>> 1) : (i_t <<< sigma[5:0]);
    wire signed [40:0] m_pre = shifted[42:2];
    wire sticky = i_tail || (sigma[6] && i_t[0]);
    wire increment = shifted[1] && (shifted[0] || sticky || m_pre[0]);
    wire signed [41:0] m_rounded =
        $signed({m_pre[40], m_pre}) + $signed({41'd0, increment});
    wire renorm = m_rounded == MAX_MAG || m_rounded == -MAX_MAG;
    wire signed [41:0] mantissa = renorm ? (m_rounded >>> 1) : m_rounded;
    wire signed [10:0] h_wide = $signed({i_h[9], i_h})
        - $signed({{4{sigma[6]}}, sigma}) + $signed({10'd0, renorm});

    always @* begin
        if (!i_float)
            o_state = {{6{i_t[44]}}, i_t};
        else if (i_bypass)
            o_state = i_bypass_state;
        else if (i_t == 45'sd0)
            o_state = 51'd0;
        else
            o_state = {h_wide[9:0], mantissa[40:0]};
    end
endmodule
