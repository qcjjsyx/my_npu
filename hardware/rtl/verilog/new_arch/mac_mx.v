module new_arch_mac_mx (
    input  wire clk,
    input  wire i_en,
    input  wire [2:0] i_mode,
    input  wire i_mx,
    input  wire signed [40:0] i_n_a,
    input  wire signed [40:0] i_n_b,
    input  wire signed [9:0] i_e_a,
    input  wire signed [9:0] i_e_b,
    input  wire signed [8:0] i_h_a,
    input  wire signed [8:0] i_h_b,
    output reg signed [40:0] o_m_a,
    output reg signed [40:0] o_m_b,
    output reg signed [9:0] o_h_a,
    output reg signed [9:0] o_h_b,
    output reg signed [44:0] o_i_a,
    output reg signed [44:0] o_i_b
);
    wire is_float = !i_mode[2] || i_mx;
    wire signed [40:0] m_a, m_b;
    wire signed [9:0] h_a, h_b;
    wire signed [44:0] int_a, int_b;
    new_arch_mac_mx_lane lane_a (
        .i_float(is_float), .i_n(i_n_a), .i_e(i_e_a), .i_h(i_h_a),
        .o_m(m_a), .o_h(h_a), .o_i(int_a)
    );
    new_arch_mac_mx_lane lane_b (
        .i_float(is_float), .i_n(i_n_b), .i_e(i_e_b), .i_h(i_h_b),
        .o_m(m_b), .o_h(h_b), .o_i(int_b)
    );
    always @(posedge clk) begin
        if (i_en) begin
            o_m_a <= m_a;
            o_m_b <= m_b;
            o_h_a <= h_a;
            o_h_b <= h_b;
            o_i_a <= int_a;
            o_i_b <= int_b;
        end
    end
endmodule

module new_arch_mac_mx_lane (
    input  wire i_float,
    input  wire signed [40:0] i_n,
    input  wire signed [9:0] i_e,
    input  wire signed [8:0] i_h,
    output reg signed [40:0] o_m,
    output reg signed [9:0] o_h,
    output reg signed [44:0] o_i
);
    localparam signed [40:0] MIN_COEFF = {1'b1, 40'd0};
    localparam signed [40:0] NEG_HALF = -41'sd549755813888; // -2**39
    wire [39:0] x = i_n[39:0] ^ {40{i_n[40]}};

    reg [2:0] byte_idx;
    reg [7:0] lead_byte;
    always @* begin
        if (|x[39:32]) begin byte_idx = 3'd4; lead_byte = x[39:32]; end
        else if (|x[31:24]) begin byte_idx = 3'd3; lead_byte = x[31:24]; end
        else if (|x[23:16]) begin byte_idx = 3'd2; lead_byte = x[23:16]; end
        else if (|x[15:8]) begin byte_idx = 3'd1; lead_byte = x[15:8]; end
        else begin byte_idx = 3'd0; lead_byte = x[7:0]; end
    end

    function [2:0] highest_one8;
        input [7:0] bits;
        integer j;
        begin
            highest_one8 = 3'd0;
            for (j = 0; j < 8; j = j+1)
                if (bits[j]) highest_one8 = j[2:0];
        end
    endfunction

    wire [5:0] t = {byte_idx, highest_one8(lead_byte)};
    wire signed [40:0] shifted = i_n <<< (6'd39 - t);
    wire negative_power = (shifted == MIN_COEFF);
    wire [5:0] k = i_n == -41'sd1 ? 6'd0 : t + {5'd0, negative_power};
    wire signed [10:0] e_wide =
        $signed({i_e[9], i_e}) + $signed({{2{i_h[8]}}, i_h});
    wire signed [10:0] h_wide = e_wide + $signed({5'd0, k});

    always @* begin
        o_m = 41'sd0;
        o_h = 10'sd0;
        o_i = 45'sd0;
        if (!i_float) begin
            o_i = i_n;
        end else if (i_n != 41'sd0) begin
            if (i_n == -41'sd1)
                o_m = NEG_HALF;
            else
                o_m = negative_power ? (shifted >>> 1) : shifted;
            o_h = h_wide[9:0];
        end
    end
endmodule
