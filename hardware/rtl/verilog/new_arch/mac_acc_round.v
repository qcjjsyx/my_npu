module mac_acc_round (
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
    // Compute both exponent candidates while the mantissa is being rounded.
    wire signed [10:0] h_sub = $signed({i_h[9], i_h}) - $signed({{4{sigma[6]}}, sigma});
    wire signed [10:0] h_inc = h_sub + 11'sd1;
    wire signed [10:0] h_wide = renorm ? h_inc : h_sub;

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
