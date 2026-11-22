module mac_mx (
    input  wire clk,
    input  wire rst_n,
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
    mac_mx_lane lane_a (
        .i_float(is_float), .i_n(i_n_a), .i_e(i_e_a), .i_h(i_h_a),
        .o_m(m_a), .o_h(h_a), .o_i(int_a)
    );
    mac_mx_lane lane_b (
        .i_float(is_float), .i_n(i_n_b), .i_e(i_e_b), .i_h(i_h_b),
        .o_m(m_b), .o_h(h_b), .o_i(int_b)
    );
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_m_a <= 41'd0;
            o_m_b <= 41'd0;
            o_h_a <= 10'd0;
            o_h_b <= 10'd0;
            o_i_a <= 45'd0;
            o_i_b <= 45'd0;
        end else if (i_en) begin
            o_m_a <= m_a;
            o_m_b <= m_b;
            o_h_a <= h_a;
            o_h_b <= h_b;
            o_i_a <= int_a;
            o_i_b <= int_b;
        end
    end
endmodule
