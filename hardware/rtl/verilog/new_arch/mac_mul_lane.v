// The mode muxes precede the four shared multiplier blocks and exponent adder.
module new_arch_mac_mul_lane #(
    parameter ODD = 0
) (
    input  wire [2:0] i_mode,
    input  wire [21:0] i_a,
    input  wire [21:0] i_b,
    output wire signed [17:0] o_w0,
    output wire signed [17:0] o_w1,
    output wire signed [9:0] o_s0,
    output wire signed [9:0] o_s1,
    output wire [8:0] o_t
);
    localparam [2:0] FP4 = 3'd0, FP8 = 3'd1, BF16 = 3'd2, FP16 = 3'd3,
                     INT4 = 3'd4, INT8 = 3'd5, INT16 = 3'd6;
    reg signed [8:0] a_w0, a_w1, b_w0, b_w1;
    reg signed [4:0] a_s0, a_s1, b_s0, b_s1;
    reg [7:0] ea, eb;

    always @* begin
        a_w0 = 9'sd0;
        a_w1 = 9'sd0;
        b_w0 = 9'sd0;
        b_w1 = 9'sd0;
        a_s0 = 5'sd0;
        a_s1 = 5'sd0;
        b_s0 = 5'sd0;
        b_s1 = 5'sd0;
        ea = 8'd0;
        eb = 8'd0;
        case (i_mode)
            FP4, INT4: begin
                a_w0 = $signed(i_a[4:0]);
                b_w0 = $signed(i_b[4:0]);
                a_w1 = $signed(i_a[14:10]);
                b_w1 = $signed(i_b[14:10]);
                a_s0 = i_a[9:5];
                b_s0 = i_b[9:5];
                a_s1 = i_a[19:15];
                b_s1 = i_b[19:15];
            end
            FP8: begin
                a_w0 = $signed(i_a[7:0]);
                b_w0 = $signed(i_b[7:0]);
                a_w1 = $signed(i_a[18:11]);
                b_w1 = $signed(i_b[18:11]);
                ea = {1'b0, i_a[21:19], 1'b0, i_a[10:8]};
                eb = {1'b0, i_b[21:19], 1'b0, i_b[10:8]};
            end
            INT8: begin
                a_w0 = $signed(i_a[7:0]);
                b_w0 = $signed(i_b[7:0]);
                a_w1 = $signed(i_a[15:8]);
                b_w1 = $signed(i_b[15:8]);
            end
            BF16: begin
                a_w0 = i_a[8:0];
                b_w0 = i_b[8:0];
                ea = i_a[16:9];
                eb = i_b[16:9];
            end
            FP16, INT16: begin
                b_w0 = {1'b0, i_b[7:0]};
                b_w1 = i_mode == FP16 ? $signed(i_b[14:8]) : $signed(i_b[15:8]);
                if (ODD) begin
                    a_w0 = i_mode == FP16 ? $signed(i_a[14:8]) : $signed(i_a[15:8]);
                end else begin
                    a_w0 = {1'b0, i_a[7:0]};
                    if (i_mode == FP16) begin
                        ea = {5'b0, i_a[17:15]};
                        eb = {5'b0, i_b[17:15]};
                    end
                end
                a_w1 = a_w0;
            end
            default: begin end
        endcase
    end

    assign o_w0 = a_w0 * b_w0;
    assign o_w1 = a_w1 * b_w1;
    assign o_s0 = a_s0 * b_s0;
    assign o_s1 = a_s1 * b_s1;
    assign o_t = {1'b0, ea} + {1'b0, eb};
endmodule
