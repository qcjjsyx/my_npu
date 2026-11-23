// Two physical lanes share the wide-element broadcast and product reconstruction.
// The cluster registers R_MUL and presents products to the frontend's R_PROD.
module mac_mul_cluster (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        i_en_mul,
    input  wire [2:0]  i_mode_mul,
    input  wire [2:0]  i_mode_prod,
    input  wire [43:0] i_a,
    input  wire [43:0] i_b,
    output wire [71:0] o_prod,
    output wire [13:0] o_q0,
    output wire [7:0]  o_q1,
    output wire [35:0] o_pair_prod,
    output wire [6:0]  o_pair_q0
);
    localparam [2:0] FP4 = 3'd0, FP8 = 3'd1, BF16 = 3'd2, FP16 = 3'd3,
                     INT4 = 3'd4, INT8 = 3'd5, INT16 = 3'd6;

    wire paired = (i_mode_mul == FP16 || i_mode_mul == INT16);
    reg signed [17:0] r_w0 [0:1];
    reg signed [17:0] r_w1 [0:1];
    reg signed [9:0]  r_s0 [0:1];
    reg signed [9:0]  r_s1 [0:1];
    reg [8:0]        r_t  [0:1];

    wire [17:0] p00 = r_w0[0];
    wire signed [17:0] p01 = r_w1[0];
    wire signed [17:0] p10 = r_w0[1];
    wire [17:0] p11 = r_w1[1];
    // FP16 uses the signed22 low portion of the shared signed24 high path.
    wire [15:0] p11_field = i_mode_prod == FP16
        ? {{2{p11[13]}}, p11[13:0]} : p11[15:0];
    wire signed [23:0] h_base = {p11_field, p00[15:8]};
    wire signed [23:0] high = h_base
        + $signed({{6{p01[17]}}, p01}) + $signed({{6{p10[17]}}, p10});
    wire signed [31:0] pair_product = {high, p00[7:0]};
    assign o_pair_prod = i_mode_prod == FP16
        ? {6'd0, pair_product[29:0]} : {4'd0, pair_product};
    assign o_pair_q0 = i_mode_prod == FP16 ? {3'b0, r_t[0][3:0]} : 7'd0;

    genvar j;
    generate for (j = 0; j < 2; j = j + 1) begin : lane
        wire [21:0] a = (j == 0 || paired) ? i_a[21:0] : i_a[43:22];
        wire [21:0] b = (j == 0 || paired) ? i_b[21:0] : i_b[43:22];
        wire signed [17:0] w0, w1;
        wire signed [9:0] s0, s1;
        wire [8:0] t;
        mac_mul_lane #(.ODD(j)) mul_lane (
            .i_mode(i_mode_mul), .i_a(a), .i_b(b),
            .o_w0(w0), .o_w1(w1), .o_s0(s0), .o_s1(s1), .o_t(t)
        );

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                r_w0[j] <= 18'd0;
                r_w1[j] <= 18'd0;
                r_s0[j] <= 10'd0;
                r_s1[j] <= 10'd0;
                r_t[j] <= 9'd0;
            end else if (i_en_mul) begin
                r_w0[j] <= w0;
                r_w1[j] <= w1;
                r_s0[j] <= s0;
                r_s1[j] <= s1;
                r_t[j] <= t;
            end
        end

        wire [17:0] wide0 = r_w0[j], wide1 = r_w1[j];
        wire [9:0] small0 = r_s0[j], small1 = r_s1[j];
        wire [8:0] exp_sum = r_t[j];
        wire signed [20:0] bf16_folded =
            $signed({{3{wide0[17]}}, wide0}) <<< exp_sum[1:0];
        reg [35:0] prod;
        reg [6:0] q0;
        reg [3:0] q1;
        always @* begin
            prod = 36'd0;
            q0 = 7'd0;
            q1 = 4'd0;
            case (i_mode_prod)
                FP4: prod = {small1[8:0], wide1[8:0], small0[8:0], wide0[8:0]};
                INT4: prod = {small1[7], small1[7:0], wide1[7], wide1[7:0],
                              small0[7], small0[7:0], wide0[7], wide0[7:0]};
                FP8, INT8: begin
                    prod = {4'd0, wide1[15:0], wide0[15:0]};
                    if (i_mode_prod == FP8) begin
                        q0 = {3'b0, exp_sum[3:0]};
                        q1 = exp_sum[7:4];
                    end
                end
                BF16: begin
                    if (wide0 != 18'd0) begin
                        prod = {15'd0, bf16_folded};
                        q0 = exp_sum[8:2];
                    end
                end
                default: begin end
            endcase
        end

        assign o_prod[j*36 +: 36] = prod;
        assign o_q0[j*7 +: 7] = q0;
        assign o_q1[j*4 +: 4] = q1;
    end endgenerate
endmodule
