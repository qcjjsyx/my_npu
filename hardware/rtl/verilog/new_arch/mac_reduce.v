// R_TREE_LO, R_TREE_HI and R_SUM.
module new_arch_mac_reduce (
    input  wire         clk,
    input  wire         i_en_lo,
    input  wire         i_en_hi,
    input  wire         i_en_sum,
    input  wire [2:0]   i_mode_sum,
    input  wire [511:0] i_tree_a,
    input  wire [511:0] i_tree_b,
    input  wire signed [9:0] i_e_a,
    input  wire signed [9:0] i_e_b,
    input  wire signed [8:0] i_h_a,
    input  wire signed [8:0] i_h_b,
    output reg signed [40:0] o_n_a,
    output reg signed [40:0] o_n_b,
    output reg signed [9:0] o_e_a,
    output reg signed [9:0] o_e_b,
    output reg signed [8:0] o_h_a,
    output reg signed [8:0] o_h_b
);
    localparam [2:0] FP4 = 3'd0, FP8 = 3'd1, FP16 = 3'd3, INT4 = 3'd4, INT8 = 3'd5;
    wire signed [35:0] sum_a, sum_b;
    new_arch_mac_sum_tree tree_a (
        .clk(clk), .i_en_lo(i_en_lo), .i_en_hi(i_en_hi), .i_leaves(i_tree_a), .o_sum(sum_a)
    );
    new_arch_mac_sum_tree tree_b (
        .clk(clk), .i_en_lo(i_en_lo), .i_en_hi(i_en_hi), .i_leaves(i_tree_b), .o_sum(sum_b)
    );

    reg signed [9:0] r_e_lo_a, r_e_lo_b, r_e_hi_a, r_e_hi_b;
    always @(posedge clk) begin
        if (i_en_lo) begin
            r_e_lo_a <= i_e_a;
            r_e_lo_b <= i_e_b;
        end
        if (i_en_hi) begin
            r_e_hi_a <= r_e_lo_a;
            r_e_hi_b <= r_e_lo_b;
        end
    end

    // FP16 supplies an unsigned22 low sum and a signed22 high sum.
    wire split_fp16 = i_mode_sum == FP16;
    wire signed [40:0] final_a = split_fp16 ? $signed({19'd0, sum_a[21:0]}) : sum_a;
    wire signed [40:0] final_b = split_fp16 ? $signed({sum_b[21:0], 19'd0}) : sum_b;
    wire signed [40:0] final_sum = final_a + final_b;
    reg signed [40:0] n_a, n_b;

    always @* begin
        n_a = sum_a;
        n_b = 41'sd0;
        case (i_mode_sum)
            FP4: begin
                n_a = $signed(final_sum[13:0] ^ 14'h2000);
                n_b = $signed(final_sum[27:14]);
            end
            INT4: begin
                n_a = $signed(final_sum[12:0] ^ 13'h1000);
                n_b = $signed(final_sum[25:13]);
            end
            FP8, INT8: n_b = sum_b;
            FP16: n_a = final_sum;
            default: begin end
        endcase
    end

    always @(posedge clk) begin
        if (i_en_sum) begin
            o_n_a <= n_a;
            o_n_b <= n_b;
            o_e_a <= n_a == 41'sd0 ? 10'sd0 : r_e_hi_a;
            o_e_b <= n_b == 41'sd0 ? 10'sd0 : r_e_hi_b;
            o_h_a <= n_a == 41'sd0 ? 9'sd0 : i_h_a;
            o_h_b <= n_b == 41'sd0 ? 9'sd0 : i_h_b;
        end
    end
endmodule

module new_arch_mac_sum_tree (
    input  wire         clk,
    input  wire         i_en_lo,
    input  wire         i_en_hi,
    input  wire [511:0] i_leaves,
    output reg signed [35:0] o_sum
);
    wire signed [32:0] level1 [0:7];
    reg [135:0] r_lo;
    wire signed [34:0] level3 [0:1];
    genvar j;
    generate
        for (j = 0; j < 8; j = j+1) begin : first_level
            assign level1[j] = $signed(i_leaves[(2*j)*32 +: 32])
                + $signed(i_leaves[(2*j+1)*32 +: 32]);
        end
        for (j = 0; j < 4; j = j+1) begin : second_level
            wire signed [33:0] sum = level1[2*j] + level1[2*j+1];
            always @(posedge clk) begin
                if (i_en_lo) r_lo[j*34 +: 34] <= sum;
            end
        end
        for (j = 0; j < 2; j = j+1) begin : third_level
            assign level3[j] = $signed(r_lo[(2*j)*34 +: 34])
                + $signed(r_lo[(2*j+1)*34 +: 34]);
        end
    endgenerate
    always @(posedge clk) begin
        if (i_en_hi) o_sum <= level3[0] + level3[1];
    end
endmodule
