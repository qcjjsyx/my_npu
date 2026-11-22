// R_TREE_LO, R_TREE_HI and R_SUM.
module mac_reduce (
    input  wire         clk,
    input  wire         rst_n,
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
    mac_sum_tree tree_a (
        .clk(clk), .rst_n(rst_n), .i_en_lo(i_en_lo), .i_en_hi(i_en_hi),
        .i_leaves(i_tree_a), .o_sum(sum_a)
    );
    mac_sum_tree tree_b (
        .clk(clk), .rst_n(rst_n), .i_en_lo(i_en_lo), .i_en_hi(i_en_hi),
        .i_leaves(i_tree_b), .o_sum(sum_b)
    );

    reg signed [9:0] r_e_lo_a, r_e_lo_b, r_e_hi_a, r_e_hi_b;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_e_lo_a <= 10'd0;
            r_e_lo_b <= 10'd0;
        end else if (i_en_lo) begin
            r_e_lo_a <= i_e_a;
            r_e_lo_b <= i_e_b;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_e_hi_a <= 10'd0;
            r_e_hi_b <= 10'd0;
        end else if (i_en_hi) begin
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

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_n_a <= 41'd0;
            o_n_b <= 41'd0;
            o_e_a <= 10'd0;
            o_e_b <= 10'd0;
            o_h_a <= 9'd0;
            o_h_b <= 9'd0;
        end else if (i_en_sum) begin
            o_n_a <= n_a;
            o_n_b <= n_b;
            o_e_a <= n_a == 41'sd0 ? 10'sd0 : r_e_hi_a;
            o_e_b <= n_b == 41'sd0 ? 10'sd0 : r_e_hi_b;
            o_h_a <= n_a == 41'sd0 ? 9'sd0 : i_h_a;
            o_h_b <= n_b == 41'sd0 ? 9'sd0 : i_h_b;
        end
    end
endmodule
