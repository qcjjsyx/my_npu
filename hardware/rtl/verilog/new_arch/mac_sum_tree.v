module mac_sum_tree (
    input  wire         clk,
    input  wire         rst_n,
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
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) r_lo[j*34 +: 34] <= 34'd0;
                else if (i_en_lo) r_lo[j*34 +: 34] <= sum;
            end
        end
        for (j = 0; j < 2; j = j+1) begin : third_level
            assign level3[j] = $signed(r_lo[(2*j)*34 +: 34])
                + $signed(r_lo[(2*j+1)*34 +: 34]);
        end
    endgenerate
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) o_sum <= 36'd0;
        else if (i_en_hi) o_sum <= level3[0] + level3[1];
    end
endmodule
