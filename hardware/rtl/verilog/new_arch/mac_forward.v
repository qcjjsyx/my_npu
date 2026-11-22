// Fixed-latency forward datapath. Input is registered at E0 and R_MX at E8.
module new_arch_mac_forward (
    input  wire         clk,
    input  wire         rst_n,
    input  wire [351:0] i_a_fields,
    input  wire [351:0] i_b_fields,
    input  wire [2:0]   i_mode,
    input  wire         i_mx,
    input  wire [15:0]  i_scale_a,
    input  wire [15:0]  i_scale_b,
    input  wire [1:0]   i_ctx,
    input  wire         i_first,
    input  wire         i_last,
    input  wire         i_valid,
    output wire [7:0]   o_meta,
    output wire         o_valid,
    output wire         o_pipe_empty,
    output wire signed [40:0] o_m_v0,
    output wire signed [40:0] o_m_v1,
    output wire signed [9:0]  o_h_v0,
    output wire signed [9:0]  o_h_v1,
    output wire signed [44:0] o_i_v0,
    output wire signed [44:0] o_i_v1
);
    // r_valid[j] and r_meta[j] correspond to registered stages E0 through E8.
    reg [8:0] r_valid;
    reg [7:0] r_meta [0:8];
    reg [351:0] r_a, r_b;
    reg signed [8:0] r_h_a [0:6];
    reg signed [8:0] r_h_b [0:6];

    function signed [9:0] scale_sum;
        input [7:0] a, b;
        begin
            scale_sum = $signed({2'b0, a}) + $signed({2'b0, b}) - 10'sd254;
        end
    endfunction

    // All legal MX modes have two active groups.
    wire signed [9:0] h_wide_a = scale_sum(i_scale_a[7:0], i_scale_b[7:0]);
    wire signed [9:0] h_wide_b = scale_sum(i_scale_a[15:8], i_scale_b[15:8]);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_valid <= 9'd0;
            r_a <= 352'd0;
            r_b <= 352'd0;
            r_meta[0] <= 8'd0;
            r_h_a[0] <= 9'd0;
            r_h_b[0] <= 9'd0;
        end else begin
            r_valid <= {r_valid[7:0], i_valid};
            if (i_valid) begin
                r_a <= i_a_fields;
                r_b <= i_b_fields;
                r_meta[0] <= {i_mode, i_mx, i_ctx, i_first, i_last};
                r_h_a[0] <= i_mx ? h_wide_a[8:0] : 9'd0;
                r_h_b[0] <= i_mx ? h_wide_b[8:0] : 9'd0;
            end
        end
    end

    wire [7:0] advance = {8{rst_n}} & r_valid[7:0];
    genvar j;
    generate
        for (j = 1; j <= 8; j = j+1) begin : metadata
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) r_meta[j] <= 8'd0;
                else if (advance[j-1]) r_meta[j] <= r_meta[j-1];
            end
        end
        for (j = 1; j <= 6; j = j+1) begin : scale
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    r_h_a[j] <= 9'd0;
                    r_h_b[j] <= 9'd0;
                end else if (advance[j-1]) begin
                    r_h_a[j] <= r_h_a[j-1];
                    r_h_b[j] <= r_h_b[j-1];
                end
            end
        end
    endgenerate

    wire [511:0] p0, tree_a, tree_b;
    wire [143:0] p1, p3;
    wire [255:0] p2;
    wire [111:0] q0, q1;
    wire signed [9:0] align_e_a, align_e_b, sum_e_a, sum_e_b;
    wire signed [8:0] sum_h_a, sum_h_b;
    wire signed [40:0] n_a, n_b;
    new_arch_mac_mul_frontend #(.CLUSTER_COUNT(8)) mul (
        .clk(clk), .rst_n(rst_n), .i_en_mul(advance[0]), .i_en_prod(advance[1]),
        .i_mode_mul(r_meta[0][7:5]), .i_mode_prod(r_meta[1][7:5]),
        .i_a(r_a), .i_b(r_b), .o_p0(p0), .o_p1(p1), .o_p2(p2), .o_p3(p3),
        .o_q0(q0), .o_q1(q1)
    );
    new_arch_mac_align align (
        .clk(clk), .rst_n(rst_n), .i_en_max(advance[2]), .i_en_align(advance[3]),
        .i_mode_max(r_meta[2][7:5]), .i_mode_align(r_meta[3][7:5]), .i_mx_align(r_meta[3][4]),
        .i_p0(p0), .i_p1(p1), .i_p2(p2), .i_p3(p3), .i_q0(q0), .i_q1(q1),
        .o_tree_a(tree_a), .o_tree_b(tree_b), .o_e_a(align_e_a), .o_e_b(align_e_b)
    );
    new_arch_mac_reduce reduce (
        .clk(clk), .rst_n(rst_n), .i_en_lo(advance[4]), .i_en_hi(advance[5]), .i_en_sum(advance[6]),
        .i_mode_sum(r_meta[6][7:5]), .i_tree_a(tree_a), .i_tree_b(tree_b),
        .i_e_a(align_e_a), .i_e_b(align_e_b), .i_h_a(r_h_a[6]), .i_h_b(r_h_b[6]),
        .o_n_a(n_a), .o_n_b(n_b), .o_e_a(sum_e_a), .o_e_b(sum_e_b),
        .o_h_a(sum_h_a), .o_h_b(sum_h_b)
    );
    new_arch_mac_mx mx (
        .clk(clk), .rst_n(rst_n), .i_en(advance[7]), .i_mode(r_meta[7][7:5]), .i_mx(r_meta[7][4]),
        .i_n_a(n_a), .i_n_b(n_b), .i_e_a(sum_e_a), .i_e_b(sum_e_b),
        .i_h_a(sum_h_a), .i_h_b(sum_h_b),
        .o_m_a(o_m_v0), .o_m_b(o_m_v1), .o_h_a(o_h_v0), .o_h_b(o_h_v1),
        .o_i_a(o_i_v0), .o_i_b(o_i_v1)
    );

    assign o_meta = r_meta[8];
    assign o_valid = r_valid[8];
    assign o_pipe_empty = (r_valid == 9'd0);
endmodule
