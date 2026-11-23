// R_MAX and R_ALIGN. R_ALIGN stores the two routed vectors of tree leaves.
module mac_align (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         i_en_max,
    input  wire         i_en_align,
    input  wire [2:0]   i_mode_max,
    input  wire [2:0]   i_mode_align,
    input  wire         i_mx_align,
    input  wire [575:0] i_prod,
    input  wire [111:0] i_q0,
    input  wire [63:0]  i_q1,
    output reg  [511:0] o_tree_a,
    output reg  [511:0] o_tree_b,
    output reg signed [9:0] o_e_a,
    output reg signed [9:0] o_e_b
);
    localparam [2:0] FP4 = 3'd0, FP8 = 3'd1, BF16 = 3'd2, FP16 = 3'd3,
                     INT4 = 3'd4, INT8 = 3'd5, INT16 = 3'd6;

    wire [15:0] nz_a, nz_b;
    wire any_a, any_b;
    wire [6:0] max_a;
    wire [3:0] max_b;
    mac_max_tree #(.Q_WIDTH(7)) max_tree_a (
        .i_q(i_q0), .i_nz(nz_a), .o_max(max_a), .o_nz(any_a)
    );
    mac_max_tree #(.Q_WIDTH(4)) max_tree_b (
        .i_q(i_q1), .i_nz(nz_b), .o_max(max_b), .o_nz(any_b)
    );

    reg [575:0] r_prod;
    reg [111:0] r_q0;
    reg [63:0] r_q1;
    reg [15:0] r_nz_a, r_nz_b;
    reg [6:0] r_max_a;
    reg [3:0] r_max_b;
    reg r_any_a, r_any_b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_prod <= 576'd0;
            r_q0 <= 112'd0;
            r_q1 <= 64'd0;
            r_nz_a <= 16'd0;
            r_nz_b <= 16'd0;
            r_max_a <= 7'd0;
            r_max_b <= 4'd0;
            r_any_a <= 1'b0;
            r_any_b <= 1'b0;
        end else if (i_en_max) begin
            r_prod <= i_prod;
            r_q0 <= i_q0;
            r_q1 <= i_q1;
            r_nz_a <= nz_a;
            r_nz_b <= nz_b;
            r_max_a <= max_a;
            r_max_b <= max_b;
            r_any_a <= any_a;
            r_any_b <= any_b;
        end
    end

    genvar j;
    generate for (j = 0; j < 16; j = j+1) begin : lane
        localparam A_WIDTH = j < 8 ? 38 : 32;
        wire [35:0] prod_j = i_prod[j*36 +: 36];
        reg nz_a_j;
        always @* begin
            case (i_mode_max)
                FP8: nz_a_j = prod_j[15:0] != 16'd0;
                BF16: nz_a_j = prod_j[20:0] != 21'd0;
                FP16: nz_a_j = j < 8 && prod_j[29:0] != 30'd0;
                default: nz_a_j = 1'b0;
            endcase
        end
        assign nz_a[j] = nz_a_j;
        assign nz_b[j] = i_mode_max == FP8 ? (prod_j[31:16] != 16'd0) : 1'b0;

        wire [35:0] r_prod_j = r_prod[j*36 +: 36];
        wire use_a = r_nz_a[j];
        wire use_b = r_nz_b[j];
        wire [6:0] diff_a = use_a ? (r_max_a - r_q0[j*7 +: 7]) : 7'd0;
        wire [3:0] diff_b = use_b ? (r_max_b - r_q1[j*4 +: 4]) : 4'd0;
        reg signed [A_WIDTH-1:0] shift_in_a;
        wire signed [31:0] shift_in_b = {r_prod_j[31:16], 16'd0};

        always @* begin
            case (i_mode_align)
                FP8:  shift_in_a = $signed({r_prod_j[15:0], 16'd0});
                BF16: shift_in_a = $signed({r_prod_j[20:0], 11'd0});
                FP16: shift_in_a = $signed({r_prod_j[29:0], 8'd0});
                default: shift_in_a = {A_WIDTH{1'b0}};
            endcase
        end

        // Only shift within the word. Greater distances use signed fill.
        wire signed [A_WIDTH-1:0] aligned_a;
        if (j < 8) begin : wide_shift
            wire far_a = diff_a >= 7'd10;
            assign aligned_a = !use_a ? 38'sd0 :
                (far_a ? $signed({38{shift_in_a[37]}}) : (shift_in_a >>> {diff_a[3:0], 2'b00}));
        end else begin : narrow_shift
            wire far_a = |diff_a[6:3];
            assign aligned_a = !use_a ? 32'sd0 :
                (far_a ? $signed({32{shift_in_a[31]}}) : (shift_in_a >>> {diff_a[2:0], 2'b00}));
        end
        wire far_b = diff_b[3];
        wire signed [31:0] aligned_b = !use_b ? 32'sd0 :
            (far_b ? $signed({32{shift_in_b[31]}}) : (shift_in_b >>> {diff_b[2:0], 2'b00}));
        wire [37:0] wide_a = $signed(aligned_a);
        wire [8:0] p0 = r_prod_j[8:0], p1 = r_prod_j[17:9];
        wire [8:0] p2 = r_prod_j[26:18], p3 = r_prod_j[35:27];
        reg [31:0] tree_a, tree_b;
        always @* begin
            tree_a = 32'd0;
            tree_b = 32'd0;
            case (i_mode_align)
                FP4: begin
                    tree_a = {{9{p2[8]}}, p2[8:0], 5'd0, (p0[8:0] ^ 9'h100)};
                    tree_b = {{9{p3[8]}}, p3[8:0], 5'd0, (p1[8:0] ^ 9'h100)};
                end
                INT4: begin
                    tree_a = {{11{p2[7]}}, p2[7:0], 5'd0, (p0[7:0] ^ 8'h80)};
                    tree_b = {{11{p3[7]}}, p3[7:0], 5'd0, (p1[7:0] ^ 8'h80)};
                end
                FP8: begin
                    tree_a = aligned_a[31:0];
                    tree_b = aligned_b;
                end
                BF16: tree_a = aligned_a[31:0];
                FP16: if (j < 8) begin
                    tree_a = {13'd0, wide_a[18:0]};
                    tree_b = {{13{wide_a[37]}}, wide_a[37:19]};
                end
                INT8: begin
                    tree_a = {{16{r_prod_j[15]}}, r_prod_j[15:0]};
                    tree_b = {{16{r_prod_j[31]}}, r_prod_j[31:16]};
                end
                INT16: if (j < 8) tree_a = r_prod_j[31:0];
                default: begin end
            endcase
        end

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                o_tree_a[j*32 +: 32] <= 32'd0;
                o_tree_b[j*32 +: 32] <= 32'd0;
            end else if (i_en_align) begin
                o_tree_a[j*32 +: 32] <= tree_a;
                o_tree_b[j*32 +: 32] <= tree_b;
            end
        end
    end endgenerate

    wire signed [9:0] e_base_a = $signed({1'b0, r_max_a, 2'b00});
    wire signed [9:0] e_base_b = $signed({4'b0, r_max_b, 2'b00});
    reg signed [9:0] e_a, e_b;
    always @* begin
        e_a = 10'sd0;
        e_b = 10'sd0;
        case (i_mode_align)
            FP4: begin e_a = -10'sd2; e_b = -10'sd2; end
            FP8: begin e_a = e_base_a - 10'sd52; e_b = e_base_b - 10'sd52; end
            BF16: e_a = e_base_a - 10'sd279;
            FP16: e_a = e_base_a - 10'sd58;
            INT8: if (i_mx_align) begin e_a = -10'sd12; e_b = -10'sd12; end
            default: begin end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_e_a <= 10'd0;
            o_e_b <= 10'd0;
        end else if (i_en_align) begin
            o_e_a <= e_a;
            o_e_b <= e_b;
        end
    end
endmodule
