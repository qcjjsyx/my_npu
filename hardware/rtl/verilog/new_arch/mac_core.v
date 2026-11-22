module mac_core (
    input  wire clk,
    input  wire rst_n,
    input  wire [351:0] i_a_fields,
    input  wire [351:0] i_b_fields,
    input  wire [2:0] i_mode,
    input  wire i_mx,
    input  wire [15:0] i_scale_a,
    input  wire [15:0] i_scale_b,
    input  wire [1:0] i_ctx,
    input  wire i_first,
    input  wire i_last,
    input  wire i_valid,
    output wire [2:0] o_ctx_free,
    output wire [2:0] o_ctx_done,
    input  wire [1:0] i_result_ctx,
    input  wire i_result_ready,
    output wire o_result_valid,
    output wire [50:0] o_state0,
    output wire [50:0] o_state1,
    output wire [2:0] o_result_mode,
    output wire o_result_mx,
    output wire o_pipe_empty
);
    function two_groups;
        input [2:0] mode;
        begin
            case (mode)
                3'd0, 3'd1, 3'd4, 3'd5: two_groups = 1'b1;
                default: two_groups = 1'b0;
            endcase
        end
    endfunction

    wire [7:0] mx_meta;
    wire mx_valid, forward_empty;
    wire signed [40:0] m_v0, m_v1;
    wire signed [9:0] h_v0, h_v1;
    wire signed [44:0] i_v0, i_v1;
    mac_forward forward (
        .clk(clk), .rst_n(rst_n), .i_a_fields(i_a_fields), .i_b_fields(i_b_fields),
        .i_mode(i_mode), .i_mx(i_mx), .i_scale_a(i_scale_a), .i_scale_b(i_scale_b),
        .i_ctx(i_ctx), .i_first(i_first), .i_last(i_last), .i_valid(i_valid),
        .o_meta(mx_meta), .o_valid(mx_valid), .o_pipe_empty(forward_empty),
        .o_m_v0(m_v0), .o_m_v1(m_v1), .o_h_v0(h_v0), .o_h_v1(h_v1),
        .o_i_v0(i_v0), .o_i_v1(i_v1)
    );

    reg [1:0] acc_valid;
    reg [7:0] acc_meta_s1, acc_meta_s2;
    wire en_s1 = rst_n && mx_valid;
    wire en_s2 = rst_n && acc_valid[0];
    wire en_s3 = rst_n && acc_valid[1];
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_valid <= 2'b00;
            acc_meta_s1 <= 8'd0;
            acc_meta_s2 <= 8'd0;
        end else begin
            acc_valid <= {acc_valid[0], mx_valid};
            if (en_s1) acc_meta_s1 <= mx_meta;
            if (en_s2) acc_meta_s2 <= acc_meta_s1;
        end
    end

    wire float_s1 = !mx_meta[7] || mx_meta[4];
    wire float_s3 = !acc_meta_s2[7] || acc_meta_s2[4];
    wire en1_s1 = en_s1 && two_groups(mx_meta[7:5]);
    wire en1_s2 = en_s2 && two_groups(acc_meta_s1[7:5]);
    wire en1_s3 = en_s3 && two_groups(acc_meta_s2[7:5]);
    wire [50:0] read_state0, read_state1;
    mac_acc_lane acc0 (
        .clk(clk), .rst_n(rst_n), .i_en_s1(en_s1), .i_en_s2(en_s2), .i_en_s3(en_s3),
        .i_ctx_s1(mx_meta[3:2]), .i_first_s1(mx_meta[1]), .i_float_s1(float_s1),
        .i_m(m_v0), .i_h(h_v0), .i_int(i_v0),
        .i_ctx_s3(acc_meta_s2[3:2]), .i_float_s3(float_s3),
        .i_result_ctx(i_result_ctx), .o_result_state(read_state0)
    );
    mac_acc_lane acc1 (
        .clk(clk), .rst_n(rst_n), .i_en_s1(en1_s1), .i_en_s2(en1_s2), .i_en_s3(en1_s3),
        .i_ctx_s1(mx_meta[3:2]), .i_first_s1(mx_meta[1]), .i_float_s1(float_s1),
        .i_m(m_v1), .i_h(h_v1), .i_int(i_v1),
        .i_ctx_s3(acc_meta_s2[3:2]), .i_float_s3(float_s3),
        .i_result_ctx(i_result_ctx), .o_result_state(read_state1)
    );

    assign o_pipe_empty = forward_empty && acc_valid == 2'b00;
    mac_ctx_control contexts (
        .clk(clk), .rst_n(rst_n), .i_mode(i_mode), .i_mx(i_mx), .i_ctx(i_ctx),
        .i_first(i_first), .i_last(i_last), .i_valid(i_valid), .i_pipe_empty(o_pipe_empty),
        .i_write_last(en_s3 && acc_meta_s2[0]), .i_write_ctx(acc_meta_s2[3:2]),
        .i_result_ctx(i_result_ctx), .i_result_ready(i_result_ready),
        .o_ctx_free(o_ctx_free), .o_ctx_done(o_ctx_done), .o_result_valid(o_result_valid),
        .o_result_mode(o_result_mode), .o_result_mx(o_result_mx)
    );
    assign o_state0 = o_result_valid ? read_state0 : 51'd0;
    assign o_state1 = (o_result_valid && two_groups(o_result_mode)) ? read_state1 : 51'd0;
endmodule
