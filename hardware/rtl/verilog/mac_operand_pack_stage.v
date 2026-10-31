module mac_operand_pack_stage (
    input mac_clk,
    input rstn,
    input  [15:0] i_raw,
    input  [2:0] i_src_fmt,
    input  [1:0] i_group_valid,
    output reg [25:0] o_payload
);

//0. 格式解包
wire fp9_selected, fp19_selected;
wire [15:0] fp19_din;
wire [7:0] fp9_din0, fp9_din1;

assign fp9_selected = (i_src_fmt == 3'd2) || (i_src_fmt == 3'd3);
assign fp19_selected = (i_src_fmt == 3'd0) || (i_src_fmt == 3'd1);
assign fp19_din = (i_group_valid[0]&&fp19_selected)? i_raw : 16'b0;
assign fp9_din0 = (i_group_valid[0]&&fp9_selected)? i_raw[7:0] : 8'b0;
assign fp9_din1 = (i_group_valid[1]&&fp9_selected)? i_raw[15:8] : 8'b0;

//1. 格式转换
wire[19:0] fp19_dout;
wire [9:0] fp9_dout0, fp9_dout1;
mac_fp19_convert u_mac_fp19_convert(
    .i_raw(fp19_din),
    .i_src_fmt(i_src_fmt),
    .o_eff_exp(fp19_dout[18:11]),
    .o_mantissa(fp19_dout[10:0])
);
mac_fp9_convert u_mac_fp9_convert0(
    .i_raw(fp9_din0),
    .i_src_fmt(i_src_fmt),
    .o_eff_exp(fp9_dout0[8:4]),
    .o_mantissa(fp9_dout0[3:0])
);
mac_fp9_convert u_mac_fp9_convert1(
    .i_raw(fp9_din1),
    .i_src_fmt(i_src_fmt),
    .o_eff_exp(fp9_dout1[8:4]),
    .o_mantissa(fp9_dout1[3:0])
);

//2. 打包
wire [25:0] packed_data;
always @(*) begin
    packed_data = 26'b0;
    case (i_src_fmt)
        3'd0: packed_data = {2'b00,fp19_dout[19:12],4'b0,fp19_dout[11:0]}; //FP16
        3'd1: packed_data = {2'b00,fp19_dout[19:12],4'b0,fp19_dout[11:0]}; //BF16
        3'd2: packed_data = {fp9_dout1[9:5],fp9_dout0[9:5],fp9_dout1[4:0],fp9_dout0[4:0]}; //E4M3
        3'd3: packed_data = {fp9_dout1[9:5],fp9_dout0[9:5],fp9_dout1[4:0],fp9_dout0[4:0]}; //E5M2
        3'd4: packed_data = {10'b0,16{i_group_valid[0]}&{i_raw}}; //INT16
        3'd5: packed_data = {10'b0,8{i_group_valid[1]}&{i_raw[15:8]},8{i_group_valid[0]}&{i_raw[7:0]}}; //INT8
        3'd6: packed_data = {18'b0,4{i_group_valid[1]}&{i_raw[7:4]},4{i_group_valid[0]}&{i_raw[3:0]}}; //INT4
        default: packed_data = 26'b0;
    endcase
end

// 寄存输出
always @(posedge mac_clk or negedge rstn) begin
    if(!rstn) begin
        o_payload <=26'b0;
    end else begin
        o_payload <= packed_data;
    end
end


endmodule
