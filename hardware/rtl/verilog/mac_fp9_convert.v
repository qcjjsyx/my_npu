module mac_fp9_convert(
    input [7:0] i_raw,
    input [2:0] i_src_fmt,
    output [4:0] o_eff_exp,
    output [4:0] o_mantissa
);

//0. 格式解包
wire src_sign;
wire [4:0] src_exp;
wire [2:0] src_significand;
wire is_zero,is_inf,is_nan,is_subnormal;
wire [7:0] max_finited;
always @(*) begin
    src_sign = 1'b0;
    src_exp = '0;
    src_significand = '0;
    is_nan = 1'b0;
    is_inf = 1'b0;
    is_zero = 1'b0;
    is_subnormal = 1'b0;
    max_finited = '0;
    case (i_src_fmt)
    3'b010: begin. //E4M3
        src_sign = i_raw[7];
        src_exp = {1'b0,i_raw[6:3]};
        src_significand = {i_raw[2:0]};
        is_nan = (src_exp == 5'd15) && (src_significand == 3'b111);
        // is_inf = (src_exp == 8'hff) && (src_significand == 10'b0);
        is_zero = (src_exp == 5'd0) && (src_significand == 3'b0);
        is_subnormal = (src_exp == 5'd0) && (src_significand != 3'b0);
        max_finited = 8'b01111110;
    end
    3'b011: begin //E5M2
        src_sign = i_raw[7];
        src_exp = {i_raw[6:2]};
        src_significand = {i_raw[1:0],1'b0};
        is_nan = (src_exp == 5'd31) && (src_significand != 3'b000);
        is_inf = (src_exp == 5'd31) && (src_significand == 3'b000);
        is_zero = (src_exp == 5'd0) && (src_significand == 3'b000);
        is_subnormal = (src_exp == 5'd0) && (src_significand != 3'b000);
        max_finited = 8'b11110110;
    end
    default: begin
        src_sign = 1'b0;
        src_exp = '0;
        src_significand = '0;
        is_nan = 1'b0;
        is_inf = 1'b0;
        is_zero = 1'b1; // 默认输出0
        is_subnormal = 1'b0;
        max_finited = '0;
    end
    endcase
end

//1. 特殊值转换
wire ftz;
wire [7:0] finited;
assign ftz = is_nan||is_zero; //nan转换为0
assign finited = is_inf? max_finited : {src_exp,src_significand}; //inf转换为最大有限值


//2. 添加隐藏位+转换为有效指数
wire [3:0] significand_with_hidden_bit;
wire [4:0] eff_exp;
assign significand_with_hidden_bit = (is_subnormal)? {1'b0,finited[2:0]} : {1'b1,finited[2:0]};
assign eff_exp = (is_subnormal? 5'b1 : finited[7:3]) 
//3. 幅值转补码
wire [3:0] mantissa;
assign mantissa = 4{src_sign}^significand_with_hidden_bit+src_sign;
//4. 输出
assign o_eff_exp = ftz? 5'b0 : eff_exp;
assign o_mantissa =ftz? 5'b0 : {src_sign,mantissa;}


endmodule`