module mac_fp19_convert(
    input [15:0] i_raw,
    input [2:0] i_src_fmt,
    output [7:0] o_eff_exp,
    output [11:0] o_mantissa

);
// 0. 格式解包
wire src_sign;
wire [7:0] src_exp;
wire [9:0] src_significand;
wire is_zero,is_inf,is_nan,is_subnormal;
wire [7:0] max_finited_exp;
wire [9:0] max_finited_significand;

always @(*) begin
    src_sign = 1'b0;
    src_exp = '0;
    src_significand = '0;
    is_nan = 1'b0;
    is_inf = 1'b0;
    is_zero = 1'b0;
    is_subnormal = 1'b0;
    max_finited_exp = '0;
    max_finited_significand = '0;
    case (i_src_fmt)
    3'b000: begin. //FP16
        src_sign = i_raw[15];
        src_exp = {3'b0,i_raw[14:10]};
        src_significand = {i_raw[9:0]};
        is_nan = (src_exp == 8'hff) && (src_significand != 10'b0);
        is_inf = (src_exp == 8'hff) && (src_significand == 10'b0);
        is_zero = (src_exp == 8'h00) && (src_significand == 10'b0);
        is_subnormal = (src_exp == 8'h00) && (src_significand != 10'b0);
        max_finited_exp = 8'd30;
        max_finited_significand = 10'h3ff;
    end
    3'b001: begin //BF16
        src_sign = i_raw[15];
        src_exp = {i_raw[14:7]};
        src_significand = {i_raw[6:0],3'b0};
        is_nan = (src_exp == 8'hff) && (src_significand != 10'b0);
        is_inf = (src_exp == 8'hff) && (src_significand == 10'b0);
        is_zero = (src_exp == 8'h00) && (src_significand == 10'b0);
        is_subnormal = (src_exp == 8'h00) && (src_significand != 10'b0);
        max_finited_exp = 8'd254;
        max_finited_significand = {7'h7f,3'b000};
    end
    default: begin
        src_sign = 1'b0;
        src_exp = '0;
        src_significand = '0;
        is_nan = 1'b0;
        is_inf = 1'b0;
        is_zero = 1'b1;
        is_subnormal = 1'b0;
        max_finited_exp = '0;
        max_finited_significand = '0; 
    end
    endcase
end

//1. 特殊值转换
wire ftz;
wire[7:0] exp;
wire[9:0] significand;
assign ftz = is_nan||is_zero; // nan转换为0
assign exp = is_inf? max_finited_exp : src_exp;
assign significand = is_inf? max_finited_significand : src_significand;
//2. 添加隐藏位+转换为有效指数
wire [10:0] significand_with_hidden_bit;
wire [7:0] eff_exp;
assign significand_with_hidden_bit = (is_subnormal)? {1'b0,significand} : {1'b1,significand};
assign eff_exp =  is_subnormal? 8'b1 : exp;
//3. 幅值转补码
wire [10:0] mantissa;
assign mantissa = 11{src_sign}^significand_with_hidden_bit+src_sign;
//4. 输出
assign o_eff_exp = ftz? 8'b0 : eff_exp;
assign o_mantissa =ftz? {1'b0,10'b0} : {src_sign,mantissa;}
endmodule