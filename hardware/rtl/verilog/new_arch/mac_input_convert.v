module mac_input_convert (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] i_raw,
    input  wire [2:0]  i_fmt,
    input  wire        i_valid,
    output reg  [21:0] o_fields,
    output reg         o_valid
);
    localparam [2:0] FMT_FP4    = 3'd0,
                     FMT_E4M3FN = 3'd1,
                     FMT_E5M2   = 3'd2,
                     FMT_BF16   = 3'd3,
                     FMT_FP16   = 3'd4,
                     FMT_INT4   = 3'd5,
                     FMT_INT8   = 3'd6,
                     FMT_INT16  = 3'd7;

    function [4:0] fp4_field;
        input [3:0] raw;
        reg signed [4:0] magnitude;
        begin
            // The coefficient is q = 2x.
            case (raw[2:0])
                3'd0: magnitude = 5'sd0;
                3'd1: magnitude = 5'sd1;
                3'd2: magnitude = 5'sd2;
                3'd3: magnitude = 5'sd3;
                3'd4: magnitude = 5'sd4;
                3'd5: magnitude = 5'sd6;
                3'd6: magnitude = 5'sd8;
                3'd7: magnitude = 5'sd12;
                default: magnitude = 5'bxxxxx;
            endcase
            fp4_field = raw[3] ? -magnitude : magnitude;
        end
    endfunction

    function [10:0] fp8_field;
        input [7:0] raw;
        input       e5m2;
        reg [4:0] exp_raw;
        reg [2:0] frac3;
        reg       hidden;
        reg [4:0] exp_unified;
        reg signed [7:0] coeff;
        begin
            exp_raw = e5m2 ? raw[6:2] : {1'b0, raw[6:3]};
            frac3 = e5m2 ? {raw[1:0], 1'b0} : raw[2:0];
            if ((e5m2 && exp_raw == 5'd31 && frac3 != 3'd0) ||
                (!e5m2 && exp_raw == 5'd15 && frac3 == 3'd7) ||
                (exp_raw == 5'd0 && frac3 == 3'd0)) begin
                fp8_field = 11'd0;
            end else begin
                if (e5m2 && exp_raw == 5'd31) begin
                    exp_raw = 5'd30;
                    frac3 = 3'b110;
                end
                hidden = (exp_raw != 5'd0);
                exp_unified = hidden ? exp_raw : 5'd1;
                if (!e5m2)
                    exp_unified = exp_unified + 5'd8;
                coeff = $signed({4'b0000, hidden, frac3});
                coeff = (raw[7] ? -coeff : coeff) <<< exp_unified[1:0];
                fp8_field = {exp_unified[4:2], coeff};
            end
        end
    endfunction

    function [16:0] bf16_field;
        input [15:0] raw;
        reg [7:0] exp_raw;
        reg [6:0] frac;
        reg       hidden;
        reg [7:0] exp_eff;
        reg signed [8:0] coeff;
        begin
            exp_raw = raw[14:7];
            frac = raw[6:0];
            if ((exp_raw == 8'hff && frac != 7'd0) ||
                (exp_raw == 8'd0 && frac == 7'd0)) begin
                bf16_field = 17'd0;
            end else begin
                if (exp_raw == 8'hff) begin
                    exp_raw = 8'hfe;
                    frac = 7'h7f;
                end
                hidden = (exp_raw != 8'd0);
                exp_eff = hidden ? exp_raw : 8'd1;
                coeff = $signed({1'b0, hidden, frac});
                bf16_field = {exp_eff, (raw[15] ? -coeff : coeff)};
            end
        end
    endfunction

    function [17:0] fp16_field;
        input [15:0] raw;
        reg [4:0] exp_raw;
        reg [9:0] frac;
        reg       hidden;
        reg [4:0] exp_eff;
        reg signed [14:0] coeff;
        begin
            exp_raw = raw[14:10];
            frac = raw[9:0];
            if ((exp_raw == 5'd31 && frac != 10'd0) ||
                (exp_raw == 5'd0 && frac == 10'd0)) begin
                fp16_field = 18'd0;
            end else begin
                if (exp_raw == 5'd31) begin
                    exp_raw = 5'd30;
                    frac = 10'h3ff;
                end
                hidden = (exp_raw != 5'd0);
                exp_eff = hidden ? exp_raw : 5'd1;
                coeff = $signed({4'b0000, hidden, frac});
                coeff = (raw[15] ? -coeff : coeff) <<< exp_eff[1:0];
                fp16_field = {exp_eff[4:2], coeff};
            end
        end
    endfunction

    reg [21:0] fields;
    wire use_e5m2 = (i_fmt == FMT_E5M2);

    always @* begin
        case (i_fmt)
            FMT_FP4: fields = {2'b00, fp4_field(i_raw[15:12]), fp4_field(i_raw[11:8]),
                                    fp4_field(i_raw[7:4]), fp4_field(i_raw[3:0])};
            FMT_E4M3FN, FMT_E5M2: fields = {fp8_field(i_raw[15:8], use_e5m2),
                                           fp8_field(i_raw[7:0], use_e5m2)};
            FMT_BF16: fields = {5'b00000, bf16_field(i_raw)};
            FMT_FP16: fields = {4'b0000, fp16_field(i_raw)};
            FMT_INT4: fields = {2'b00, i_raw[15], i_raw[15:12], i_raw[11], i_raw[11:8],
                                       i_raw[7], i_raw[7:4], i_raw[3], i_raw[3:0]};
            FMT_INT8, FMT_INT16: fields = {6'b000000, i_raw};
            default: fields = 22'bx;
        endcase
    end

    // rst_n asserts asynchronously and is released by the clk-domain reset source.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_fields <= 22'd0;
            o_valid <= 1'b0;
        end else begin
            o_valid <= i_valid;
            if (i_valid)
                o_fields <= fields;
        end
    end
endmodule
