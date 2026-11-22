module mac_acc_align (
    input  wire i_float,
    input  wire [50:0] i_old,
    input  wire signed [40:0] i_m,
    input  wire signed [9:0] i_h,
    input  wire signed [44:0] i_int,
    output reg signed [44:0] o_a,
    output reg signed [44:0] o_b,
    output reg signed [9:0] o_h,
    output reg o_tail,
    output reg o_bypass,
    output reg [50:0] o_bypass_state
);
    wire signed [40:0] old_m = i_old[40:0];
    wire signed [9:0] old_h = i_old[50:41];
    wire use_align = i_float && old_m != 41'sd0 && i_m != 41'sd0;
    wire old_is_high = old_h >= i_h;
    wire signed [9:0] high_h = use_align ? (old_is_high ? old_h : i_h) : 10'sd0;
    wire signed [9:0] low_h = use_align ? (old_is_high ? i_h : old_h) : 10'sd0;
    wire signed [10:0] d = $signed({high_h[9], high_h}) - $signed({low_h[9], low_h});
    wire [40:0] high_m = use_align ? (old_is_high ? old_m : i_m) : 41'd0;
    wire [40:0] low_m = use_align ? (old_is_high ? i_m : old_m) : 41'd0;

    wire signed [42:0] shifted [0:6];
    wire [6:0] sticky;
    assign shifted[0] = {low_m, 2'b00};
    assign sticky[0] = 1'b0;
    genvar j;
    generate for (j = 0; j < 6; j = j+1) begin : shift
        localparam DISTANCE = 1 << j;
        assign shifted[j+1] = d[j] ? ($signed(shifted[j]) >>> DISTANCE) : shifted[j];
        assign sticky[j+1] = sticky[j] || (d[j] && (|shifted[j][DISTANCE-1:0]));
    end endgenerate
    wire far = d >= 11'sd43;
    wire signed [42:0] low_aligned = far ? {43{low_m[40]}} : shifted[6];

    always @* begin
        o_a = 45'sd0;
        o_b = 45'sd0;
        o_h = 10'sd0;
        o_tail = 1'b0;
        o_bypass = 1'b0;
        o_bypass_state = 51'd0;
        if (!i_float) begin
            o_a = i_old[44:0];
            o_b = i_int;
        end else if (!use_align) begin
            o_bypass = 1'b1;
            if (old_m != 41'sd0)
                o_bypass_state = i_old;
            else if (i_m != 41'sd0)
                o_bypass_state = {i_h, i_m};
        end else begin
            o_a = $signed({high_m, 2'b00});
            o_b = low_aligned;
            o_h = high_h;
            o_tail = far ? (|low_m) : sticky[6];
        end
    end
endmodule
