module mac_ctx_control (
    input  wire clk,
    input  wire rst_n,
    input  wire [2:0] i_mode,
    input  wire i_mx,
    input  wire [1:0] i_ctx,
    input  wire i_first,
    input  wire i_last,
    input  wire i_valid,
    input  wire i_pipe_empty,
    input  wire i_write_last,
    input  wire [1:0] i_write_ctx,
    input  wire [1:0] i_result_ctx,
    input  wire i_result_ready,
    output wire [2:0] o_ctx_free,
    output wire [2:0] o_ctx_done,
    output reg o_result_valid,
    output wire [2:0] o_result_mode,
    output wire o_result_mx
);
    localparam [1:0] FREE = 2'b00, ACTIVE = 2'b01, CLOSING = 2'b10, DONE = 2'b11;
    reg [1:0] r_lifecycle [0:2];
    reg [2:0] batch_mode;
    reg batch_mx;

    genvar j;
    generate for (j = 0; j < 3; j = j+1) begin : slot
        localparam [1:0] CTX = j;
        assign o_ctx_free[j] = r_lifecycle[j] == FREE;
        assign o_ctx_done[j] = r_lifecycle[j] == DONE;

        // Legal traffic has at most one lifecycle-changing event per slot.
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                r_lifecycle[j] <= FREE;
            else if (i_write_last && i_write_ctx == CTX)
                r_lifecycle[j] <= DONE;
            else if (o_ctx_done[j] && i_result_ready && i_result_ctx == CTX)
                r_lifecycle[j] <= FREE;
            else if (i_valid && i_ctx == CTX) begin
                if (r_lifecycle[j] == FREE && i_first)
                    r_lifecycle[j] <= i_last ? CLOSING : ACTIVE;
                else if (r_lifecycle[j] == ACTIVE && i_last)
                    r_lifecycle[j] <= CLOSING;
            end
        end
    end endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            batch_mode <= 3'd0;
            batch_mx <= 1'b0;
        end else if (i_valid && i_first && (&o_ctx_free) && i_pipe_empty) begin
            batch_mode <= i_mode;
            batch_mx <= i_mx;
        end
    end

    always @* begin
        case (i_result_ctx)
            2'd0: o_result_valid = o_ctx_done[0];
            2'd1: o_result_valid = o_ctx_done[1];
            2'd2: o_result_valid = o_ctx_done[2];
            default: o_result_valid = 1'b0;
        endcase
    end
    assign o_result_mode = o_result_valid ? batch_mode : 3'd0;
    assign o_result_mx = o_result_valid ? batch_mx : 1'b0;
endmodule
