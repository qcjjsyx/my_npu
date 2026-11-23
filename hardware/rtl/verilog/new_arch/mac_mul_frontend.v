// Assemble a parameterized array of clusters and register the R_PROD slots.
module mac_mul_frontend #(
    parameter integer CLUSTER_COUNT = 8
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          i_en_mul,
    input  wire                          i_en_prod,
    input  wire [2:0]                    i_mode_mul,
    input  wire [2:0]                    i_mode_prod,
    input  wire [44*CLUSTER_COUNT-1:0]   i_a,
    input  wire [44*CLUSTER_COUNT-1:0]   i_b,
    output reg  [72*CLUSTER_COUNT-1:0]   o_prod,
    output reg  [14*CLUSTER_COUNT-1:0]   o_q0,
    output reg  [8*CLUSTER_COUNT-1:0]    o_q1
);
    localparam [2:0] FP16 = 3'd3, INT16 = 3'd6;

    wire [72*CLUSTER_COUNT-1:0] lane_prod;
    wire [36*CLUSTER_COUNT-1:0] pair_prod;
    wire [14*CLUSTER_COUNT-1:0] lane_q0;
    wire [8*CLUSTER_COUNT-1:0] lane_q1;
    wire [7*CLUSTER_COUNT-1:0] pair_q0;

    genvar j;
    generate for (j = 0; j < CLUSTER_COUNT; j = j + 1) begin : cluster
        mac_mul_cluster u_cluster (
            .clk(clk), .rst_n(rst_n), .i_en_mul(i_en_mul),
            .i_mode_mul(i_mode_mul), .i_mode_prod(i_mode_prod),
            .i_a(i_a[j*44 +: 44]), .i_b(i_b[j*44 +: 44]),
            .o_prod(lane_prod[j*72 +: 72]),
            .o_q0(lane_q0[j*14 +: 14]), .o_q1(lane_q1[j*8 +: 8]),
            .o_pair_prod(pair_prod[j*36 +: 36]), .o_pair_q0(pair_q0[j*7 +: 7])
        );
    end endgenerate

    // Wide elements contribute one product per cluster. Pack them into
    // the low CLUSTER_COUNT product slots before the R_PROD boundary.
    wire paired = (i_mode_prod == FP16 || i_mode_prod == INT16);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_prod <= {(72*CLUSTER_COUNT){1'b0}};
            o_q0 <= {(14*CLUSTER_COUNT){1'b0}};
            o_q1 <= {(8*CLUSTER_COUNT){1'b0}};
        end else if (i_en_prod) begin
            o_prod <= paired ? {{(36*CLUSTER_COUNT){1'b0}}, pair_prod} : lane_prod;
            o_q0 <= paired ? {{(7*CLUSTER_COUNT){1'b0}}, pair_q0} : lane_q0;
            o_q1 <= lane_q1;
        end
    end
endmodule
