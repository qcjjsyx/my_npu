// Assemble a parameterized array of clusters and register the R_PROD slots.
module new_arch_mac_mul_frontend #(
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
    output reg  [64*CLUSTER_COUNT-1:0]   o_p0,
    output reg  [18*CLUSTER_COUNT-1:0]   o_p1,
    output reg  [32*CLUSTER_COUNT-1:0]   o_p2,
    output reg  [18*CLUSTER_COUNT-1:0]   o_p3,
    output reg  [14*CLUSTER_COUNT-1:0]   o_q0,
    output reg  [14*CLUSTER_COUNT-1:0]   o_q1
);
    localparam [2:0] FP16 = 3'd3, INT16 = 3'd6;

    wire [64*CLUSTER_COUNT-1:0] lane_p0;
    wire [18*CLUSTER_COUNT-1:0] lane_p1, lane_p3;
    wire [32*CLUSTER_COUNT-1:0] lane_p2, pair_p0;
    wire [14*CLUSTER_COUNT-1:0] lane_q0, lane_q1;
    wire [7*CLUSTER_COUNT-1:0] pair_q0;

    genvar j;
    generate for (j = 0; j < CLUSTER_COUNT; j = j + 1) begin : cluster
        new_arch_mac_mul_cluster u_cluster (
            .clk(clk), .rst_n(rst_n), .i_en_mul(i_en_mul),
            .i_mode_mul(i_mode_mul), .i_mode_prod(i_mode_prod),
            .i_a(i_a[j*44 +: 44]), .i_b(i_b[j*44 +: 44]),
            .o_p0(lane_p0[j*64 +: 64]), .o_p1(lane_p1[j*18 +: 18]),
            .o_p2(lane_p2[j*32 +: 32]), .o_p3(lane_p3[j*18 +: 18]),
            .o_q0(lane_q0[j*14 +: 14]), .o_q1(lane_q1[j*14 +: 14]),
            .o_pair_p0(pair_p0[j*32 +: 32]), .o_pair_q0(pair_q0[j*7 +: 7])
        );
    end endgenerate

    // Wide elements contribute one product per cluster. Pack them into
    // the low CLUSTER_COUNT product slots before the R_PROD boundary.
    wire paired = (i_mode_prod == FP16 || i_mode_prod == INT16);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_p0 <= {(64*CLUSTER_COUNT){1'b0}};
            o_p1 <= {(18*CLUSTER_COUNT){1'b0}};
            o_p2 <= {(32*CLUSTER_COUNT){1'b0}};
            o_p3 <= {(18*CLUSTER_COUNT){1'b0}};
            o_q0 <= {(14*CLUSTER_COUNT){1'b0}};
            o_q1 <= {(14*CLUSTER_COUNT){1'b0}};
        end else if (i_en_prod) begin
            o_p0 <= paired ? {{(32*CLUSTER_COUNT){1'b0}}, pair_p0} : lane_p0;
            o_q0 <= paired ? {{(7*CLUSTER_COUNT){1'b0}}, pair_q0} : lane_q0;
            o_p1 <= lane_p1;
            o_p2 <= lane_p2;
            o_p3 <= lane_p3;
            o_q1 <= lane_q1;
        end
    end
endmodule
