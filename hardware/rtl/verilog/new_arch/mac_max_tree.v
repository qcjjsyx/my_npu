module new_arch_mac_max_tree #(
    parameter Q_WIDTH = 7
) (
    input  wire [16*Q_WIDTH-1:0] i_q,
    input  wire [15:0]           i_nz,
    output wire [Q_WIDTH-1:0]    o_max,
    output wire                  o_nz
);
    // Heap order: 15 internal nodes, then 16 leaves. Compare unsigned Q only
    // after giving priority to a valid (nonzero product) child.
    wire [Q_WIDTH:0] tree [0:30];
    genvar j;
    generate
        for (j = 0; j < 16; j = j+1) begin : leaf
            assign tree[j+15] = i_nz[j]
                ? {1'b1, i_q[j*Q_WIDTH +: Q_WIDTH]} : {(Q_WIDTH+1){1'b0}};
        end
        for (j = 0; j < 15; j = j+1) begin : branch
            wire [Q_WIDTH:0] left = tree[2*j+1];
            wire [Q_WIDTH:0] right = tree[2*j+2];
            wire take_left = left[Q_WIDTH] &&
                (!right[Q_WIDTH] || left[Q_WIDTH-1:0] >= right[Q_WIDTH-1:0]);
            assign tree[j] = take_left ? left : right;
        end
    endgenerate
    assign {o_nz, o_max} = tree[0];
endmodule
