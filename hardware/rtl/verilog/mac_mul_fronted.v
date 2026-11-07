module mac_mul_fronted (
    input [25:0] i_a,
    input [25:0] i_b,
    input [2:0] i_mode,
    input [1:0] i_group_valid,
    input i_INT16_phase,
    output [23:0] o_product0,
    output [17:0] o_product1
    output [8:0] o_exp_sum0,
    output [5:0] o_exp_sum1
    output [1:0] o_product_nz
);

endmodule
  