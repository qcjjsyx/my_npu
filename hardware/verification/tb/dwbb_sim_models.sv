// Test-only functional models for the DesignWare arithmetic blocks used by
// mac_psum_stage. Do not add this file to a synthesis source list.
// DW02_tree preserves the sum modulo input_width and exposes it as two
// carry-save rows; this model uses serial 3:2 compression for functional
// simulation, not as a timing or implementation model of the vendor IP.
module DW02_tree #(
  parameter integer num_inputs = 8,
  parameter integer input_width = 8,
  parameter integer verif_en = 1
) (
  input  [num_inputs*input_width-1:0] INPUT,
  output reg [input_width-1:0] OUT0,
  output reg [input_width-1:0] OUT1
);
  reg [input_width-1:0] sum_work;
  reg [input_width-1:0] carry_work;
  reg [input_width-1:0] next_sum;
  reg [input_width-1:0] next_carry;
  integer i;

  always @* begin
    sum_work = INPUT[0 +: input_width];
    carry_work = {input_width{1'b0}};
    next_sum = {input_width{1'b0}};
    next_carry = {input_width{1'b0}};
    for (i = 1; i < num_inputs; i = i + 1) begin
      next_sum = sum_work ^ carry_work ^ INPUT[i*input_width +: input_width];
      next_carry = ((sum_work & carry_work) |
                    (sum_work & INPUT[i*input_width +: input_width]) |
                    (carry_work & INPUT[i*input_width +: input_width])) << 1;
      sum_work = next_sum;
      carry_work = next_carry;
    end
    OUT0 = sum_work;
    OUT1 = carry_work;
  end

  // Kept in the signature to match the DesignWare module parameter list.
  wire unused_verif_en = verif_en[0];
endmodule

module DW01_add #(
  parameter integer width = 8
) (
  input  [width-1:0] A,
  input  [width-1:0] B,
  input              CI,
  output [width-1:0] SUM,
  output             CO
);
  wire [width:0] extended_sum = {1'b0, A} + {1'b0, B} + CI;
  assign SUM = extended_sum[width-1:0];
  assign CO = extended_sum[width];
endmodule
