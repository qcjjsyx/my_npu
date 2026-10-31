// One of the 16 physical multiplication frontends (IP_Design.md 5.2-5.7):
// one signed12 x signed12 and one signed9 x signed9 multiplier. No local
// product addition, no lane swapping, no second-phase weight shift.
//
// Inactive multiplier inputs are tied to zero (operand isolation); the
// matching o_product*_active flags tell the parent stage which physical
// result registers may update. Integer modes drive zero exponent sums and
// nonzero flags so the exponent adders and comparators stay quiet.
module mac_mul_frontend (
  input  mac_types_pkg::operand_t i_operand_a,
  input  mac_types_pkg::operand_t i_operand_b,
  input  mac_types_pkg::compute_mode_t i_mode,
  input  logic [1:0] i_group_valid,
  input  logic i_int16_phase,
  output mac_types_pkg::product12_t o_product0,
  output mac_types_pkg::product9_t o_product1,
  output logic [8:0] o_exp_sum0,
  output logic [5:0] o_exp_sum1,
  output logic [1:0] o_product_nz,
  output logic o_product0_active,
  output logic o_product1_active
);
  import mac_types_pkg::*;

  logic signed [11:0] mul12_a, mul12_b;
  logic signed [8:0] mul9_a, mul9_b;
  product12_t mul12_p;
  product9_t mul9_p;

  // The assignment context is the full physical result width (section 5.4).
  assign mul12_p = mul12_a * mul12_b;
  assign mul9_p = mul9_a * mul9_b;
  assign o_product0 = mul12_p;
  assign o_product1 = mul9_p;

  // Mode-dependent operand routing (section 5.3). The INT16 low byte is
  // unsigned: it gains a positive sign bit before port expansion. The high
  // byte keeps its original sign. All payloads are already finite two's
  // complement, so no further negate or special-value work happens here.
  always_comb begin
    mul12_a = '0;
    mul12_b = '0;
    mul9_a = '0;
    mul9_b = '0;
    o_product0_active = 1'b0;
    o_product1_active = 1'b0;

    case (i_mode)
      MODE_FP19: begin
        if (i_group_valid[0]) begin
          mul12_a = $signed(i_operand_a[11:0]);
          mul12_b = $signed(i_operand_b[11:0]);
          o_product0_active = 1'b1;
        end
        // MUL9 receives zero operands in FP19.
      end
      MODE_FP9: begin
        if (i_group_valid[0]) begin
          mul12_a = $signed({{7{i_operand_a[4]}}, i_operand_a[4:0]});
          mul12_b = $signed({{7{i_operand_b[4]}}, i_operand_b[4:0]});
          o_product0_active = 1'b1;
        end
        if (i_group_valid[1]) begin
          mul9_a = $signed({{4{i_operand_a[9]}}, i_operand_a[9:5]});
          mul9_b = $signed({{4{i_operand_b[9]}}, i_operand_b[9:5]});
          o_product1_active = 1'b1;
        end
      end
      MODE_INT8: begin
        if (i_group_valid[0]) begin
          mul12_a = $signed({{4{i_operand_a[7]}}, i_operand_a[7:0]});
          mul12_b = $signed({{4{i_operand_b[7]}}, i_operand_b[7:0]});
          o_product0_active = 1'b1;
        end
        if (i_group_valid[1]) begin
          mul9_a = $signed({i_operand_a[15], i_operand_a[15:8]});
          mul9_b = $signed({i_operand_b[15], i_operand_b[15:8]});
          o_product1_active = 1'b1;
        end
      end
      MODE_INT4: begin
        if (i_group_valid[0]) begin
          mul12_a = $signed({{8{i_operand_a[3]}}, i_operand_a[3:0]});
          mul12_b = $signed({{8{i_operand_b[3]}}, i_operand_b[3:0]});
          o_product0_active = 1'b1;
        end
        if (i_group_valid[1]) begin
          mul9_a = $signed({{5{i_operand_a[7]}}, i_operand_a[7:4]});
          mul9_b = $signed({{5{i_operand_b[7]}}, i_operand_b[7:4]});
          o_product1_active = 1'b1;
        end
      end
      MODE_INT16: begin
        if (i_group_valid[0]) begin
          mul12_a = $signed({4'b0000, i_operand_a[7:0]});
          mul9_a = $signed({i_operand_a[15], i_operand_a[15:8]});
          if (!i_int16_phase) begin
            mul12_b = $signed({4'b0000, i_operand_b[7:0]});
            mul9_b = $signed({1'b0, i_operand_b[7:0]});
          end else begin
            mul12_b = $signed({{4{i_operand_b[15]}}, i_operand_b[15:8]});
            mul9_b = $signed({i_operand_b[15], i_operand_b[15:8]});
          end
          o_product0_active = 1'b1;
          o_product1_active = 1'b1; // WIDE high limb belongs to group0.
        end
      end
      default: begin end // Invalid mode has no valid-transaction semantics.
    endcase
  end

  // Exponent sums and product-nonzero flags (section 5.7). Both sides are
  // zero-extended before the additions so no top carry is lost. Integer
  // modes keep these outputs at zero; downstream MAX logic is mode-gated.
  always_comb begin
    o_exp_sum0 = '0;
    o_exp_sum1 = '0;
    o_product_nz = '0;
    case (i_mode)
      MODE_FP19: begin
        if (i_group_valid[0]) begin
          o_exp_sum0 = {1'b0, i_operand_a[23:16]} +
                       {1'b0, i_operand_b[23:16]};
          o_product_nz[0] = (i_operand_a[11:0] != 12'd0) &&
                            (i_operand_b[11:0] != 12'd0);
        end
      end
      MODE_FP9: begin
        if (i_group_valid[0]) begin
          o_exp_sum0 = {3'b000, ({1'b0, i_operand_a[20:16]} +
                                 {1'b0, i_operand_b[20:16]})};
          o_product_nz[0] = (i_operand_a[4:0] != 5'd0) &&
                            (i_operand_b[4:0] != 5'd0);
        end
        if (i_group_valid[1]) begin
          o_exp_sum1 = {1'b0, i_operand_a[25:21]} +
                       {1'b0, i_operand_b[25:21]};
          o_product_nz[1] = (i_operand_a[9:5] != 5'd0) &&
                            (i_operand_b[9:5] != 5'd0);
        end
      end
      default: begin end // Integer modes: exponents and nz stay zero.
    endcase
  end
endmodule
