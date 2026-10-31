// Four-layer two-input compare/select tree for the per-group maximum
// exponent (IP_Design.md 6.3). A zero product is not an exponent candidate;
// ties keep the lower index. An all-zero group reports has_nonzero=0 and a
// deterministic zero placeholder exponent.
module mac_align_max16 #(
  parameter integer EXP_W
) (
  input  logic [EXP_W-1:0] i_exp [0:15],
  input  logic [15:0] i_nz,
  output logic [EXP_W-1:0] o_max_exp,
  output logic o_has_nonzero
);
  // Level 0 holds the 16 masked leaves; level 4 is the root.
  logic [EXP_W-1:0] node_exp [0:4][0:15];
  logic node_v [0:4][0:15];

  for (genvar i = 0; i < 16; i++) begin : gen_leaf
    assign node_v[0][i] = i_nz[i];
    assign node_exp[0][i] = i_nz[i] ? i_exp[i] : '0;
  end

  for (genvar l = 0; l < 4; l++) begin : gen_level
    for (genvar i = 0; i < (8 >> l); i++) begin : gen_node
      assign node_v[l+1][i] = node_v[l][2*i] | node_v[l][2*i+1];
      assign node_exp[l+1][i] =
        !node_v[l][2*i]   ? node_exp[l][2*i+1] :
        !node_v[l][2*i+1] ? node_exp[l][2*i]   :
        (node_exp[l][2*i] >= node_exp[l][2*i+1]
           ? node_exp[l][2*i] : node_exp[l][2*i+1]);
    end
  end

  assign o_max_exp = node_exp[4][0];
  assign o_has_nonzero = node_v[4][0];
endmodule
