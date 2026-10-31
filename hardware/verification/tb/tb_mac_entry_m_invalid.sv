module tb_mac_entry_m_invalid;
  import mac_types_pkg::*;
  logic i_clk = 1'b0;
  always #5 i_clk = ~i_clk;
  logic i_rstn, i_in_valid, o_in_ready, o_m_valid;
  logic [15:0] i_raw_a [0:15], i_raw_b [0:15];
  src_fmt_t i_src_fmt_a, i_src_fmt_b;
  logic [1:0] i_group_valid, i_acc_first;
  logic [3:0] i_tag [0:1];
  logic i_mx_enable;
  logic [7:0] i_scale_a [0:1], i_scale_b [0:1];
  integer bad_case;

  mac_entry_m_stage #(.TAG_W(4)) dut (
    .i_clk(i_clk), .i_rstn(i_rstn),
    .i_in_valid(i_in_valid), .o_in_ready(o_in_ready),
    .i_raw_a(i_raw_a), .i_raw_b(i_raw_b),
    .i_src_fmt_a(i_src_fmt_a), .i_src_fmt_b(i_src_fmt_b),
    .i_group_valid(i_group_valid), .i_acc_first(i_acc_first),
    .i_tag(i_tag), .i_mx_enable(i_mx_enable),
    .i_scale_a(i_scale_a), .i_scale_b(i_scale_b),
    .o_m_valid(o_m_valid), .i_m_ready(1'b1),
    .o_m_product0(), .o_m_product1(),
    .o_m_exp_sum0(), .o_m_exp_sum1(), .o_m_product_nz(),
    .o_m_meta(), .o_m_tag()
  );

  initial begin
    if (!$value$plusargs("CASE=%d", bad_case)) $fatal(1, "missing CASE");
    i_rstn = 1'b0;
    i_in_valid = 1'b0;
    i_src_fmt_a = SRC_FP16;
    i_src_fmt_b = SRC_FP16;
    i_group_valid = 2'b01;
    i_acc_first = 2'b01;
    i_mx_enable = 1'b0;
    for (int i = 0; i < DOT_N; i++) begin
      i_raw_a[i] = '0;
      i_raw_b[i] = '0;
    end
    for (int g = 0; g < 2; g++) begin
      i_tag[g] = '0;
      i_scale_a[g] = 8'd127;
      i_scale_b[g] = 8'd127;
    end
    @(posedge i_clk);
    #1;
    i_rstn = 1'b1;
    case (bad_case)
      1: i_src_fmt_b = SRC_BF16;
      2: begin i_src_fmt_a = SRC_INT16; i_src_fmt_b = SRC_INT16; i_raw_a[0] = 16'h8000; end
      3: begin i_src_fmt_a = SRC_E4M3; i_src_fmt_b = SRC_E5M2;
               i_mx_enable = 1'b1; i_scale_a[0] = 8'hff; end
      4: i_group_valid = 2'b00;
      5: begin i_src_fmt_a = SRC_INT4; i_src_fmt_b = SRC_INT4; i_raw_a[0] = 16'h0100; end
      6: begin i_src_fmt_a = SRC_INT8; i_src_fmt_b = SRC_INT8; i_raw_b[0] = 16'h0080; end
      7: begin i_src_fmt_a = SRC_INT4; i_src_fmt_b = SRC_INT4; i_raw_b[0] = 16'h0008; end
      8: i_mx_enable = 1'b1;
      default: $fatal(1, "unknown CASE");
    endcase
    i_in_valid = 1'b1;
    #1;
    if (!o_in_ready) $fatal(1, "invalid test did not reach input handshake");
    @(posedge i_clk);
    #1;
    $fatal(1, "invalid transaction was silently accepted");
  end
endmodule
