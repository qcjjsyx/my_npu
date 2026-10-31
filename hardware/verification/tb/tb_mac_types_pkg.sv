module tb_mac_types_pkg;
  import mac_types_pkg::*;

  initial begin
    if (DOT_N != 16 || OPERAND_W != 26 || PSUM_W != 56) $fatal(1, "width constants");
    if ($bits(contribution0_t) != 36 || $bits(contribution1_t) != 36 ||
        $bits(acc_sig_t) != 36 || $bits(acc_exp_t) != 8 ||
        ACC_WORK_W != 39) $fatal(1, "Q2.34 interface widths");
    if ($bits(m_lane_t) != 59) $fatal(1, "M lane width");
    if ($bits(align_lane_t) != 56) $fatal(1, "ALIGN lane width");
    if ($bits(txn_meta_t) != 50) $fatal(1, "transaction metadata width");
    if ($bits(align_meta_t) != 19) $fatal(1, "ALIGN metadata width");
    if ($bits(mx_meta_t) != 13) $fatal(1, "MX metadata width");
    if (!legal_src_pair(SRC_E4M3, SRC_E5M2)) $fatal(1, "mixed FP8");
    if (legal_src_pair(SRC_FP16, SRC_BF16)) $fatal(1, "unsupported mixed FP19");
    if (bias_sum_for(SRC_BF16, SRC_BF16) != 8'd254) $fatal(1, "bias sum");
    if (mode_for(SRC_INT8) != MODE_INT8) $fatal(1, "mode map");
    if (acc_kind_for(MODE_INT8, 1'b1) != ACC_TC_FLOAT) $fatal(1, "MXINT8 kind");
    if (acc_kind_for(MODE_INT8, 1'b0) != ACC_INT32) $fatal(1, "INT8 kind");
    if (!is_wide_mode(MODE_INT16) || is_wide_mode(MODE_FP9)) $fatal(1, "pack mode");
    if (!is_float_mode(MODE_FP19) || !is_float_mode(MODE_FP9) ||
        is_float_mode(MODE_INT8) || is_float_mode(MODE_INT16) ||
        is_float_mode(MODE_INT4)) $fatal(1, "float mode map");
    if (lane1_active(MODE_FP19, 2'b01)) $fatal(1, "FP19 must isolate lane1");
    if (!lane1_active(MODE_INT16, 2'b01)) $fatal(1, "INT16 lane1 high limb");
    if (lane1_active(MODE_INT8, 2'b01) ||
        !lane1_active(MODE_INT8, 2'b10)) $fatal(1, "DUAL lane1 follows group1");
    if (!lane1_active(MODE_FP9, 2'b11) ||
        !lane1_active(MODE_INT4, 2'b11)) $fatal(1, "DUAL dual-valid lane1");
    $display("mac_types_pkg PASS");
  end
endmodule
