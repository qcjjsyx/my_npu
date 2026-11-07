module tb_mac_q234_pipeline;
  import mac_types_pkg::*;
  logic i_clk = 1'b0;
  always #5 i_clk = ~i_clk;
  logic i_rstn, i_in_valid, o_in_ready, o_m_valid, o_m_ready;
  logic o_align_valid, i_align_ready;
  logic [15:0] i_raw_a [0:15], i_raw_b [0:15];
  src_fmt_t i_src_fmt_a, i_src_fmt_b;
  logic [1:0] i_group_valid, i_acc_first;
  logic [3:0] i_tag [0:1];
  logic i_mx_enable;
  logic [7:0] i_scale_a [0:1], i_scale_b [0:1];
  product12_t o_m_product0 [0:15];
  product9_t o_m_product1 [0:15];
  logic [8:0] o_m_exp_sum0 [0:15];
  logic [5:0] o_m_exp_sum1 [0:15];
  logic [1:0] o_m_product_nz [0:15];
  txn_meta_t o_m_meta, o_align_meta;
  align_meta_t o_align_ameta;
  logic [3:0] o_m_tag [0:1], o_align_tag [0:1];
  align_lane0_t o_align_lane0 [0:15];
  align_lane1_t o_align_lane1 [0:15];

  mac_entry_m_stage #(.TAG_W(4)) u_m (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_in_valid(i_in_valid),
    .o_in_ready(o_in_ready), .i_raw_a(i_raw_a), .i_raw_b(i_raw_b),
    .i_src_fmt_a(i_src_fmt_a), .i_src_fmt_b(i_src_fmt_b),
    .i_group_valid(i_group_valid), .i_acc_first(i_acc_first),
    .i_tag(i_tag), .i_mx_enable(i_mx_enable),
    .i_scale_a(i_scale_a), .i_scale_b(i_scale_b),
    .o_m_valid(o_m_valid), .i_m_ready(o_m_ready),
    .o_m_product0(o_m_product0), .o_m_product1(o_m_product1),
    .o_m_exp_sum0(o_m_exp_sum0), .o_m_exp_sum1(o_m_exp_sum1), .o_m_product_nz(o_m_product_nz),
    .o_m_meta(o_m_meta), .o_m_tag(o_m_tag)
  );

  mac_align_stage #(.TAG_W(4)) u_align (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_m_valid(o_m_valid),
    .o_m_ready(o_m_ready), .i_m_product0(o_m_product0), .i_m_product1(o_m_product1),
    .i_m_exp_sum0(o_m_exp_sum0), .i_m_exp_sum1(o_m_exp_sum1), .i_m_product_nz(o_m_product_nz),
    .i_m_meta(o_m_meta), .i_m_tag(o_m_tag),
    .o_align_valid(o_align_valid), .i_align_ready(i_align_ready),
    .o_align_lane0(o_align_lane0), .o_align_lane1(o_align_lane1),
    .o_align_meta(o_align_meta), .o_align_ameta(o_align_ameta),
    .o_align_tag(o_align_tag)
  );

  logic psum_valid, psum_ready, mx_valid, mx_ready, acc_ready, admit;
  psum_packed_t psum_data;
  txn_meta_t psum_meta;
  align_meta_t psum_ameta;
  logic [3:0] psum_tag [0:1], mx_tag [0:1];
  contribution0_t c0;
  contribution1_t c1;
  acc_exp_t e0, e1;
  mx_meta_t mx_meta;
  logic acc_update;
  logic [1:0] acc_groups, acc_valid;
  acc_kind_t acc_kind, state_kind [0:1];
  logic [3:0] acc_tag [0:1];
  logic signed [31:0] int_state [0:1];
  acc_sig_t float_sig [0:1];
  acc_exp_t float_exp [0:1];
  logic signed [1:0] inf_sign [0:1];
  // Test-only admission gate injects stalls at the real MX->ACC boundary.
  // Both sides observe the same transfer; the production sink remains ready.
  assign mx_ready = admit && acc_ready;
  mac_psum_stage #(.TAG_W(4)) u_psum (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_align_valid(o_align_valid),
    .o_align_ready(i_align_ready), .i_align_lane0(o_align_lane0),
    .i_align_lane1(o_align_lane1), .i_align_meta(o_align_meta),
    .i_align_ameta(o_align_ameta), .i_align_tag(o_align_tag),
    .o_psum_valid(psum_valid), .i_psum_ready(psum_ready), .o_psum_data(psum_data),
    .o_psum_meta(psum_meta), .o_psum_ameta(psum_ameta), .o_psum_tag(psum_tag)
  );
  mac_mx_stage #(.TAG_W(4)) u_mx (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_psum_valid(psum_valid),
    .o_psum_ready(psum_ready), .i_psum_data(psum_data),
    .i_psum_meta(psum_meta), .i_psum_ameta(psum_ameta), .i_psum_tag(psum_tag),
    .o_mx_valid(mx_valid), .i_mx_ready(mx_ready),
    .o_mx_contribution0(c0), .o_mx_contribution1(c1),
    .o_mx_exp0(e0), .o_mx_exp1(e1), .o_mx_meta(mx_meta), .o_mx_tag(mx_tag)
  );
  mac_acc_stage #(.TAG_W(4)) u_acc (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_mx_valid(mx_valid && admit),
    .o_mx_ready(acc_ready), .i_mx_contribution0(c0), .i_mx_contribution1(c1),
    .i_mx_exp0(e0), .i_mx_exp1(e1), .i_mx_meta(mx_meta), .i_mx_tag(mx_tag),
    .o_acc_update_valid(acc_update), .o_acc_update_group_valid(acc_groups),
    .o_acc_update_kind(acc_kind), .o_acc_update_tag(acc_tag),
    .o_acc_state_valid(acc_valid), .o_acc_state_kind(state_kind),
    .o_acc_int32(int_state), .o_acc_float_sig(float_sig),
    .o_acc_float_exp(float_exp), .o_acc_float_inf_sign(inf_sign)
  );

  localparam N=128;
  logic [1:0] gv_mem [0:N-1], first_mem [0:N-1];
  logic [3:0] tag_mem [0:N-1][0:1];
  logic [7:0] sa_mem [0:N-1][0:1], sb_mem [0:N-1][0:1];
  logic kind_mem [0:N-1];
  logic [35:0] data_mem [0:N-1][0:1];
  logic [7:0] exp_mem [0:N-1][0:1];
  logic [1:0] inf_mem [0:N-1][0:1];
  logic [15:0] a_mem [0:N-1][0:15], b_mem [0:N-1][0:15];
  integer fd, scan, count, fmt_a, fmt_b, mx_flag;
  integer src, dst, cycles, stall_count, accept_count, mx_count;
  logic expected_commit;
  string vec;
  reg [63:0] v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14;
  reg [15:0] raw_a_scan, raw_b_scan;

  task automatic load_input(input integer index);
    i_group_valid = gv_mem[index];
    i_acc_first = first_mem[index];
    for (integer g=0; g<2; g=g+1) begin
      i_tag[g] = tag_mem[index][g];
      i_scale_a[g] = sa_mem[index][g];
      i_scale_b[g] = sb_mem[index][g];
    end
    for (integer lane=0; lane<16; lane=lane+1) begin
      i_raw_a[lane] = a_mem[index][lane];
      i_raw_b[lane] = b_mem[index][lane];
    end
  endtask

  initial begin
    if (!$value$plusargs("VEC=%s",vec)) $fatal(1,"missing VEC");
    fd=$fopen(vec,"r");
    if (!fd) $fatal(1,"cannot open VEC");
    scan=$fscanf(fd,"%d %d %d %d\n",fmt_a,fmt_b,mx_flag,count);
    if (scan!=4 || count<1 || count>N) $fatal(1,"bad config");
    case (fmt_a)
      0:i_src_fmt_a=SRC_FP16; 1:i_src_fmt_a=SRC_BF16;
      2:i_src_fmt_a=SRC_E4M3; 3:i_src_fmt_a=SRC_E5M2;
      4:i_src_fmt_a=SRC_INT16; 5:i_src_fmt_a=SRC_INT8; 6:i_src_fmt_a=SRC_INT4;
      default:$fatal(1,"bad format A");
    endcase
    case (fmt_b)
      0:i_src_fmt_b=SRC_FP16; 1:i_src_fmt_b=SRC_BF16;
      2:i_src_fmt_b=SRC_E4M3; 3:i_src_fmt_b=SRC_E5M2;
      4:i_src_fmt_b=SRC_INT16; 5:i_src_fmt_b=SRC_INT8; 6:i_src_fmt_b=SRC_INT4;
      default:$fatal(1,"bad format B");
    endcase
    i_mx_enable=mx_flag;
    for (integer n=0;n<count;n=n+1) begin
      scan=$fscanf(fd,"%h %h %h %h %h %h %h %h %h %h %h %h %h %h %h\n",
        v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14);
      gv_mem[n]=v0;
      first_mem[n]=v1;
      tag_mem[n][0]=v2;
      tag_mem[n][1]=v3;
      sa_mem[n][0]=v4;
      sa_mem[n][1]=v5;
      sb_mem[n][0]=v6;
      sb_mem[n][1]=v7;
      kind_mem[n]=v8;
      data_mem[n][0]=v9;
      exp_mem[n][0]=v10;
      inf_mem[n][0]=v11;
      data_mem[n][1]=v12;
      exp_mem[n][1]=v13;
      inf_mem[n][1]=v14;
      if(scan!=15) $fatal(1,"bad snapshot");
      for(integer lane=0;lane<16;lane=lane+1) begin
        scan=$fscanf(fd,"%h %h\n",raw_a_scan,raw_b_scan);
        a_mem[n][lane]=raw_a_scan; b_mem[n][lane]=raw_b_scan;
        if(scan!=2) $fatal(1,"bad raw vector");
      end
    end
    $fclose(fd);
    i_rstn=0; i_in_valid=0; admit=0;
    load_input(0);
    repeat(2) @(negedge i_clk);
    i_rstn=1;
    src=0;dst=0;cycles=0;stall_count=0;accept_count=0;mx_count=0;
    while(dst<count) begin
      @(negedge i_clk);
      admit=(cycles%19)<11;
      i_in_valid=(src<count);
      if(src<count) load_input(src);
      @(posedge i_clk);
      if(i_in_valid && o_in_ready) begin src=src+1; accept_count=accept_count+1; end
      if(i_in_valid && !o_in_ready) stall_count=stall_count+1;
      expected_commit=mx_valid && mx_ready && !(mx_meta.mode==MODE_INT16 && !mx_meta.int16_phase);
      if(mx_valid && mx_ready) mx_count=mx_count+1;
      #1;
      if(acc_update !== expected_commit) $fatal(1,"pipeline commit count/phase");
      if(acc_update) begin
        if(acc_groups!==gv_mem[dst] || acc_kind!==kind_mem[dst]) $fatal(1,"pipeline group/kind");
        for(integer g=0;g<2;g=g+1) if(gv_mem[dst][g]) begin
          if(acc_tag[g]!==tag_mem[dst][g] || !acc_valid[g] || state_kind[g]!==kind_mem[dst])
            $fatal(1,"pipeline tag/state valid %0d group%0d",dst,g);
          if(kind_mem[dst]) begin
            if(float_sig[g]!==data_mem[dst][g] || float_exp[g]!==exp_mem[dst][g] || inf_sign[g]!==inf_mem[dst][g])
              $fatal(1,"pipeline float item%0d group%0d got=%h@%h expected=%h@%h",dst,g,float_sig[g],float_exp[g],data_mem[dst][g],exp_mem[dst][g]);
          end else if(int_state[g]!==data_mem[dst][g][31:0]) $fatal(1,"pipeline integer %0d",dst);
        end
        dst=dst+1;
      end
      cycles=cycles+1;
      if(cycles>3000) $fatal(1,"pipeline timeout");
    end
    if(accept_count!=count || mx_count!=(fmt_a==4 ? count*2 : count) || stall_count==0)
      $fatal(1,"pipeline acceptance/stall coverage");
    @(negedge i_clk); i_in_valid=0; admit=1;
    repeat(8) begin @(posedge i_clk); #1; if(acc_update) $fatal(1,"duplicate update after drain"); end
    // Fill the pipeline with an unconsumed transaction, then reset. No stale
    // MX or INT16 phase may commit after all stage valids have been cleared.
    @(negedge i_clk); admit=0; load_input(0); i_in_valid=1;
    @(posedge i_clk); if(!o_in_ready) $fatal(1,"reset test source not accepted");
    @(negedge i_clk); i_in_valid=0;
    repeat(8) @(negedge i_clk);
    if(!mx_valid) $fatal(1,"reset test did not fill MX");
    i_rstn=0;
    @(negedge i_clk);
    if(mx_valid || psum_valid || o_align_valid || o_m_valid || acc_valid || acc_update || u_acc.int16_pending_q)
      $fatal(1,"reset did not clear pipeline/state");
    i_rstn=1; admit=1;
    repeat(8) begin @(posedge i_clk); #1; if(acc_update) $fatal(1,"reset replayed pending work"); end
    $display("mac_q234_pipeline PASS: %0d transactions",count);
    $finish;
  end
endmodule
