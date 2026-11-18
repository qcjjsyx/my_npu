// Test-only packed-operand MAC boundary. No entry conversion/holding slot.
module tb_mac_packed_txt;
  import mac_types_pkg::*;
  logic i_clk=0;
  always #5 i_clk=~i_clk;
  logic i_rstn=0, i_in_valid=0, o_in_ready, o_m_valid, o_m_ready;
  operand_t operand_a[0:15], operand_b[0:15];
  txn_meta_t input_meta, o_m_meta, o_align_meta;
  logic [3:0] input_tag[0:1], o_m_tag[0:1], o_align_tag[0:1];
  product12_t o_m_product0[0:15], product0[0:15];
  product9_t o_m_product1[0:15], product1[0:15];
  logic [8:0] o_m_exp_sum0[0:15], exp_sum0[0:15];
  logic [5:0] o_m_exp_sum1[0:15], exp_sum1[0:15];
  logic [1:0] o_m_product_nz[0:15], product_nz[0:15];
  logic active0[0:15],active1[0:15];
  logic o_align_valid,i_align_ready;
  align_meta_t o_align_ameta;
  align_lane0_t o_align_lane0[0:15];
  align_lane1_t o_align_lane1[0:15];
  assign o_in_ready=i_rstn && (!o_m_valid || o_m_ready);
  for(genvar j=0;j<16;j=j+1) begin: gen_mul
    mac_mul_frontend u_mul(
      .i_operand_a(operand_a[j]),.i_operand_b(operand_b[j]),
      .i_mode(input_meta.mode),.i_group_valid(input_meta.group_valid),
      .i_int16_phase(input_meta.int16_phase),.o_product0(product0[j]),
      .o_product1(product1[j]),.o_exp_sum0(exp_sum0[j]),.o_exp_sum1(exp_sum1[j]),
      .o_product_nz(product_nz[j]),.o_product0_active(active0[j]),.o_product1_active(active1[j])
    );
  end
  // One elastic M output register. This is the only replacement for entry.
  always @(posedge i_clk) begin
    if(!i_rstn) begin
      o_m_valid<=0; o_m_meta<='0; o_m_tag[0]<=0;o_m_tag[1]<=0;
      for(integer j=0;j<16;j=j+1) begin
        o_m_product0[j]<=0;o_m_product1[j]<=0;
        o_m_exp_sum0[j]<=0;o_m_exp_sum1[j]<=0;o_m_product_nz[j]<=0;
      end
    end else if(o_in_ready) begin
      o_m_valid<=i_in_valid;
      if(i_in_valid) begin
        o_m_meta<=input_meta; o_m_tag[0]<=input_tag[0];o_m_tag[1]<=input_tag[1];
        for(integer j=0;j<16;j=j+1) begin
          o_m_product0[j]<=product0[j];o_m_product1[j]<=product1[j];
          o_m_exp_sum0[j]<=exp_sum0[j];o_m_exp_sum1[j]<=exp_sum1[j];
          o_m_product_nz[j]<=product_nz[j];
        end
      end
    end
  end
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


  localparam N=32768;
  reg [63:0] ih[0:N-1][0:14], eh[0:N-1][0:17], th[0:N-1][0:10];
  reg [25:0] a_mem[0:N-1][0:15],b_mem[0:N-1][0:15];
  reg [63:0] lanes[0:N-1][0:15][0:6];
  reg [63:0] token;
  integer fd,ef,tf,scan,count,src,mc,ac,pc,xc,cycles,stalls,accepted;
  integer schedule=1, negative=0;
  logic took_input,took_mx;
  string inputs,expected,intermediate;

  task read_hex(input integer file_id,output reg[63:0] value);
    integer got;
    begin
      got=$fscanf(file_id,"%h",value);
      if(got!=1) $fatal(1,"missing/malformed token record=%0d",count);
    end
  endtask
  task set_input(input integer n);
    begin
      case(ih[n][2])
        0:input_meta.mode=MODE_FP19;1:input_meta.mode=MODE_FP9;
        2:input_meta.mode=MODE_INT16;3:input_meta.mode=MODE_INT8;
        4:input_meta.mode=MODE_INT4;default:$fatal(1,"invalid mode in TXT");
      endcase
      if(ih[n][3])input_meta.acc_kind=ACC_TC_FLOAT;else input_meta.acc_kind=ACC_INT32;
      input_meta.group_valid=ih[n][4];input_meta.acc_first=ih[n][5];
      input_meta.int16_phase=ih[n][6];input_meta.bias_sum=ih[n][7];
      input_meta.mx_enable=ih[n][8];input_meta.scale_a[0]=ih[n][9];
      input_meta.scale_a[1]=ih[n][10];input_meta.scale_b[0]=ih[n][11];
      input_meta.scale_b[1]=ih[n][12];input_tag[0]=ih[n][13];input_tag[1]=ih[n][14];
      for(integer j=0;j<16;j=j+1)begin operand_a[j]=a_mem[n][j];operand_b[j]=b_mem[n][j];end
    end
  endtask
  task check_meta(input integer n,input txn_meta_t meta,input logic[3:0] t0,t1);
    begin
      if(meta.mode!==ih[n][2][2:0] || meta.acc_kind!==ih[n][3][0] ||
         meta.group_valid!==ih[n][4][1:0] || meta.acc_first!==ih[n][5][1:0] ||
         meta.int16_phase!==ih[n][6][0] || meta.bias_sum!==ih[n][7][7:0] ||
         meta.mx_enable!==ih[n][8][0] || meta.scale_a[0]!==ih[n][9][7:0] ||
         meta.scale_a[1]!==ih[n][10][7:0] || meta.scale_b[0]!==ih[n][11][7:0] ||
         meta.scale_b[1]!==ih[n][12][7:0] || t0!==ih[n][13][3:0] || t1!==ih[n][14][3:0])
        $fatal(1,"metadata mismatch seq=%0h step=%0h",ih[n][0],ih[n][1]);
    end
  endtask
  task check_ameta(input integer n,input align_meta_t meta);
    begin
      if(meta.emax0!==th[n][2][8:0] || meta.emax1!==th[n][3][5:0] ||
         meta.group_nz!==th[n][4][1:0] || meta.inexact!==th[n][5][1:0])
        $fatal(1,"ALIGN metadata mismatch record=%0d",n);
    end
  endtask
  task check_state(input integer n,input logic commit_cycle);
    begin
      if(acc_update!==(commit_cycle ? eh[n][2][0] : 1'b0) ||
         acc_groups!==(commit_cycle ? eh[n][3][1:0] : 2'b0) ||
         acc_kind!==eh[n][4][0] || acc_tag[0]!==eh[n][5][3:0] ||
         acc_tag[1]!==eh[n][6][3:0] || acc_valid!==eh[n][7][1:0])
        $fatal(1,"ACC update mismatch seq=%0h step=%0h",eh[n][0],eh[n][1]);
      for(integer g=0;g<2;g=g+1)begin
        if(state_kind[g]!==eh[n][8+5*g][0] || int_state[g]!==eh[n][9+5*g][31:0] ||
           float_sig[g]!==eh[n][10+5*g][35:0] || float_exp[g]!==eh[n][11+5*g][7:0] ||
           inf_sign[g]!==eh[n][12+5*g][1:0])
          $fatal(1,"ACC state mismatch seq=%0h step=%0h group=%0d",eh[n][0],eh[n][1],g);
      end
    end
  endtask
  initial begin
    input_meta='0;input_tag[0]=0;input_tag[1]=0;admit=0;
    scan=$value$plusargs("SCHEDULE=%d",schedule);
    scan=$value$plusargs("NEGATIVE=%d",negative);
    if(!$value$plusargs("INPUTS=%s",inputs))$fatal(1,"missing INPUTS");
    fd=$fopen(inputs,"r");if(!fd)$fatal(1,"cannot open INPUTS");
    if(!negative)begin
      if(!$value$plusargs("EXPECTED=%s",expected) || !$value$plusargs("INTERMEDIATE=%s",intermediate))$fatal(1,"missing expected files");
      ef=$fopen(expected,"r");tf=$fopen(intermediate,"r");
      if(!ef || !tf)$fatal(1,"cannot open expected files");
    end
    count=0;
    scan=$fscanf(fd,"%h",token);
    while(scan==1)begin
      if(count>=N)$fatal(1,"too many records (max32768)");
      ih[count][0]=token;
      for(integer k=1;k<15;k=k+1)begin read_hex(fd,token);ih[count][k]=token;end
      for(integer j=0;j<16;j=j+1)begin
        read_hex(fd,token);a_mem[count][j]=token;
        read_hex(fd,token);b_mem[count][j]=token;
      end
      if(!negative)begin
        for(integer k=0;k<18;k=k+1)begin read_hex(ef,token);eh[count][k]=token;end
        for(integer k=0;k<11;k=k+1)begin read_hex(tf,token);th[count][k]=token;end
        if(ih[count][0]!==eh[count][0] || ih[count][1]!==eh[count][1] ||
           ih[count][0]!==th[count][0] || ih[count][1]!==th[count][1])$fatal(1,"file ID mismatch");
        for(integer j=0;j<16;j=j+1)for(integer k=0;k<7;k=k+1)begin
          read_hex(tf,token);lanes[count][j][k]=token;
        end
      end
      count=count+1;scan=$fscanf(fd,"%h",token);
    end
    if((scan!=-1 && !(scan==0 && $feof(fd))) || count==0)$fatal(1,"malformed/empty inputs");
    $fclose(fd);
    if(!negative)begin
      scan=$fscanf(ef,"%h",token);if(scan!=-1 && !(scan==0 && $feof(ef)))$fatal(1,"extra EXPECTED data");
      scan=$fscanf(tf,"%h",token);if(scan!=-1 && !(scan==0 && $feof(tf)))$fatal(1,"extra INTERMEDIATE data");
      $fclose(ef);$fclose(tf);
    end
    repeat(2)@(negedge i_clk);
    i_rstn=1;src=0;mc=0;ac=0;pc=0;xc=0;cycles=0;stalls=0;accepted=0;
    while(xc<count)begin
      @(negedge i_clk);
      admit=(schedule==0) || ((cycles%19)<11);
      // Keep valid asserted if the previous offer has not been accepted.
      if(!i_in_valid) i_in_valid=(src<count) && ((schedule==0) || cycles%7!=0);
      if(src<count)set_input(src);
      @(posedge i_clk);
      took_input=i_in_valid && o_in_ready;
      took_mx=mx_valid && mx_ready;
      if(i_in_valid && !o_in_ready)stalls=stalls+1;
      if(o_m_valid && o_m_ready)begin
        if(mc>=count)$fatal(1,"duplicate M result");
        if(!negative)begin
          check_meta(mc,o_m_meta,o_m_tag[0],o_m_tag[1]);
          for(integer j=0;j<16;j=j+1)
            if(o_m_product0[j]!==lanes[mc][j][0][23:0] || o_m_product1[j]!==lanes[mc][j][1][17:0] ||
               o_m_exp_sum0[j]!==lanes[mc][j][2][8:0] || o_m_exp_sum1[j]!==lanes[mc][j][3][5:0] ||
               o_m_product_nz[j]!==lanes[mc][j][4][1:0])$fatal(1,"M mismatch record=%0d lane=%0d",mc,j);
        end
        mc=mc+1;
      end
      if(o_align_valid && i_align_ready)begin
        if(ac>=count)$fatal(1,"duplicate ALIGN result");
        if(!negative)begin
          check_meta(ac,o_align_meta,o_align_tag[0],o_align_tag[1]);check_ameta(ac,o_align_ameta);
          for(integer j=0;j<16;j=j+1)
            if(o_align_lane0[j]!==lanes[ac][j][5][31:0] || o_align_lane1[j]!==lanes[ac][j][6][23:0])$fatal(1,"ALIGN mismatch record=%0d lane=%0d",ac,j);
        end
        ac=ac+1;
      end
      if(psum_valid && psum_ready)begin
        if(pc>=count)$fatal(1,"duplicate PSUM result");
        if(!negative)begin
          check_meta(pc,psum_meta,psum_tag[0],psum_tag[1]);check_ameta(pc,psum_ameta);
          if(psum_data!==th[pc][6][55:0])$fatal(1,"PSUM mismatch record=%0d",pc);
        end
        pc=pc+1;
      end
      if(took_mx)begin
        if(!negative)begin
          if(c0!==th[xc][7][35:0] || c1!==th[xc][8][35:0] || e0!==th[xc][9][7:0] || e1!==th[xc][10][7:0])$fatal(1,"MX mismatch record=%0d",xc);
          if(mx_meta.mode!==ih[xc][2][2:0] || mx_meta.acc_kind!==ih[xc][3][0] ||
             mx_meta.group_valid!==ih[xc][4][1:0] || mx_meta.acc_first!==ih[xc][5][1:0] ||
             mx_meta.int16_phase!==ih[xc][6][0] || mx_meta.group_nz!==th[xc][4][1:0] ||
             mx_meta.inexact!==th[xc][5][1:0] || mx_tag[0]!==ih[xc][13][3:0] || mx_tag[1]!==ih[xc][14][3:0])$fatal(1,"MX metadata mismatch record=%0d",xc);
        end
        xc=xc+1;
      end
      #1;
      if(!negative && xc>0)check_state(xc-1,took_mx);
      if(took_input)begin src=src+1;accepted=accepted+1;i_in_valid=0;end
      cycles=cycles+1;
      if(cycles>count*25+100)$fatal(1,"timeout");
    end
    if(negative)$fatal(1,"negative vector was NOT rejected");
    if(accepted!=count || mc!=count || ac!=count || pc!=count)$fatal(1,"transaction count mismatch");
    if(schedule!=0 && count>20 && stalls==0)$fatal(1,"no upstream stall coverage");
    @(negedge i_clk);i_in_valid=0;admit=1;
    repeat(8)begin @(posedge i_clk);#1;check_state(count-1,0);end
    @(negedge i_clk);i_rstn=0;
    @(posedge i_clk);#1;
    if(acc_valid || acc_update || o_m_valid || o_align_valid || psum_valid || mx_valid)$fatal(1,"reset valid mismatch");
    for(integer g=0;g<2;g=g+1)
      if(int_state[g] || float_sig[g] || float_exp[g] || inf_sign[g])$fatal(1,"reset state mismatch");
    $display("mac_packed_txt PASS: %0d steps schedule=%0d stalls=%0d",count,schedule,stalls);
    $finish;
  end
endmodule
