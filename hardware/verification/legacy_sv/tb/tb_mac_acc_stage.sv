module tb_mac_acc_stage;
  import mac_types_pkg::*;
  localparam integer TAG_W = 8;

  logic i_clk = 1'b0;
  logic i_rstn;
  logic i_mx_valid;
  logic o_mx_ready;
  contribution0_t i_mx_contribution0;
  contribution1_t i_mx_contribution1;
  acc_exp_t i_mx_exp0, i_mx_exp1;
  mx_meta_t i_mx_meta;
  logic [TAG_W-1:0] i_mx_tag [0:1];
  logic o_acc_update_valid;
  logic [1:0] o_acc_update_group_valid;
  acc_kind_t o_acc_update_kind;
  logic [TAG_W-1:0] o_acc_update_tag [0:1];
  logic [1:0] o_acc_state_valid;
  acc_kind_t o_acc_state_kind [0:1];
  logic signed [31:0] o_acc_int32 [0:1];
  acc_sig_t o_acc_float_sig [0:1];
  acc_exp_t o_acc_float_exp [0:1];

  logic signed [1:0] o_acc_float_inf_sign [0:1];
  reg [1:0] exp_inf0, exp_inf1;

  integer vec_file, scan_count, vec_count, bad_case;
  string vec_path;
  reg [2:0] vec_mode;
  reg vec_kind;
  reg [1:0] vec_group_valid, vec_first;
  reg vec_phase;
  reg [35:0] vec_contribution0, exp_float_sig0, exp_float_sig1;
  reg [35:0] vec_contribution1;
  reg [7:0] vec_exp0, vec_exp1, exp_float_exp0, exp_float_exp1;
  reg [7:0] vec_tag0, vec_tag1;
  reg exp_update_valid;
  reg [1:0] exp_update_group_valid, exp_state_valid;
  reg exp_update_kind, exp_state_kind0, exp_state_kind1;
  reg [31:0] exp_int0, exp_int1;

  always #5 i_clk = ~i_clk;

  mac_acc_stage #(.TAG_W(TAG_W)) dut (
    .i_clk(i_clk), .i_rstn(i_rstn), .i_mx_valid(i_mx_valid),
    .o_mx_ready(o_mx_ready),
    .i_mx_contribution0(i_mx_contribution0),
    .i_mx_contribution1(i_mx_contribution1),
    .i_mx_exp0(i_mx_exp0), .i_mx_exp1(i_mx_exp1),
    .i_mx_meta(i_mx_meta), .i_mx_tag(i_mx_tag),
    .o_acc_update_valid(o_acc_update_valid),
    .o_acc_update_group_valid(o_acc_update_group_valid),
    .o_acc_update_kind(o_acc_update_kind), .o_acc_update_tag(o_acc_update_tag),
    .o_acc_state_valid(o_acc_state_valid), .o_acc_state_kind(o_acc_state_kind),
    .o_acc_int32(o_acc_int32), .o_acc_float_sig(o_acc_float_sig),
    .o_acc_float_exp(o_acc_float_exp),
    .o_acc_float_inf_sign(o_acc_float_inf_sign)
  );

  task automatic set_mode(input reg [2:0] mode_code);
    begin
      case (mode_code)
        3'd0: i_mx_meta.mode = MODE_FP19;
        3'd1: i_mx_meta.mode = MODE_FP9;
        3'd2: i_mx_meta.mode = MODE_INT16;
        3'd3: i_mx_meta.mode = MODE_INT8;
        3'd4: i_mx_meta.mode = MODE_INT4;
        default: $fatal(1, "bad vector mode");
      endcase
    end
  endtask

  task automatic compare_after_accept;
    reg [1:0] got_kinds;
    begin
      #1;
      got_kinds = {o_acc_state_kind[1], o_acc_state_kind[0]};
      if (o_acc_update_valid !== exp_update_valid ||
          o_acc_update_group_valid !== exp_update_group_valid ||
          o_acc_state_valid !== exp_state_valid ||
          o_acc_int32[0] !== exp_int0 || o_acc_int32[1] !== exp_int1 ||
          o_acc_float_inf_sign[0] !== exp_inf0 ||
          o_acc_float_inf_sign[1] !== exp_inf1 ||
          o_acc_float_sig[0] !== exp_float_sig0 ||
          o_acc_float_sig[1] !== exp_float_sig1 ||
          o_acc_float_exp[0] !== exp_float_exp0 ||
          o_acc_float_exp[1] !== exp_float_exp1 ||
          o_acc_state_kind[0] !== exp_state_kind0 ||
          o_acc_state_kind[1] !== exp_state_kind1)
        $fatal(1, "ACC vector %0d mismatch: state_valid=%b kinds=%b int=%h/%h float=%h@%h %h@%h update=%b/%b",
               vec_count, o_acc_state_valid, got_kinds, o_acc_int32[0], o_acc_int32[1],
               o_acc_float_sig[0], o_acc_float_exp[0],
               o_acc_float_sig[1], o_acc_float_exp[1],
               o_acc_update_valid, o_acc_update_group_valid);
      if (o_acc_update_valid &&
          (o_acc_update_kind !== exp_update_kind ||
           o_acc_update_tag[0] !== vec_tag0 || o_acc_update_tag[1] !== vec_tag1))
        $fatal(1, "ACC vector %0d update metadata mismatch", vec_count);
    end
  endtask

  initial begin
    i_rstn = 1'b0;
    i_mx_valid = 1'b0;
    i_mx_contribution0 = '0;
    i_mx_contribution1 = '0;
    i_mx_exp0 = '0;
    i_mx_exp1 = '0;
    i_mx_meta = '0;
    i_mx_tag[0] = '0;
    i_mx_tag[1] = '0;
    vec_count = 0;
    bad_case = 0;
    scan_count = $value$plusargs("BAD=%d",bad_case);
    repeat (2) @(negedge i_clk);
    i_rstn = 1'b1;

    if (!$value$plusargs("VEC=%s", vec_path)) $fatal(1, "missing +VEC=<path>");
    vec_file = $fopen(vec_path, "r");
    if (vec_file == 0) $fatal(1, "failed to open vector file");

    while (!$feof(vec_file)) begin
      scan_count = $fscanf(vec_file,
        "%h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h\n",
        vec_mode, vec_kind, vec_group_valid, vec_first, vec_phase,
        vec_contribution0, vec_contribution1, vec_exp0, vec_exp1, vec_tag0, vec_tag1,
        exp_update_valid, exp_update_group_valid, exp_update_kind,
        exp_state_valid, exp_state_kind0, exp_state_kind1,
        exp_int0, exp_int1, exp_float_sig0, exp_float_exp0,
        exp_float_sig1, exp_float_exp1, exp_inf0, exp_inf1);
      if (scan_count == 0) begin
        // EOF after the final newline.
      end else if (scan_count != 25) begin
        $fatal(1, "malformed ACC vector line: scanned %0d fields", scan_count);
      end else begin
        @(negedge i_clk);
        i_mx_valid = 1'b1;
        set_mode(vec_mode);
        if (vec_kind) i_mx_meta.acc_kind = ACC_TC_FLOAT;
        else i_mx_meta.acc_kind = ACC_INT32;
        i_mx_meta.group_valid = vec_group_valid;
        i_mx_meta.acc_first = vec_first;
        i_mx_meta.int16_phase = vec_phase;
        i_mx_contribution0 = $signed(vec_contribution0);
        i_mx_contribution1 = $signed(vec_contribution1);
        i_mx_exp0 = $signed(vec_exp0);
        i_mx_exp1 = $signed(vec_exp1);
        i_mx_tag[0] = vec_tag0;
        i_mx_tag[1] = vec_tag1;
        if (vec_count == 0) begin
          if (bad_case == 1) i_mx_meta.acc_first = 0;
          if (bad_case == 2) i_mx_exp0 = 0;
          if (bad_case == 3) i_mx_exp0 = 255;
          if (bad_case == 4) begin
            i_mx_meta.mode = MODE_INT16;
            i_mx_meta.acc_kind = ACC_INT32;
            i_mx_meta.acc_first = 0;
            i_mx_meta.int16_phase = 1;
          end
        end
        if (!o_mx_ready) $fatal(1, "ACC unexpectedly backpressured");
        @(posedge i_clk);
        compare_after_accept();
        vec_count = vec_count + 1;
        if ((vec_count % 10) == 0) begin
          // A real valid bubble must not replay the just-consumed MX update;
          // this also holds INT16's phase0 work state across an idle cycle.
          @(negedge i_clk);
          i_mx_valid = 1'b0;
          @(posedge i_clk);
          #1;
          if (o_acc_update_valid !== 1'b0 ||
              o_acc_update_group_valid !== 2'b00 ||
              o_acc_state_valid !== exp_state_valid ||
              o_acc_int32[0] !== exp_int0 || o_acc_int32[1] !== exp_int1 ||
              o_acc_float_inf_sign[0] !== exp_inf0 ||
          o_acc_float_inf_sign[1] !== exp_inf1 ||
          o_acc_float_sig[0] !== exp_float_sig0 ||
              o_acc_float_sig[1] !== exp_float_sig1 ||
              o_acc_float_exp[0] !== exp_float_exp0 ||
              o_acc_float_exp[1] !== exp_float_exp1 ||
              o_acc_state_kind[0] !== exp_state_kind0 ||
              o_acc_state_kind[1] !== exp_state_kind1)
            $fatal(1, "ACC valid bubble changed state or replayed update after vector %0d", vec_count-1);
        end
      end
    end
    @(negedge i_clk);
    i_mx_valid = 1'b0;
    $fclose(vec_file);
    i_rstn = 0;
    @(posedge i_clk); #1;
    if (o_acc_state_valid || o_acc_update_valid || dut.int16_pending_q ||
        o_acc_float_sig[0] || o_acc_float_sig[1] ||
        o_acc_float_exp[0] || o_acc_float_exp[1] ||
        o_acc_float_inf_sign[0] || o_acc_float_inf_sign[1])
      $fatal(1,"ACC reset state");
    $display("mac_acc_stage PASS: %0d vectors", vec_count);
    $finish;
  end
endmodule
