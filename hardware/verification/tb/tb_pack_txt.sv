module tb_pack_txt;
  import mac_types_pkg::*;
  logic[15:0] raw;
  logic[2:0] fmt;
  logic[1:0] groups;
  operand_t payload;
  reg[31:0] case_id;
  reg[25:0] expected;
  integer fd,scan,count;
  string path;
  mac_operand_pack dut(.i_raw(raw),.i_src_fmt(fmt),.i_group_valid(groups),.o_payload(payload));
  initial begin
    if(!$value$plusargs("VEC=%s",path))$fatal(1,"missing VEC");
    fd=$fopen(path,"r");if(!fd)$fatal(1,"cannot open VEC");
    count=0;
    scan=$fscanf(fd,"%h %h %h %h %h",case_id,fmt,groups,raw,expected);
    while(scan==5)begin
      #1;
      if(payload!==expected)$fatal(1,"pack mismatch case=%0h raw=%0h actual=%0h expected=%0h",case_id,raw,payload,expected);
      count=count+1;
      scan=$fscanf(fd,"%h %h %h %h %h",case_id,fmt,groups,raw,expected);
    end
    if((scan!=-1 && !(scan==0 && $feof(fd))) || count==0)$fatal(1,"empty/malformed pack vectors");
    $fclose(fd);$display("pack_txt PASS: %0d vectors",count);$finish;
  end
endmodule
