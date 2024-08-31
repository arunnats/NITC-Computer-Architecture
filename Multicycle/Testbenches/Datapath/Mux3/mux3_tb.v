`timescale 1ps/1ps  
module mux3_tb;  
parameter WIDTH = 32;  
reg [WIDTH-1:0] d0, d1, d2; 
  reg [1:0]s; 
  wire [WIDTH-1:0] y;  
 
initial 
begin  
    $dumpfile("mux3_tb.vcd");  
    $dumpvars(0, mux3_tb);

    d0 = 32'h1234abcd;  
    d1 = 32'habcd1234;  
    d2 = 32'h11112222;  
    s = 0;  
    #4 $finish;
end  
 
always #1 s = s + 1;  
 
mux3 #(WIDTH) uut(d0, d1, d2, s, y);  
endmodule 