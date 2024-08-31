`timescale 1ps/1ps  
module mux4_tb;  
parameter WIDTH = 32;  
reg [WIDTH-1:0] d0, d1, d2, d3; 
  reg [2:0] s; 
  wire [WIDTH-1:0] y;  
 
initial 
begin  
    $dumpfile("mux4_tb.vcd");  
    $dumpvars(0, mux4_tb);
    
    d0 = 32'h1234abcd;  
    d1 = 32'habcd1234;  
    d2 = 32'h11112222;  
    d3 = 32'h0;  
    s = 0;  
    #4 $finish;
end  
 
always #1 s = s + 1;  
 
mux4 #(WIDTH) uut(d0, d1, d2, d3, s, y);  
endmodule 