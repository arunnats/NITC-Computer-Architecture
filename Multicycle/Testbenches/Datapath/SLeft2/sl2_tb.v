`timescale 1ps/1ps  
module sl2_tb;  
    reg [31:0] a;  
    wire [31:0] y;  
    
    initial 
    begin  
        $dumpfile("sl2_tb.vcd"); 
        $dumpvars(0, sl2_tb);  

        a = 32'hffffffff;  
        #1 a = 32'habcd1234;  
        #1 $finish;  
    end  
    
    sl2 uut(a, y);  
endmodule  
