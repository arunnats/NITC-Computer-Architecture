`timescale 1ps/1ps  
module sl2_tb;  
    reg [15:0] a;         // 16-bit input
    wire [15:0] y;        // 16-bit output
    
    initial 
    begin  
        $dumpfile("sl2_tb.vcd"); 
        $dumpvars(0, sl2_tb);  

        a = 16'hffff;     // Test case with all bits set
        #1 a = 16'habcd;  // Test case with specific value
        #1 $finish;  
    end  
    
    sl2 uut(a, y);  
endmodule  
