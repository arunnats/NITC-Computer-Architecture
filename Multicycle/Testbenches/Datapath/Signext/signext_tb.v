`timescale 1ps/1ps  
module signext_tb;  
    reg[7:0] a;         // 8-bit input
    wire[15:0] y;       // 16-bit output
    
    initial 
    begin  
        $dumpfile("signext_tb.vcd");  
        $dumpvars(0, signext_tb);

        a = 8'hf3;      // Test sign extension of negative number
        #1 a = 8'h4f;   // Test sign extension of positive number
        #1 $finish;  
    end  
    
    signext uut(a, y);  
endmodule 
