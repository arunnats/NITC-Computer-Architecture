`timescale 1ps/1ps  
module signext_tb;  
    reg[15:0] a;  
    wire[31:0] y;  
    
    initial 
    begin  
        $dumpfile("signext_tb.vcd");  
        $dumpvars(0, signext_tb);

        a = 32'hfff3;  
        #1 a = 32'h004f;  
        #1 $finish;  
    end  
    
    signext uut(a, y);  
endmodule 