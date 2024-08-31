`timescale 1ps/1ps  
module mux2_tb;  
    parameter WIDTH = 32;  
    reg [WIDTH-1:0] d0, d1; 
    reg s; 
    wire [WIDTH-1:0] y;  
    
    initial begin  
        $dumpfile("mux2_tb.vcd");  
        $dumpvars(0, mux2_tb);

        d0 = 32'h1234abcd;  
        d1 = 32'habcd1234;  
        s = 0;  
        #2 $finish;
    end 

    always 
        #1 s = s + 1; 

    mux2 #(WIDTH) uut(d0, d1, s, y);  
endmodule  