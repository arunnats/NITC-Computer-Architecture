`timescale 1ps/1ps  
module mux2_tb;  
    parameter WIDTH = 16;  // Set width to 16 bits
    reg [WIDTH-1:0] d0, d1;  
    reg s;  
    wire [WIDTH-1:0] y;  
    
    initial begin  
        $dumpfile("mux2_tb.vcd");  
        $dumpvars(0, mux2_tb);

        d0 = 16'h1234;  // 16-bit values
        d1 = 16'habcd;  
        s = 0;  
        #2 $finish;
    end 

    always 
        #1 s = s + 1;  // Toggle selection signal

    mux2 #(WIDTH) uut(d0, d1, s, y);  
endmodule  
