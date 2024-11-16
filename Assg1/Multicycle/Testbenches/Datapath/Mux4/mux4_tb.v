`timescale 1ps/1ps  
module mux4_tb;  
    parameter WIDTH = 16;  // Set width to 16 bits
    reg [WIDTH-1:0] d0, d1, d2, d3;  
    reg [1:0] s;  
    wire [WIDTH-1:0] y;  
    
    initial begin  
        $dumpfile("mux4_tb.vcd");  
        $dumpvars(0, mux4_tb);

        d0 = 16'h1234;  // 16-bit values
        d1 = 16'habcd;  
        d2 = 16'h1111;  
        d3 = 16'h0;  
        s = 0;  
        #4 $finish;
    end  

    always #1 s = s + 1;  // Increment selection signal

    mux4 #(WIDTH) uut(d0, d1, d2, d3, s, y);  
endmodule  
