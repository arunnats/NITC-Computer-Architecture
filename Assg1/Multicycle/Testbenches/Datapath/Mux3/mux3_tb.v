`timescale 1ps/1ps  
module mux3_tb;  
    parameter WIDTH = 16;  // Set width to 16 bits
    reg [WIDTH-1:0] d0, d1, d2;  
    reg [1:0] s;  
    wire [WIDTH-1:0] y;  
    
    initial begin  
        $dumpfile("mux3_tb.vcd");  
        $dumpvars(0, mux3_tb);

        d0 = 16'h1234;  // 16-bit values
        d1 = 16'habcd;  
        d2 = 16'h1111;  
        s = 0;  
        #4 $finish;
    end  

    always #1 s = s + 1;  // Increment selection signal

    mux3 #(WIDTH) uut(d0, d1, d2, s, y);  
endmodule  
