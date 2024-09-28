`timescale 1ps/1ps  
module flopenr_tb;  
    parameter WIDTH = 16;  // Set width to 16 bits
    reg  clk, reset, en;  
    reg  [WIDTH-1:0] d;    // 16-bit data
    wire  [WIDTH-1:0] q;   // 16-bit output
    
    // Clock generation
    always begin  
        clk <= 1; #1; clk <= 0; #1;  
    end  
    
    // Testbench behavior
    initial begin  
        $dumpfile("flopenr_tb.vcd");  
        $dumpvars(0, flopenr_tb);

        // Initialize with 16-bit values
        d = 16'hABCD;  // Initial value of `d`
        reset = 0;  
        en = 1;  
        #2;  
        d = 16'h2005;  // Change `d` to another 16-bit value
        #2;  
        reset = 1;     // Trigger reset
        #2;  
        reset = 0;  
        en = 0;        // Disable writing to `q`
        #2 $finish;    // End simulation
    end

    // Instantiate the module under test (UUT)
    flopenr #(WIDTH) uut(clk, reset, en, d, q); 
endmodule 
