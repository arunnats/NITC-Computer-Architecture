module regfile_tb();  
    reg clk, we3;  
    reg [2:0] a1, a2, a3;  // 3-bit register addresses
    reg [15:0] wd3;        // 16-bit write data
    wire [15:0] rd1, rd2;  // 16-bit read data
    
    regfile UUT (clk, we3, a1, a2, a3, wd3, rd1, rd2);  
    
    initial begin  
        $dumpfile("regfile_tb.vcd");  
        $dumpvars(0, regfile_tb);

        // Initialize signals
        wd3 = 16'b0;  
        a1 = 3'b0;  
        a2 = 3'b0;  
        a3 = 3'b0;  
        we3 = 1'b0;  
        clk = 1'b0;  
        
        #100  
        we3 = 1'b1;  // Enable writing
        
        // Test cases with 16-bit data and 3-bit addresses
        #20  
        wd3 = 16'habcd;  // Write 16-bit data
        a1 = 3'b0;       // Read register 0
        a2 = 3'b0;       // Read register 0
        a3 = 3'b001;     // Write to register 1
        
        #20  
        wd3 = 16'h0123;  // Write 16-bit data
        a1 = 3'b001;     // Read register 1
        a2 = 3'b0;       // Read register 0
        a3 = 3'b010;     // Write to register 2
        
        #20  
        wd3 = 16'hcccc;  // Write 16-bit data
        a1 = 3'b010;     // Read register 2
        a2 = 3'b001;     // Read register 1
        a3 = 3'b011;     // Write to register 3
        
        #20  
        wd3 = 16'h3333;  // Write 16-bit data
        a1 = 3'b010;     // Read register 2
        a2 = 3'b011;     // Read register 3
        a3 = 3'b001;     // Write to register 1
        
        $finish;
    end  
    
    // Clock generation
    always begin  
        #10;  
        clk = ~clk;  
    end  
endmodule
